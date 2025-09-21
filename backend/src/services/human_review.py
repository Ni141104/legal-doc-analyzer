"""
Human Review Service
Manages attorney review queue and workflow for generated clause cards.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

# Google Cloud imports (will be properly imported in production)
try:
    from google.cloud import firestore
    from google.cloud import tasks_v2
    from google.auth import exceptions as auth_exceptions
except ImportError:
    firestore = None
    tasks_v2 = None
    auth_exceptions = None

from ..models.schemas import (
    ClauseCard, ReviewAssignment, ReviewTask, ReviewStatus, ReviewFeedback,
    Attorney
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class ReviewPriority(str, Enum):
    """Priority levels for review tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReviewWorkflowService:
    """Service for managing human review workflow."""
    
    def __init__(self):
        """Initialize the human review service."""
        self.db = None
        self.task_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients."""
        try:
            if firestore:
                self.db = firestore.Client(project=settings.GOOGLE_CLOUD_PROJECT_ID)
                logger.info("Firestore client initialized")
            
            if tasks_v2:
                self.task_client = tasks_v2.CloudTasksClient()
                logger.info("Cloud Tasks client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize review service clients: {str(e)}")
    
    async def submit_for_review(
        self, 
        clause_card: ClauseCard,
        document_id: str,
        reviewer_id: Optional[str] = None,
        priority: ReviewPriority = ReviewPriority.MEDIUM,
        deadline_hours: int = 24
    ) -> ReviewTask:
        """
        Submit a clause card for attorney review.
        
        Args:
            clause_card: ClauseCard to review
            document_id: ID of source document
            reviewer_id: Optional specific reviewer assignment
            priority: Priority level for review
            deadline_hours: Hours until review deadline
            
        Returns:
            ReviewTask with assignment details
        """
        try:
            # Create review task
            review_task = ReviewTask(
                task_id=f"review_{clause_card.clause_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                clause_card=clause_card,
                document_id=document_id,
                status=ReviewStatus.PENDING,
                priority=priority,
                submitted_at=datetime.utcnow(),
                deadline=datetime.utcnow() + timedelta(hours=deadline_hours),
                assigned_reviewer_id=reviewer_id,
                review_feedback=None
            )
            
            # Assign reviewer if not specified
            if not reviewer_id:
                assigned_reviewer = await self._assign_reviewer(review_task)
                review_task.assigned_reviewer_id = assigned_reviewer.attorney_id
            
            # Store in Firestore
            await self._store_review_task(review_task)
            
            # Create notification task
            await self._schedule_reviewer_notification(review_task)
            
            # Update metrics
            await self._update_review_metrics(document_id, "submitted")
            
            logger.info(
                f"Submitted clause {clause_card.clause_id} for review "
                f"(task: {review_task.task_id}, reviewer: {review_task.assigned_reviewer_id})"
            )
            
            return review_task
            
        except Exception as e:
            logger.error(f"Failed to submit clause card for review: {str(e)}")
            raise
    
    async def _assign_reviewer(self, review_task: ReviewTask) -> Attorney:
        """Assign the best available reviewer for a task."""
        try:
            # Get available attorneys
            available_attorneys = await self._get_available_attorneys()
            
            if not available_attorneys:
                # Fallback to default reviewer
                return Attorney(
                    attorney_id="default_reviewer",
                    name="Default Reviewer",
                    email="reviewer@example.com",
                    specializations=[review_task.clause_card.clause_type],
                    workload_capacity=100,
                    current_workload=0
                )
            
            # Score attorneys based on specialization and workload
            best_attorney = None
            best_score = -1
            
            for attorney in available_attorneys:
                score = self._calculate_assignment_score(attorney, review_task)
                if score > best_score:
                    best_score = score
                    best_attorney = attorney
            
            if not best_attorney:
                best_attorney = available_attorneys[0]  # Fallback
            
            return best_attorney
            
        except Exception as e:
            logger.error(f"Reviewer assignment failed: {str(e)}")
            # Return default reviewer on error
            return Attorney(
                attorney_id="error_reviewer",
                name="Error Reviewer",
                email="error@example.com",
                specializations=[],
                workload_capacity=1,
                current_workload=0
            )
    
    def _calculate_assignment_score(self, attorney: Attorney, review_task: ReviewTask) -> float:
        """Calculate assignment score for attorney-task pair."""
        try:
            score = 0.0
            
            # Specialization match (40% weight)
            if review_task.clause_card.clause_type in attorney.specializations:
                score += 0.4
            elif any(spec in review_task.clause_card.clause_type.lower() 
                    for spec in attorney.specializations):
                score += 0.2
            
            # Workload factor (40% weight)
            if attorney.workload_capacity > 0:
                workload_ratio = attorney.current_workload / attorney.workload_capacity
                score += 0.4 * (1.0 - min(workload_ratio, 1.0))
            
            # Priority factor (20% weight)
            priority_weights = {
                ReviewPriority.URGENT: 1.0,
                ReviewPriority.HIGH: 0.8,
                ReviewPriority.MEDIUM: 0.6,
                ReviewPriority.LOW: 0.4
            }
            score += 0.2 * priority_weights.get(review_task.priority, 0.5)
            
            return score
            
        except Exception as e:
            logger.error(f"Assignment scoring failed: {str(e)}")
            return 0.0
    
    async def _get_available_attorneys(self) -> List[Attorney]:
        """Get list of available attorneys."""
        try:
            if not self.db:
                return []
            
            # Query attorneys from Firestore
            attorneys_ref = self.db.collection('attorneys')
            docs = attorneys_ref.where('available', '==', True).stream()
            
            attorneys = []
            for doc in docs:
                data = doc.to_dict()
                attorney = Attorney(
                    attorney_id=doc.id,
                    name=data.get('name', 'Unknown'),
                    email=data.get('email', 'unknown@example.com'),
                    specializations=data.get('specializations', []),
                    workload_capacity=data.get('workload_capacity', 10),
                    current_workload=data.get('current_workload', 0)
                )
                attorneys.append(attorney)
            
            return attorneys
            
        except Exception as e:
            logger.error(f"Failed to get available attorneys: {str(e)}")
            return []
    
    async def _store_review_task(self, review_task: ReviewTask):
        """Store review task in Firestore."""
        try:
            if not self.db:
                logger.warning("Firestore not available, skipping task storage")
                return
            
            # Convert to dict for storage
            task_data = {
                'task_id': review_task.task_id,
                'clause_id': review_task.clause_card.clause_id,
                'document_id': review_task.document_id,
                'status': review_task.status.value,
                'priority': review_task.priority.value,
                'submitted_at': review_task.submitted_at,
                'deadline': review_task.deadline,
                'assigned_reviewer_id': review_task.assigned_reviewer_id,
                'clause_data': review_task.clause_card.dict(),
                'created_at': datetime.utcnow()
            }
            
            # Store in Firestore
            doc_ref = self.db.collection('review_tasks').document(review_task.task_id)
            doc_ref.set(task_data)
            
            logger.info(f"Stored review task {review_task.task_id} in Firestore")
            
        except Exception as e:
            logger.error(f"Failed to store review task: {str(e)}")
    
    async def _schedule_reviewer_notification(self, review_task: ReviewTask):
        """Schedule notification to assigned reviewer."""
        try:
            if not self.task_client:
                logger.warning("Cloud Tasks not available, skipping notification")
                return
            
            # Create notification task
            notification_payload = {
                'task_id': review_task.task_id,
                'reviewer_id': review_task.assigned_reviewer_id,
                'clause_id': review_task.clause_card.clause_id,
                'priority': review_task.priority.value,
                'deadline': review_task.deadline.isoformat()
            }
            
            # In production, this would create a Cloud Task
            # For now, just log the notification
            logger.info(
                f"Scheduled notification for reviewer {review_task.assigned_reviewer_id} "
                f"about task {review_task.task_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to schedule reviewer notification: {str(e)}")
    
    async def get_review_queue(
        self, 
        reviewer_id: Optional[str] = None,
        status: Optional[ReviewStatus] = None,
        priority: Optional[ReviewPriority] = None
    ) -> List[ReviewTask]:
        """
        Get review queue with optional filters.
        
        Args:
            reviewer_id: Filter by assigned reviewer
            status: Filter by review status
            priority: Filter by priority level
            
        Returns:
            List of matching review tasks
        """
        try:
            if not self.db:
                return []
            
            # Build query
            query = self.db.collection('review_tasks')
            
            if reviewer_id:
                query = query.where('assigned_reviewer_id', '==', reviewer_id)
            
            if status:
                query = query.where('status', '==', status.value)
            
            if priority:
                query = query.where('priority', '==', priority.value)
            
            # Order by deadline and priority
            query = query.order_by('deadline')
            
            docs = query.stream()
            
            review_tasks = []
            for doc in docs:
                data = doc.to_dict()
                
                # Reconstruct clause card
                clause_data = data.get('clause_data', {})
                clause_card = ClauseCard(**clause_data) if clause_data else None
                
                if clause_card:
                    review_task = ReviewTask(
                        task_id=data['task_id'],
                        clause_card=clause_card,
                        document_id=data['document_id'],
                        status=ReviewStatus(data['status']),
                        priority=ReviewPriority(data['priority']),
                        submitted_at=data['submitted_at'],
                        deadline=data['deadline'],
                        assigned_reviewer_id=data['assigned_reviewer_id'],
                        review_feedback=data.get('review_feedback')
                    )
                    review_tasks.append(review_task)
            
            logger.info(f"Retrieved {len(review_tasks)} review tasks from queue")
            return review_tasks
            
        except Exception as e:
            logger.error(f"Failed to get review queue: {str(e)}")
            return []
    
    async def submit_review_feedback(
        self, 
        task_id: str,
        reviewer_id: str,
        feedback: ReviewFeedback
    ) -> bool:
        """
        Submit review feedback for a task.
        
        Args:
            task_id: Review task ID
            reviewer_id: ID of reviewing attorney
            feedback: Review feedback with corrections
            
        Returns:
            True if feedback submitted successfully
        """
        try:
            if not self.db:
                logger.warning("Firestore not available, cannot submit feedback")
                return False
            
            # Get task document
            task_ref = self.db.collection('review_tasks').document(task_id)
            task_doc = task_ref.get()
            
            if not task_doc.exists:
                logger.error(f"Review task {task_id} not found")
                return False
            
            # Verify reviewer assignment
            task_data = task_doc.to_dict()
            if task_data.get('assigned_reviewer_id') != reviewer_id:
                logger.error(f"Reviewer {reviewer_id} not assigned to task {task_id}")
                return False
            
            # Update task with feedback
            update_data = {
                'status': ReviewStatus.COMPLETED.value,
                'review_feedback': feedback.dict(),
                'completed_at': datetime.utcnow(),
                'reviewed_by': reviewer_id
            }
            
            task_ref.update(update_data)
            
            # Update reviewer workload
            await self._update_reviewer_workload(reviewer_id, -1)
            
            # Update metrics
            await self._update_review_metrics(task_data['document_id'], "completed")
            
            # If major issues found, schedule re-generation
            if feedback.approved == False and feedback.severity == "high":
                await self._schedule_regeneration(task_id, feedback)
            
            logger.info(
                f"Review feedback submitted for task {task_id} by {reviewer_id} "
                f"(approved: {feedback.approved})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit review feedback: {str(e)}")
            return False
    
    async def _update_reviewer_workload(self, reviewer_id: str, delta: int):
        """Update reviewer's current workload."""
        try:
            if not self.db:
                return
            
            attorney_ref = self.db.collection('attorneys').document(reviewer_id)
            attorney_doc = attorney_ref.get()
            
            if attorney_doc.exists:
                current_workload = attorney_doc.to_dict().get('current_workload', 0)
                new_workload = max(0, current_workload + delta)
                
                attorney_ref.update({'current_workload': new_workload})
                logger.info(f"Updated reviewer {reviewer_id} workload: {new_workload}")
            
        except Exception as e:
            logger.error(f"Failed to update reviewer workload: {str(e)}")
    
    async def _update_review_metrics(self, document_id: str, action: str):
        """Update review metrics for analytics."""
        try:
            if not self.db:
                return
            
            metrics_ref = self.db.collection('review_metrics').document(document_id)
            metrics_doc = metrics_ref.get()
            
            if metrics_doc.exists:
                metrics = metrics_doc.to_dict()
            else:
                metrics = {'document_id': document_id}
            
            # Update action count
            action_key = f"{action}_count"
            metrics[action_key] = metrics.get(action_key, 0) + 1
            metrics['last_updated'] = datetime.utcnow()
            
            metrics_ref.set(metrics)
            
        except Exception as e:
            logger.error(f"Failed to update review metrics: {str(e)}")
    
    async def _schedule_regeneration(self, task_id: str, feedback: ReviewFeedback):
        """Schedule re-generation of clause card based on feedback."""
        try:
            # In production, this would trigger the generation pipeline
            logger.info(
                f"Scheduling re-generation for task {task_id} due to feedback: "
                f"{feedback.corrections}"
            )
            
            # Could create a Cloud Task to trigger re-generation
            # with the feedback as input for improvement
            
        except Exception as e:
            logger.error(f"Failed to schedule regeneration: {str(e)}")
    
    async def get_review_analytics(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get review process analytics.
        
        Args:
            start_date: Start date for analytics window
            end_date: End date for analytics window
            
        Returns:
            Dictionary with review analytics
        """
        try:
            if not self.db:
                return {}
            
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Query review tasks in date range
            tasks_ref = self.db.collection('review_tasks')
            query = tasks_ref.where('submitted_at', '>=', start_date)\
                            .where('submitted_at', '<=', end_date)
            
            docs = query.stream()
            
            # Calculate analytics
            total_tasks = 0
            completed_tasks = 0
            approved_tasks = 0
            avg_review_time = 0
            priority_distribution = {}
            reviewer_performance = {}
            
            review_times = []
            
            for doc in docs:
                data = doc.to_dict()
                total_tasks += 1
                
                status = data.get('status')
                priority = data.get('priority', 'medium')
                reviewer_id = data.get('assigned_reviewer_id')
                
                # Priority distribution
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
                
                if status == ReviewStatus.COMPLETED.value:
                    completed_tasks += 1
                    
                    # Check if approved
                    feedback = data.get('review_feedback', {})
                    if feedback.get('approved'):
                        approved_tasks += 1
                    
                    # Calculate review time
                    submitted_at = data.get('submitted_at')
                    completed_at = data.get('completed_at')
                    
                    if submitted_at and completed_at:
                        review_time = (completed_at - submitted_at).total_seconds() / 3600  # hours
                        review_times.append(review_time)
                        
                        # Reviewer performance
                        if reviewer_id not in reviewer_performance:
                            reviewer_performance[reviewer_id] = {
                                'total': 0, 'approved': 0, 'avg_time': 0
                            }
                        
                        reviewer_performance[reviewer_id]['total'] += 1
                        if feedback.get('approved'):
                            reviewer_performance[reviewer_id]['approved'] += 1
            
            # Calculate averages
            if review_times:
                avg_review_time = sum(review_times) / len(review_times)
            
            for reviewer_id in reviewer_performance:
                perf = reviewer_performance[reviewer_id]
                perf['approval_rate'] = perf['approved'] / max(perf['total'], 1)
            
            completion_rate = completed_tasks / max(total_tasks, 1)
            approval_rate = approved_tasks / max(completed_tasks, 1)
            
            analytics = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'summary': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'approved_tasks': approved_tasks,
                    'completion_rate': completion_rate,
                    'approval_rate': approval_rate,
                    'avg_review_time_hours': avg_review_time
                },
                'priority_distribution': priority_distribution,
                'reviewer_performance': reviewer_performance,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated review analytics for {total_tasks} tasks")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate review analytics: {str(e)}")
            return {}


# Global service instance
human_review_service = ReviewWorkflowService()