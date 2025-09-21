"""
API package for the Legal Document Analyzer.
"""

from .main import app
from .orchestrator import DocumentOrchestrator

__all__ = ["app", "DocumentOrchestrator"]