#!/bin/bash

# Legal Document Analyzer - Deployment Script
# This script helps deploy the application to various platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --railway      Deploy backend to Railway"
    echo "  --vercel       Deploy frontend to Vercel"
    echo "  --docker       Build and run with Docker"
    echo "  --check        Check if all requirements are met"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --check      # Check deployment readiness"
    echo "  $0 --railway    # Deploy to Railway"
    echo "  $0 --vercel     # Deploy to Vercel"
    echo "  $0 --docker     # Run with Docker locally"
}

check_requirements() {
    print_info "Checking deployment requirements..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    # Check if all required files exist
    required_files=(
        "backend/simple_main.py"
        "backend/requirements.txt"
        "frontend/package.json"
        "DEPLOYMENT.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file missing: $file"
            exit 1
        fi
    done
    
    print_success "All requirements met!"
}

deploy_railway() {
    print_info "Deploying backend to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not found. Installing..."
        npm install -g @railway/cli
    fi
    
    print_info "Please follow these steps:"
    echo "1. Go to https://railway.app and create an account"
    echo "2. Run: railway login"
    echo "3. Run: railway link (select your project)"
    echo "4. Set environment variables in Railway dashboard:"
    echo "   - CORS_ORIGINS=[\"https://your-vercel-app.vercel.app\"]"
    echo "   - PORT=8080"
    echo "5. Railway will automatically deploy from your git repository"
    
    print_success "Railway deployment instructions provided!"
}

deploy_vercel() {
    print_info "Deploying frontend to Vercel..."
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        print_warning "Vercel CLI not found. Installing..."
        npm install -g vercel
    fi
    
    cd frontend
    
    print_info "Please follow these steps:"
    echo "1. Run: vercel login"
    echo "2. Run: vercel --prod"
    echo "3. Set environment variable:"
    echo "   - NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app"
    
    cd ..
    print_success "Vercel deployment instructions provided!"
}

deploy_docker() {
    print_info "Building and running with Docker..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_info "Building and starting services..."
    docker-compose -f docker-compose.prod.yml up --build -d
    
    print_success "Application running at:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend:  http://localhost:8080"
    echo ""
    echo "To stop: docker-compose -f docker-compose.prod.yml down"
}

commit_and_push() {
    print_info "Committing deployment configurations..."
    
    git add .
    git commit -m "Add deployment configurations

- Added Railway, Vercel, and Docker deployment configs
- Created production environment templates
- Added deployment documentation and scripts
- Ready for production deployment" || print_warning "Nothing to commit"
    
    git push origin main
    print_success "Changes pushed to repository!"
}

main() {
    case $1 in
        --check)
            check_requirements
            ;;
        --railway)
            check_requirements
            deploy_railway
            ;;
        --vercel)
            check_requirements
            deploy_vercel
            ;;
        --docker)
            check_requirements
            deploy_docker
            ;;
        --commit)
            commit_and_push
            ;;
        --help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

main "$@"