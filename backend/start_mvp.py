#!/usr/bin/env python3
"""
Legal Document Analyzer MVP - Quick Start Script
Simplified startup for hackathon demo without authentication.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment configuration."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found. Using default settings.")
        print("💡 For full functionality, copy .env.example to .env and configure your Google Cloud settings.")
    else:
        print("✅ Environment file found")
    
    return True

def start_server():
    """Start the FastAPI server in MVP mode."""
    print("🚀 Starting Legal Document Analyzer MVP...")
    print("📝 Running in NO-AUTH mode for prototype")
    print("🌐 Frontend should connect to: http://localhost:8080")
    print("📚 API Documentation: http://localhost:8080/docs")
    print("-" * 50)
    
    # Use the simplified API without authentication
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
    
    try:
        import uvicorn
        from src.api.mvp_api import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info",
            access_log=True
        )
    except ImportError:
        # Fallback to subprocess if imports fail
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.mvp_api:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload",
            "--log-level", "info"
        ]
        subprocess.run(cmd)

def main():
    """Main startup function."""
    print("🔧 Legal Document Analyzer MVP - Starting Up...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()