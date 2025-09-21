#!/usr/bin/env python3
"""
Start script for the Legal Document Analyzer FastAPI backend
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from simple_main import app
        import uvicorn
        
        print("🚀 Starting Legal Document Analyzer API...")
        print("📄 API Documentation: http://localhost:8080/docs")
        print("🔧 ReDoc Documentation: http://localhost:8080/redoc")
        print("❤️  Health Check: http://localhost:8080/health")
        print()
        
        uvicorn.run(
            "simple_main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure FastAPI and Uvicorn are installed:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)