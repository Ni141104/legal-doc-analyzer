"""
Legal Document Analyzer Backend
Production-ready system for analyzing legal documents with anti-hallucination measures.
"""

__version__ = "1.0.0"
__author__ = "Legal Document Analyzer Team"

from .api import app
from .models import *
from .extractors import *
from .retrieval import *
from .generation import *
from .verification import *

__all__ = ["app"]