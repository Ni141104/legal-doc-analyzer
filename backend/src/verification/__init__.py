"""
Verification package for the Legal Document Analyzer.
"""

from .nli_verifier import (
    EntailmentVerifier,
    VerificationPipeline,
    VerificationResult,
    VerificationReport,
    EntailmentLabel,
    NLIModel,
    HuggingFaceNLIModel,
    VertexAINLIModel,
    RuleBasedNLIModel,
    create_verifier
)

__all__ = [
    "EntailmentVerifier",
    "VerificationPipeline",
    "VerificationResult",
    "VerificationReport",
    "EntailmentLabel",
    "NLIModel",
    "HuggingFaceNLIModel",
    "VertexAINLIModel", 
    "RuleBasedNLIModel",
    "create_verifier"
]