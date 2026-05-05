"""Retrieval and reasoning layer for CiteCheck Pro."""

from .models import RetrievalHit, RetrievalResponse
from .ensemble import EnsembleRetriever
from .reranker import CrossEncoderReranker
from .prompt import PromptBuilder
from .ollama_client import OllamaClient
from .orchestrator import RetrievalOrchestrator, build_default_orchestrator

__all__ = [
    "RetrievalHit",
    "RetrievalResponse",
    "EnsembleRetriever",
    "CrossEncoderReranker",
    "PromptBuilder",
    "OllamaClient",
    "RetrievalOrchestrator",
    "build_default_orchestrator",
]
