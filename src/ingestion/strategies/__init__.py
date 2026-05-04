"""
Storage strategies for ingested documents.

Each strategy consumes the same ConvertedDocument format and saves it
appropriately for its target storage system.
"""

from .base import BaseStorageStrategy
from .rag_strategy import RAGStrategy
from .graph_strategy import GraphStrategy
from .strategy_factory import StorageStrategyFactory

__all__ = [
    "BaseStorageStrategy",
    "RAGStrategy", 
    "GraphStrategy",
    "StorageStrategyFactory",
]
