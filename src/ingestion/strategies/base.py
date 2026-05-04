"""
Abstract base class for storage strategies.

Defines the interface that all storage strategies implement.
Different strategies handle saving to RAG or Graph backends.

"""

from abc import ABC, abstractmethod
from typing import Optional

from ..models import ConvertedDocument


class BaseStorageStrategy(ABC):
    """
    Abstract base class for all storage strategies.
    
    Each strategy knows how to save a ConvertedDocument (Markdown format)
    to its specific backend (Vector DB for RAG, Knowledge Graph for Graph RAG).
    
    FLOW:
    Input: ConvertedDocument (Markdown-formatted)
    ↓ [strategy.save()]
    ↓ Process/transform for specific backend
    ↓ Save to backend (ChromaDB, Neo4j)
    Output: Success confirmation + metadata
    """

    
    @abstractmethod
    def save(self, document: ConvertedDocument) -> dict:
        """
        Save a converted document to the storage backend.
        
        Each strategy implements this differently:
        - RAGStrategy: Split text → embed → store in vector DB
        - GraphStrategy: Extract entities → create graph → store in graph DB
        
        Args:
            document: ConvertedDocument with Markdown content and metadata
            
        Returns:
            Dictionary with save results and metadata:
            {
                "success": bool,
                "doc_id": str,
                "chunks_created": int,  # For RAG
                "entities_created": int,  # For Graph
                "message": str,
            }
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get human-readable name of this strategy.
        
        Returns:
            Strategy name (e.g., "RAG", "Graph RAG")
        """
        pass
