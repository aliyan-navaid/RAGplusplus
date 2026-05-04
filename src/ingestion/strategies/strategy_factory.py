"""
Storage strategy factory.

Factory pattern for creating appropriate storage strategies
based on configuration or user choice.

Usage:
    factory = StorageStrategyFactory()
    rag_strategy = factory.create("rag")
    graph_strategy = factory.create("graph")
"""

import logging
from typing import Union

from .base import BaseStorageStrategy
from .rag_strategy import RAGStrategy
from .graph_strategy import GraphStrategy

logger = logging.getLogger(__name__)


class StorageStrategyFactory:
    """
    Factory for creating storage strategy instances.
    
    Supports:
    - "rag" or "RAG": Create RAGStrategy (Vector DB)
    - "graph" or "Graph RAG": Create GraphStrategy (Knowledge Graph)
    
    PATTERN: Factory Pattern
    """
    
    _strategies = {
        "rag": RAGStrategy,
        "graph": GraphStrategy,
        "graph_rag": GraphStrategy,
    }
    
    @classmethod
    def create(cls, strategy_type: str) -> BaseStorageStrategy:
        """
        Create a storage strategy instance.
        
        Args:
            strategy_type: Strategy type ("rag" or "graph")
            
        Returns:
            BaseStorageStrategy implementation
            
        Raises:
            ValueError: If strategy type is unknown
        """
        strategy_type_lower = strategy_type.lower().strip()
        
        if strategy_type_lower not in cls._strategies:
            supported = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Supported: {supported}"
            )
        
        strategy_class = cls._strategies[strategy_type_lower]
        logger.info(f"Creating strategy: {strategy_class.__name__}")
        
        return strategy_class()
    
    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: type
    ) -> None:
        """
        Register a custom strategy class.
        
        Allows extending factory with custom strategies.
        
        Args:
            name: Strategy name (e.g., "custom")
            strategy_class: Class implementing BaseStorageStrategy
        """
        if not issubclass(strategy_class, BaseStorageStrategy):
            raise TypeError(
                f"{strategy_class} must inherit from BaseStorageStrategy"
            )
        
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered custom strategy: {name}")
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
