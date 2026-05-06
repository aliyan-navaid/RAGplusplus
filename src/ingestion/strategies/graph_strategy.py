"""
Graph RAG storage strategy.

Implements the strategy for saving documents to Knowledge Graph:
ConvertedDocument → Entity/Relation Extraction → Graph DB (Neo4j)

FLOW:
Input: ConvertedDocument (Markdown)
↓ [GraphStrategy.save()]
1. Extract entities from Markdown content
2. Extract relationships between entities
3. Store entities + relationships in Neo4j
4. Return success + metadata
Output: {success, doc_id, entities_created, relationships_created, ...}
"""

import logging
import hashlib
from typing import Optional
from datetime import datetime

from ..models import ConvertedDocument
from .base import BaseStorageStrategy
from src.repositories.graph.neo4j_repository import Neo4jRepository
from src.services.graph_indexer import index_markdown_graph

logger = logging.getLogger(__name__)


class GraphStrategy(BaseStorageStrategy):
    """
    Graph RAG storage strategy.
    
    This strategy:
    1. Takes ConvertedDocument with Markdown content
    2. Extracts entities and relationships from text
    3. Creates knowledge graph in Neo4j
    4. Links entities with document metadata
    
    INPUTS:
    - document: ConvertedDocument with markdown_content and sections
    
    OUTPUTS:
    - Dictionary with:
      - success: Whether save was successful
      - doc_id: Unique identifier for the document
      - entities_created: Number of entities extracted
      - relationships_created: Number of relationships created
    """
    
    def __init__(self, graph_uri: str = "bolt://localhost:7687"):
        """
        Initialize Graph RAG strategy.
        
        Args:
            graph_uri: Neo4j connection URI (default: local)
        """
        self.graph_uri = graph_uri
        self.repo = Neo4jRepository()
        logger.info(f"GraphStrategy initialized (URI: {graph_uri})")
    
    def save(self, document: ConvertedDocument) -> dict:
        """
        Save document to Knowledge Graph.
        
        FLOW:
        1. Generate unique document ID
        2. Extract entities from Markdown content
        3. Extract relationships between entities
        4. Create Document node in Neo4j
        5. Create Entity nodes
        6. Create relationships (CONTAINS, MENTIONS, etc.)
        7. Return metadata
        
        Args:
            document: ConvertedDocument to save
            
        Returns:
            Save result dictionary
        """
        try:
            logger.info(f"Graph Strategy: Saving document '{document.title or 'Untitled'}'")
            
            # Generate document ID
            doc_id = self._generate_doc_id(document)
            
            # Store in Neo4j using the shared graph indexer
            graph_result = index_markdown_graph(
                document.markdown_content,
                graph_repo=self.repo,
                source=document.source,
                title=document.title,
            )

            result = {
                "success": True,
                "doc_id": graph_result.get("doc_id", doc_id),
                "entities_created": graph_result.get("entities_created", 0),
                "relationships_created": graph_result.get("relationships_created", 0),
                "strategy": self.get_strategy_name(),
                "message": (
                    "Document saved to Graph store ("
                    f"{graph_result.get('entities_created', 0)} entities, "
                    f"{graph_result.get('relationships_created', 0)} relationships)"
                ),
            }
            
            logger.info(f"Graph Strategy: Save successful - {result}")
            return result
            
        except Exception as e:
            logger.error(f"Graph Strategy error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "strategy": self.get_strategy_name(),
                "message": f"Failed to save document: {str(e)}",
            }
    
    def _generate_doc_id(self, document: ConvertedDocument) -> str:
        """
        Generate unique document ID.
        
        Uses title + source + timestamp for uniqueness.
        """
        id_source = f"{document.title or 'doc'}_{document.source or 'unknown'}_{datetime.now().isoformat()}"
        doc_id = hashlib.md5(id_source.encode()).hexdigest()[:12]
        return doc_id
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Graph RAG"
