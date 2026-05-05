"""
Base repository interface for graph database operations
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from .models import DocumentNode, ChunkNode, EntityNode, Relationship


class BaseGraphRepository(ABC):
    """Abstract base class for graph database repositories."""
    
    # Document operations
    @abstractmethod
    def create_document(self, document: DocumentNode) -> str:
        """Create a document node. Returns document ID."""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[DocumentNode]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def update_document(self, document_id: str, document: DocumentNode) -> bool:
        """Update a document node."""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its relationships."""
        pass
    
    @abstractmethod
    def list_documents(self, limit: int = 100, skip: int = 0) -> List[DocumentNode]:
        """List all documents with pagination."""
        pass
    
    # Chunk operations
    @abstractmethod
    def create_chunk(self, chunk: ChunkNode) -> str:
        """Create a chunk node. Returns chunk ID."""
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkNode]:
        """Get a chunk by ID."""
        pass
    
    @abstractmethod
    def update_chunk(self, chunk_id: str, chunk: ChunkNode) -> bool:
        """Update a chunk node."""
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk node."""
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> List[ChunkNode]:
        """Get all chunks of a document, ordered by chunk_index."""
        pass
    
    # Entity operations
    @abstractmethod
    def create_entity(self, entity: EntityNode) -> str:
        """Create an entity node. Returns entity ID."""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Get an entity by ID."""
        pass
    
    @abstractmethod
    def get_entity_by_name_and_type(self, name: str, entity_type: str) -> Optional[EntityNode]:
        """Get an entity by name and type (respects unique constraint)."""
        pass
    
    @abstractmethod
    def update_entity(self, entity_id: str, entity: EntityNode) -> bool:
        """Update an entity node."""
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity node."""
        pass
    
    @abstractmethod
    def list_entities(self, entity_type: Optional[str] = None, limit: int = 100) -> List[EntityNode]:
        """List entities, optionally filtered by type."""
        pass
    
    # Relationship operations
    @abstractmethod
    def create_relationship(self, source_id: str, target_id: str, rel_type: str, 
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two nodes."""
        pass
    
    @abstractmethod
    def delete_relationship(self, source_id: str, target_id: str, rel_type: str) -> bool:
        """Delete a relationship between two nodes."""
        pass
    
    # Graph query operations
    @abstractmethod
    def get_document_context(self, chunk_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get document context for a chunk (document -> chunks -> entities)."""
        pass
    
    @abstractmethod
    def search_entities_by_name(self, name: str, limit: int = 10) -> List[EntityNode]:
        """Search for entities by name (substring search)."""
        pass
    
    @abstractmethod
    def get_related_entities(self, entity_id: str, limit: int = 10) -> List[EntityNode]:
        """Get entities related to a given entity."""
        pass
    
    # Cleanup
    @abstractmethod
    def clear_all(self) -> bool:
        """Delete all nodes and relationships (for testing)."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass
