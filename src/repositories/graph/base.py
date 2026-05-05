"""Base repository interface for graph database operations."""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from .models import Document, Chunk, Entity, GraphRelationship


class BaseGraphRepository(ABC):
    """Abstract base class for graph database repositories."""
    
    # Document operations
    @abstractmethod
    def create_document(self, document: Document) -> Document:
        """Create a document node."""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its relationships."""
        pass
    
    # Chunk operations
    @abstractmethod
    def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a chunk node."""
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, doc_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        pass
    
    # Entity operations
    @abstractmethod
    def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node."""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        pass
    
    @abstractmethod
    def get_or_create_entity(self, name: str, entity_type: str) -> Entity:
        """Get existing entity or create new one."""
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        pass
    
    # Relationship operations
    @abstractmethod
    def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between two nodes."""
        pass
    
    @abstractmethod
    def delete_relationship(self, from_id: str, to_id: str, rel_type: str) -> bool:
        """Delete a relationship."""
        pass
    
    # Query operations
    @abstractmethod
    def get_document_context(self, doc_id: str) -> Dict[str, Any]:
        """Get document with all its chunks and entities."""
        pass
    
    @abstractmethod
    def find_related_entities(self, entity_id: str, depth: int = 1) -> List[Entity]:
        """Find entities related to a given entity."""
        pass
    
    @abstractmethod
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name."""
        pass
    
    # Cleanup
    @abstractmethod
    def clear_all(self) -> bool:
        """Delete all nodes and relationships (for testing)."""
        pass
