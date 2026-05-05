"""
Tests for Neo4j Graph Database Repository
"""
import pytest
import os
from datetime import datetime

# Add parent directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from repositories.graph import (
    Neo4jRepository,
    DocumentNode,
    ChunkNode,
    EntityNode,
)


@pytest.fixture
def repo():
    """Fixture to provide Neo4j repository instance."""
    repo = Neo4jRepository()
    # Clear database before test
    repo.clear_all()
    yield repo
    # Cleanup after test
    repo.clear_all()
    repo.close()


class TestDocumentOperations:
    """Test Document CRUD operations."""
    
    def test_create_document(self, repo):
        """Test creating a document."""
        doc = DocumentNode(
            title="Test Document",
            source="/path/to/document.pdf",
            file_type="pdf"
        )
        doc_id = repo.create_document(doc)
        assert doc_id is not None
        assert len(doc_id) > 0
    
    def test_get_document(self, repo):
        """Test retrieving a document."""
        doc = DocumentNode(
            title="Test Document",
            source="/path/to/document.pdf",
            file_type="pdf"
        )
        doc_id = repo.create_document(doc)
        retrieved = repo.get_document(doc_id)
        
        assert retrieved is not None
        assert retrieved.title == "Test Document"
        assert retrieved.source == "/path/to/document.pdf"
        assert retrieved.file_type == "pdf"
    
    def test_update_document(self, repo):
        """Test updating a document."""
        doc = DocumentNode(
            title="Original Title",
            source="/path/to/document.pdf",
            file_type="pdf"
        )
        doc_id = repo.create_document(doc)
        
        # Update the document
        updated_doc = DocumentNode(
            id=doc_id,
            title="Updated Title",
            source="/path/to/document.pdf",
            file_type="pdf"
        )
        result = repo.update_document(doc_id, updated_doc)
        assert result is True
        
        # Verify update
        retrieved = repo.get_document(doc_id)
        assert retrieved.title == "Updated Title"
    
    def test_delete_document(self, repo):
        """Test deleting a document."""
        doc = DocumentNode(
            title="Test Document",
            source="/path/to/document.pdf",
            file_type="pdf"
        )
        doc_id = repo.create_document(doc)
        
        result = repo.delete_document(doc_id)
        assert result is True
        
        # Verify deletion
        retrieved = repo.get_document(doc_id)
        assert retrieved is None
    
    def test_list_documents(self, repo):
        """Test listing documents."""
        # Create multiple documents
        for i in range(3):
            doc = DocumentNode(
                title=f"Document {i}",
                source=f"/path/to/doc{i}.pdf",
                file_type="pdf"
            )
            repo.create_document(doc)
        
        docs = repo.list_documents(limit=100)
        assert len(docs) == 3


class TestChunkOperations:
    """Test Chunk CRUD operations."""
    
    def test_create_chunk(self, repo):
        """Test creating a chunk."""
        chunk = ChunkNode(
            content="This is a test chunk content.",
            chunk_index=0,
            tokens_count=5
        )
        chunk_id = repo.create_chunk(chunk)
        assert chunk_id is not None
    
    def test_get_chunk(self, repo):
        """Test retrieving a chunk."""
        chunk = ChunkNode(
            content="This is a test chunk content.",
            chunk_index=0,
            tokens_count=5
        )
        chunk_id = repo.create_chunk(chunk)
        retrieved = repo.get_chunk(chunk_id)
        
        assert retrieved is not None
        assert retrieved.content == "This is a test chunk content."
        assert retrieved.chunk_index == 0
        assert retrieved.tokens_count == 5
    
    def test_update_chunk(self, repo):
        """Test updating a chunk."""
        chunk = ChunkNode(
            content="Original content",
            chunk_index=0,
            tokens_count=5
        )
        chunk_id = repo.create_chunk(chunk)
        
        updated_chunk = ChunkNode(
            id=chunk_id,
            content="Updated content",
            chunk_index=0,
            tokens_count=10
        )
        result = repo.update_chunk(chunk_id, updated_chunk)
        assert result is True
        
        retrieved = repo.get_chunk(chunk_id)
        assert retrieved.content == "Updated content"
        assert retrieved.tokens_count == 10
    
    def test_delete_chunk(self, repo):
        """Test deleting a chunk."""
        chunk = ChunkNode(
            content="Test content",
            chunk_index=0,
            tokens_count=5
        )
        chunk_id = repo.create_chunk(chunk)
        
        result = repo.delete_chunk(chunk_id)
        assert result is True
        
        retrieved = repo.get_chunk(chunk_id)
        assert retrieved is None


class TestEntityOperations:
    """Test Entity CRUD operations."""
    
    def test_create_entity(self, repo):
        """Test creating an entity."""
        entity = EntityNode(
            name="John Doe",
            type="PERSON"
        )
        entity_id = repo.create_entity(entity)
        assert entity_id is not None
    
    def test_get_entity(self, repo):
        """Test retrieving an entity."""
        entity = EntityNode(
            name="John Doe",
            type="PERSON"
        )
        entity_id = repo.create_entity(entity)
        retrieved = repo.get_entity(entity_id)
        
        assert retrieved is not None
        assert retrieved.name == "John Doe"
        assert retrieved.type == "PERSON"
    
    def test_get_entity_by_name_and_type(self, repo):
        """Test retrieving entity by name and type."""
        entity = EntityNode(
            name="Apple Inc",
            type="ORG"
        )
        repo.create_entity(entity)
        
        retrieved = repo.get_entity_by_name_and_type("Apple Inc", "ORG")
        assert retrieved is not None
        assert retrieved.name == "Apple Inc"
        assert retrieved.type == "ORG"
    
    def test_update_entity(self, repo):
        """Test updating an entity."""
        entity = EntityNode(
            name="Original Name",
            type="PERSON"
        )
        entity_id = repo.create_entity(entity)
        
        updated_entity = EntityNode(
            id=entity_id,
            name="Updated Name",
            type="PERSON"
        )
        result = repo.update_entity(entity_id, updated_entity)
        assert result is True
        
        retrieved = repo.get_entity(entity_id)
        assert retrieved.name == "Updated Name"
    
    def test_delete_entity(self, repo):
        """Test deleting an entity."""
        entity = EntityNode(
            name="John Doe",
            type="PERSON"
        )
        entity_id = repo.create_entity(entity)
        
        result = repo.delete_entity(entity_id)
        assert result is True
        
        retrieved = repo.get_entity(entity_id)
        assert retrieved is None
    
    def test_list_entities(self, repo):
        """Test listing entities."""
        # Create entities of different types
        entities_data = [
            ("John Doe", "PERSON"),
            ("Jane Smith", "PERSON"),
            ("Apple Inc", "ORG"),
        ]
        
        for name, entity_type in entities_data:
            entity = EntityNode(name=name, type=entity_type)
            repo.create_entity(entity)
        
        # List all entities
        all_entities = repo.list_entities(limit=100)
        assert len(all_entities) == 3
        
        # List only PERSON entities
        persons = repo.list_entities(entity_type="PERSON", limit=100)
        assert len(persons) == 2
    
    def test_search_entities_by_name(self, repo):
        """Test searching entities by name."""
        entities_data = [
            ("John Doe", "PERSON"),
            ("Jane Smith", "PERSON"),
            ("Apple Inc", "ORG"),
        ]
        
        for name, entity_type in entities_data:
            entity = EntityNode(name=name, type=entity_type)
            repo.create_entity(entity)
        
        # Search for "John"
        results = repo.search_entities_by_name("John", limit=10)
        assert len(results) == 1
        assert results[0].name == "John Doe"


class TestRelationships:
    """Test relationship operations."""
    
    def test_create_relationship(self, repo):
        """Test creating a relationship between nodes."""
        # Create nodes
        doc = DocumentNode(title="Doc", source="path", file_type="pdf")
        doc_id = repo.create_document(doc)
        
        chunk = ChunkNode(content="content", chunk_index=0, tokens_count=5)
        chunk_id = repo.create_chunk(chunk)
        
        # Create relationship
        result = repo.create_relationship(doc_id, chunk_id, "CONTAINS")
        assert result is True
    
    def test_delete_relationship(self, repo):
        """Test deleting a relationship."""
        # Create nodes and relationship
        doc = DocumentNode(title="Doc", source="path", file_type="pdf")
        doc_id = repo.create_document(doc)
        
        chunk = ChunkNode(content="content", chunk_index=0, tokens_count=5)
        chunk_id = repo.create_chunk(chunk)
        
        repo.create_relationship(doc_id, chunk_id, "CONTAINS")
        
        # Delete relationship
        result = repo.delete_relationship(doc_id, chunk_id, "CONTAINS")
        assert result is True


class TestGraphQueries:
    """Test complex graph queries."""
    
    def test_get_chunks_by_document(self, repo):
        """Test retrieving all chunks of a document."""
        # Create document
        doc = DocumentNode(title="Test Doc", source="path", file_type="pdf")
        doc_id = repo.create_document(doc)
        
        # Create chunks
        for i in range(3):
            chunk = ChunkNode(
                content=f"Content {i}",
                chunk_index=i,
                tokens_count=5
            )
            chunk_id = repo.create_chunk(chunk)
            repo.create_relationship(doc_id, chunk_id, "CONTAINS")
        
        # Retrieve chunks
        chunks = repo.get_chunks_by_document(doc_id)
        assert len(chunks) == 3
        
        # Verify order
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_get_document_context(self, repo):
        """Test getting document context for a chunk."""
        # Create document
        doc = DocumentNode(title="Test Doc", source="path", file_type="pdf")
        doc_id = repo.create_document(doc)
        
        # Create chunk
        chunk = ChunkNode(content="Test content", chunk_index=0, tokens_count=5)
        chunk_id = repo.create_chunk(chunk)
        repo.create_relationship(doc_id, chunk_id, "CONTAINS")
        
        # Create entities
        entity1 = EntityNode(name="John", type="PERSON")
        entity1_id = repo.create_entity(entity1)
        repo.create_relationship(chunk_id, entity1_id, "MENTIONS")
        
        entity2 = EntityNode(name="Apple", type="ORG")
        entity2_id = repo.create_entity(entity2)
        repo.create_relationship(chunk_id, entity2_id, "MENTIONS")
        
        # Get context
        context = repo.get_document_context(chunk_id)
        assert context["document"] is not None
        assert context["chunk"] is not None
        assert len(context["entities"]) == 2
    
    def test_get_related_entities(self, repo):
        """Test getting related entities."""
        # Create entities
        entity1 = EntityNode(name="John", type="PERSON")
        entity1_id = repo.create_entity(entity1)
        
        entity2 = EntityNode(name="Jane", type="PERSON")
        entity2_id = repo.create_entity(entity2)
        
        entity3 = EntityNode(name="Bob", type="PERSON")
        entity3_id = repo.create_entity(entity3)
        
        # Create relationships
        repo.create_relationship(entity1_id, entity2_id, "RELATED_TO")
        repo.create_relationship(entity1_id, entity3_id, "RELATED_TO")
        
        # Get related entities
        related = repo.get_related_entities(entity1_id, limit=10)
        assert len(related) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
