"""
Quick test to verify Neo4j connection and basic repository functionality
"""
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, 'src'))

from repositories.graph import Neo4jRepository, DocumentNode, ChunkNode, EntityNode

def test_basic_workflow():
    """Test basic Neo4j workflow"""
    print("Initializing Neo4j Repository...")
    repo = Neo4jRepository()
    
    try:
        # Clear database
        print("Clearing database...")
        repo.clear_all()
        
        # Test Document creation
        print("\n1. Testing Document Creation...")
        doc = DocumentNode(
            title="Test Resume",
            source="tests/Ingestion_pipline/Faizan_Jawaid_Resume_extracted.md",
            file_type="md"
        )
        doc_id = repo.create_document(doc)
        print(f"✓ Created document: {doc_id}")
        
        # Test Document retrieval
        print("\n2. Testing Document Retrieval...")
        retrieved_doc = repo.get_document(doc_id)
        print(f"✓ Retrieved document: {retrieved_doc.title}")
        
        # Test Chunk creation and relationship
        print("\n3. Testing Chunk Creation...")
        chunks_data = [
            "Faizan Jawaid - Senior Developer with 5 years experience",
            "Skills: Python, Java, Neo4j, FastAPI, Machine Learning",
            "Education: BS Computer Science from XYZ University"
        ]
        
        chunk_ids = []
        for i, content in enumerate(chunks_data):
            chunk = ChunkNode(
                content=content,
                chunk_index=i,
                tokens_count=len(content.split())
            )
            chunk_id = repo.create_chunk(chunk)
            chunk_ids.append(chunk_id)
            print(f"✓ Created chunk {i}: {chunk_id}")
            
            # Create CONTAINS relationship
            repo.create_relationship(doc_id, chunk_id, "CONTAINS")
            print(f"  → Created CONTAINS relationship")
        
        # Test Entity creation
        print("\n4. Testing Entity Creation...")
        entities = [
            ("Faizan Jawaid", "PERSON"),
            ("Python", "CONCEPT"),
            ("Neo4j", "CONCEPT"),
            ("XYZ University", "ORG")
        ]
        
        entity_ids = []
        for name, entity_type in entities:
            entity = EntityNode(name=name, type=entity_type)
            entity_id = repo.create_entity(entity)
            entity_ids.append(entity_id)
            print(f"✓ Created entity: {name} ({entity_type})")
            
            # Create MENTIONS relationship from first chunk
            repo.create_relationship(chunk_ids[0], entity_id, "MENTIONS")
        
        # Test graph queries
        print("\n5. Testing Graph Queries...")
        chunks = repo.get_chunks_by_document(doc_id)
        print(f"✓ Retrieved {len(chunks)} chunks from document")
        
        context = repo.get_document_context(chunk_ids[0])
        print(f"✓ Document context: {len(context['entities'])} entities mentioned")
        
        print("\n6. Testing Entity Search...")
        results = repo.search_entities_by_name("Python", limit=10)
        print(f"✓ Found {len(results)} entities matching 'Python'")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        repo.close()

if __name__ == "__main__":
    test_basic_workflow()
