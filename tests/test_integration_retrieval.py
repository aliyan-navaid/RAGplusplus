"""Integration tests using actual (non-mocked) components.

This test suite exercises the full retrieval pipeline with real ChromaDB,
real embedding models, real graph database (if available), and real Ollama LLM.
"""

import os
import sys
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

# Check for required dependencies
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import ollama
    HAS_OLLAMA_PACKAGE = True
except ImportError:
    HAS_OLLAMA_PACKAGE = False

from src.services.embeddings import EmbeddingModel
from src.repositories.vector.chroma_repository import ChromaVectorRepository
from src.services.vector_indexer import index_markdown
from src.retrieval.ensemble import EnsembleRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.prompt import PromptBuilder
from src.retrieval.ollama_client import OllamaClient
from src.retrieval.orchestrator import RetrievalOrchestrator

# Try to import Neo4j repository
try:
    from src.repositories.graph.neo4j_repository import Neo4jRepository
    HAS_NEO4J = True
except Exception:
    HAS_NEO4J = False


SAMPLE_MD_PATH = os.path.join(os.path.dirname(__file__), 'sample.md')
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_available_ollama_model():
    """Return a usable Phi-3 model name from the local Ollama registry."""
    if not HAS_OLLAMA_PACKAGE:
        return None

    try:
        client = ollama.Client()
        models = client.list()
    except Exception:
        return None

    if isinstance(models, dict):
        model_entries = models.get('models', [])
    else:
        model_entries = getattr(models, 'models', [])
    model_names = []
    for model in model_entries:
        if isinstance(model, dict):
            name = model.get('name') or model.get('model')
            if name:
                model_names.append(name)
        else:
            name = getattr(model, 'name', None) or getattr(model, 'model', None)
            if name:
                model_names.append(name)

    preferred_names = ('phi3:mini', 'phi3-mini', 'phi3')
    for preferred_name in preferred_names:
        for model_name in model_names:
            if model_name == preferred_name or model_name.startswith(preferred_name):
                return model_name

    return None


@pytest.fixture
def sample_markdown():
    """Load sample.md for indexing."""
    if not os.path.exists(SAMPLE_MD_PATH):
        pytest.skip(f"Sample markdown not found at {SAMPLE_MD_PATH}")
    with open(SAMPLE_MD_PATH, 'r') as f:
        return f.read()


@pytest.fixture
def clean_test_chroma(tmp_path):
    """Provide a clean Chroma directory for testing using pytest's tmp_path."""
    test_chroma_dir = tmp_path / "test_chroma"
    test_chroma_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_chroma_dir)
    # tmp_path is automatically cleaned up by pytest


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_integration_vector_indexing_and_search(sample_markdown, clean_test_chroma):
    """Test that we can index markdown into real Chroma and search it."""
    # Create real vector repository pointing to test directory
    repo = ChromaVectorRepository(persist_directory=clean_test_chroma)
    
    # Create real embedder
    embedder = EmbeddingModel(model_name='all-MiniLM-L6-v2')
    
    # Index the sample markdown
    doc_ids = index_markdown(sample_markdown, repo, model=embedder)
    
    assert len(doc_ids) > 0, "Should have indexed some documents"
    
    # Now query the repository with a real embedding
    query = "Adamjee Government Science College"
    query_embedding = embedder.embed([query])[0]
    
    results = repo.search(query_embedding, top_k=5)
    
    assert len(results) > 0, "Should find results for query about Adamjee"
    assert all(len(r) == 4 for r in results), "Each result should be (id, score, metadata, text)"


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_integration_ensemble_retriever(sample_markdown, clean_test_chroma):
    """Test ensemble retriever combining vector search (real) and optional graph."""
    # Create real vector repository
    repo = ChromaVectorRepository(persist_directory=clean_test_chroma)
    embedder = EmbeddingModel(model_name='all-MiniLM-L6-v2')
    
    # Index the sample
    index_markdown(sample_markdown, repo, model=embedder)
    
    # Create ensemble retriever (graph_repo will be None if Neo4j unavailable, which is fine)
    graph_repo = None
    if HAS_NEO4J:
        try:
            graph_repo = Neo4jRepository()
        except Exception:
            pass
    
    retriever = EnsembleRetriever(vector_repo=repo, graph_repo=graph_repo, embedder=embedder)
    
    # Retrieve results
    query = "recommendation systems"
    hits = retriever.retrieve(query, top_k=5)
    
    assert len(hits) > 0, "Should retrieve hits for 'recommendation systems'"
    assert all(isinstance(h, type(hits[0])) for h in hits), "All results should be same type"


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_integration_reranking_and_prompt(sample_markdown, clean_test_chroma):
    """Test reranking and prompt building with real retrieved results."""
    # Index
    repo = ChromaVectorRepository(persist_directory=clean_test_chroma)
    embedder = EmbeddingModel(model_name='all-MiniLM-L6-v2')
    index_markdown(sample_markdown, repo, model=embedder)
    
    # Retrieve
    retriever = EnsembleRetriever(vector_repo=repo, graph_repo=None, embedder=embedder)
    hits = retriever.retrieve("machine learning projects", top_k=5)
    
    # Rerank
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("machine learning projects", hits, top_k=5)
    
    assert len(reranked) > 0, "Should have reranked results"
    
    # Build prompt
    prompt_builder = PromptBuilder()
    built_prompt = prompt_builder.build("machine learning projects", reranked)
    
    assert built_prompt.prompt, "Prompt should not be empty"
    assert "machine learning" in built_prompt.prompt.lower() or len(built_prompt.prompt) > 50, \
        "Prompt should contain context"
    assert isinstance(built_prompt.citations, list), "Citations should be a list"


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
@pytest.mark.skipif(not HAS_OLLAMA_PACKAGE, reason="ollama package not installed")
def test_integration_end_to_end_with_ollama(sample_markdown, clean_test_chroma):
    """End-to-end integration test: index, retrieve, rerank, generate with real Ollama.
    
    This test exercises the complete pipeline with actual components:
    - Real ChromaDB for vector storage
    - Real sentence-transformers for embeddings
    - Real Ollama for LLM generation
    - Optional real Neo4j (skipped if unavailable)
    """
    # Check if Ollama is available and resolve the actual installed model name.
    ollama_model = get_available_ollama_model()
    if not ollama_model:
        pytest.skip("Ollama service not running or phi3 model not available")
    
    # Setup real components
    vector_repo = ChromaVectorRepository(persist_directory=clean_test_chroma)
    embedder = EmbeddingModel(model_name='all-MiniLM-L6-v2')
    
    # Index the sample document
    index_markdown(sample_markdown, vector_repo, model=embedder)
    
    # Build orchestrator with all real components
    graph_repo = None
    if HAS_NEO4J:
        try:
            graph_repo = Neo4jRepository()
        except Exception:
            pass
    
    orchestrator = RetrievalOrchestrator(
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        embedder=embedder,
        retriever=EnsembleRetriever(vector_repo=vector_repo, graph_repo=graph_repo, embedder=embedder),
        reranker=CrossEncoderReranker(),
        prompt_builder=PromptBuilder(),
        llm_client=OllamaClient(model=ollama_model),
        model_name=ollama_model,
    )
    
    # Execute query
    query = "What is Adamjee's educational background?"
    response = orchestrator.answer(query, top_k=5)
    
    # Verify response structure
    assert response.query == query, "Response query should match input"
    assert response.answer, "LLM should generate a non-empty answer"
    assert response.prompt, "Prompt should be constructed"
    assert isinstance(response.hits, list), "Hits should be a list"
    assert len(response.hits) > 0, "Should have retrieved hits"
    
    # Verify hits contain expected fields
    for hit in response.hits:
        assert hasattr(hit, 'id'), "Hit should have id"
        assert hasattr(hit, 'content'), "Hit should have content"
        assert hasattr(hit, 'score'), "Hit should have score"
    
    # Verify citations are formatted
    assert isinstance(response.citations, list), "Citations should be a list"
    
    # Print for manual inspection
    print(f"\n--- Integration Test Results ---")
    print(f"Query: {query}")
    print(f"Answer: {response.answer}")
    print(f"Number of hits: {len(response.hits)}")
    print(f"Top hit content preview: {response.hits[0].content[:100] if response.hits else 'N/A'}...")


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_integration_exact_phrase_matching(sample_markdown, clean_test_chroma):
    """Test that exact phrase matching works in real retrieval pipeline."""
    # Setup
    vector_repo = ChromaVectorRepository(persist_directory=clean_test_chroma)
    embedder = EmbeddingModel(model_name='all-MiniLM-L6-v2')
    index_markdown(sample_markdown, vector_repo, model=embedder)
    
    # Retrieve with ensemble (should combine vector + optional graph results)
    retriever = EnsembleRetriever(vector_repo=vector_repo, graph_repo=None, embedder=embedder)
    hits = retriever.retrieve("Adamjee Govt", top_k=10)
    
    # Rerank (should prioritize exact phrase matches)
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("Adamjee Govt", hits, top_k=10)
    
    assert len(reranked) > 0, "Should find results for 'Adamjee Govt'"
    
    # Check that top result contains the exact phrase
    top_result = reranked[0]
    assert "Adamjee" in top_result.content, \
        f"Top result should contain 'Adamjee', got: {top_result.content[:100]}"
