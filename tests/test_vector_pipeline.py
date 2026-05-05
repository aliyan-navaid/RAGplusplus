import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.services.text_splitter import split_markdown_by_words
from src.services.embeddings import EmbeddingModel
from src.repositories.vector.in_memory_vector_repository import InMemoryVectorRepository
from src.services.vector_indexer import index_markdown


def test_splitter_basic():
    md = ' '.join([f'word{i}' for i in range(1000)])
    chunks = split_markdown_by_words(md, chunk_size_words=300, overlap_words=50)
    assert len(chunks) >= 3
    # check overlap
    assert chunks[0]['meta']['end_word'] > chunks[1]['meta']['start_word']


def test_embedding_fallback_consistent():
    model = EmbeddingModel()
    a = model.embed(['hello world'])[0]
    b = model.embed(['hello world'])[0]
    # element-wise close check
    assert len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    assert len(a) > 0


def test_in_memory_repo_add_and_search():
    repo = InMemoryVectorRepository()
    model = EmbeddingModel()
    docs = [
        {'id': '1', 'text': 'apple banana', 'embedding': model.embed(['apple banana'])[0], 'metadata': {'doc': 'd1'}},
        {'id': '2', 'text': 'orange pear', 'embedding': model.embed(['orange pear'])[0], 'metadata': {'doc': 'd1'}},
    ]
    ids = repo.add_documents(docs)
    assert ids == ['1', '2']
    q_emb = model.embed(['apple'])[0]
    results = repo.search(q_emb, top_k=1)
    assert len(results) == 1
    top_id, score, meta, text = results[0]
    assert top_id == '1'


def test_end_to_end_index_and_search():
    repo = InMemoryVectorRepository()
    md = 'This is a test document. ' + ' '.join([f'chunkword{i}' for i in range(800)])
    ids = index_markdown(md, repo, chunk_size_words=200, overlap_words=20, model=EmbeddingModel())
    assert len(ids) >= 3
    # search for a phrase expected in first chunk
    model = EmbeddingModel()
    q_emb = model.embed(['test document'])[0]
    res = repo.search(q_emb, top_k=3)
    assert len(res) > 0
