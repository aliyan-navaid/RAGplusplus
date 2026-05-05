import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# skip the test if chromadb isn't installed in the environment
pytest.importorskip('chromadb')

from src.repositories.vector.chroma_repository import ChromaVectorRepository
from src.services.embeddings import EmbeddingModel


def test_chroma_add_and_search(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    repo = ChromaVectorRepository(collection_name=f"test_collection_{tmp_path.name}", persist_directory=persist_dir)
    model = EmbeddingModel()

    texts = ["apple banana", "orange pear", "banana split"]
    docs = []
    for i, t in enumerate(texts):
        emb = model.embed([t])[0]
        docs.append({
            'id': f'd{i}',
            'text': t,
            'embedding': emb,
            'metadata': {'idx': i}
        })

    ids = repo.add_documents(docs)
    assert ids == [d['id'] for d in docs]

    q_emb = model.embed(['banana'])[0]
    results = repo.search(q_emb, top_k=2)
    assert len(results) == 2
    found_ids = [r[0] for r in results]
    assert 'd0' in found_ids or 'd2' in found_ids
