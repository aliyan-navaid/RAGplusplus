from typing import List, Dict, Any
import uuid

from .text_splitter import split_markdown_by_words
from .embeddings import EmbeddingModel


def index_markdown(markdown: str, repo, chunk_size_words: int = 400, overlap_words: int = 50, model: EmbeddingModel = None) -> List[str]:
    """Split markdown, embed chunks, and store into a vector repository.

    repo: instance of VectorRepository
    model: EmbeddingModel instance (optional)
    Returns list of stored ids.
    """
    if model is None:
        model = EmbeddingModel()

    chunks = split_markdown_by_words(markdown, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
    texts = [c['text'] for c in chunks]
    embeddings = model.embed(texts)

    docs = []
    for c, emb in zip(chunks, embeddings):
        doc_id = str(uuid.uuid4())
        docs.append({
            'id': doc_id,
            'text': c['text'],
            'embedding': emb,
            'metadata': c.get('meta', {})
        })

    return repo.add_documents(docs)
