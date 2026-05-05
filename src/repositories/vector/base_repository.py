from typing import List, Dict, Any


class VectorRepository:
    """Abstract repository interface for a vector database.

    Implementations must provide `add_documents` and `search`.
    """

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store.

        Each doc should be a dict with keys: `id`, `text`, `embedding`, `metadata`.
        Returns list of stored ids.
        """
        raise NotImplementedError()

    def search(self, query_embedding, top_k: int = 5):
        """Search the vector store by embedding.

        Args:
            query_embedding: embedding vector for query
            top_k: number of results to return
            metadata_filter: optional dict of metadata to filter results (exact match)

        Returns list of (id, score, metadata, text)."""
        raise NotImplementedError()
