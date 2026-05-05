from typing import List, Dict, Any, Tuple
import math
from .base_repository import VectorRepository


class InMemoryVectorRepository(VectorRepository):
    """A simple in-memory vector repository for testing and local dev.

    Stores embeddings in RAM and performs cosine-similarity search.
    """

    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        ids = []
        for d in docs:
            if 'embedding' not in d:
                raise ValueError('Document missing embedding')
            self._items.append(dict(d))
            ids.append(d.get('id'))
        return ids

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        # pure-python cosine similarity
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    def search(self, query_embedding, top_k: int = 5, metadata_filter: dict | None = None) -> List[Tuple[str, float, Dict[str, Any], str]]:
        q = list(map(float, query_embedding))
        scores = []
        for item in self._items:
            # apply metadata filter if provided
            if metadata_filter:
                md = item.get('metadata') or {}
                # require all keys to match exactly
                ok = True
                for k, v in metadata_filter.items():
                    if md.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            emb = list(map(float, item['embedding']))
            score = self._cosine_sim(q, emb)
            scores.append((item.get('id'), score, item.get('metadata'), item.get('text')))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
