from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from src.retrieval.models import RetrievalHit


def _normalize_text(value: str) -> str:
    cleaned = []
    for char in (value or "").lower():
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _token_set(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return set(normalized.split()) if normalized else set()


class CrossEncoderReranker:
    """Re-ranks candidate passages using a cross-encoder when available.

    Falls back to an exact-phrase/token-overlap heuristic when the model
    cannot be loaded or inference fails.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _source_bias(self, source: str) -> float:
        source = (source or "").lower()
        if source == "vector":
            return 0.35
        if source == "graph_entity":
            return 0.12
        if source == "graph_related":
            return 0.05
        return 0.0

    def _source_rank(self, source: str) -> int:
        source = (source or "").lower()
        if source == "vector":
            return 3
        if source == "graph_entity":
            return 2
        if source == "graph_related":
            return 1
        return 0

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._model = None
        return self._model

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: Optional[int] = None) -> List[RetrievalHit]:
        if not hits:
            return []

        model = self._load_model()
        if model is not None:
            try:
                pairs = [(query, hit.content) for hit in hits]
                scores = model.predict(pairs)
                query_norm = _normalize_text(query)
                reranked = []
                for hit, score in zip(hits, scores):
                    doc_norm = _normalize_text(hit.content)
                    exact_phrase = 1.0 if query_norm and query_norm in doc_norm else 0.0
                    overlap = len(_token_set(query) & _token_set(hit.content)) / max(1, len(_token_set(query)))
                    combined = float(score) + self._source_bias(hit.source)
                    reranked.append((exact_phrase, overlap, self._source_rank(hit.source), replace(hit, score=combined)))
                reranked.sort(key=lambda item: (item[0], item[1], item[2], item[3].score), reverse=True)
                return [item[3] for item in reranked[:top_k]] if top_k else [item[3] for item in reranked]
            except Exception:
                pass

        query_norm = _normalize_text(query)
        query_tokens = _token_set(query)
        reranked: List[RetrievalHit] = []
        for hit in hits:
            doc_norm = _normalize_text(hit.content)
            doc_tokens = _token_set(hit.content)
            exact_phrase = 1.0 if query_norm and query_norm in doc_norm else 0.0
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            score = (2.0 * exact_phrase) + (1.5 * overlap) + hit.score + self._source_bias(hit.source)
            reranked.append((exact_phrase, overlap, self._source_rank(hit.source), replace(hit, score=score)))

        reranked.sort(key=lambda item: (item[0], item[1], item[2], item[3].score), reverse=True)
        ordered = [item[3] for item in reranked]
        return ordered[:top_k] if top_k else ordered
