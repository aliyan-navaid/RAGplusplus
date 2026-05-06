from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.retrieval.models import RetrievalHit


def _normalize_text(value: str) -> str:
    cleaned = []
    for char in (value or "").lower():
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _token_set(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return set(normalized.split()) if normalized else set()


class EnsembleRetriever:
    """Combines vector retrieval with graph retrieval into a single ranked list."""

    def __init__(self, vector_repo: Any, graph_repo: Any = None, embedder: Any = None):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.embedder = embedder

    def _embed_query(self, query: str) -> List[float]:
        if self.embedder is None:
            raise ValueError("An embedding model is required for vector retrieval")
        return self.embedder.embed([query])[0]

    def _vector_hits(self, query: str, top_k: int, metadata_filter: dict | None = None) -> List[RetrievalHit]:
        query_embedding = self._embed_query(query)
        raw_hits = self.vector_repo.search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
        hits: List[RetrievalHit] = []
        for item_id, distance, metadata, document in raw_hits:
            score = 1.0 / (1.0 + float(distance))
            hits.append(
                RetrievalHit(
                    id=str(item_id),
                    content=str(document or ""),
                    score=score,
                    source="vector",
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def _graph_terms(self, query: str) -> List[str]:
        tokens = [token for token in _token_set(query) if len(token) >= 3]
        lower_query = (query or "").lower()

        # Expand for common intent words so graph entities (e.g., University, College)
        # can still be matched even when the query doesn't name them explicitly.
        expansions: List[str] = []
        if any(word in lower_query for word in ["study", "studied", "education", "school", "university", "college", "degree"]):
            expansions.extend(["university", "college", "school", "bachelor", "degree", "education"])
        if any(word in lower_query for word in ["work", "experience", "job", "intern", "internship"]):
            expansions.extend(["company", "project", "experience", "internship"])
        if any(word in lower_query for word in ["skill", "skills", "technology", "tech", "tools"]):
            expansions.extend(["skills", "languages", "tools", "technologies"])

        combined = tokens + expansions
        # De-duplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for tok in combined:
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(tok)

        ordered.sort(key=len, reverse=True)
        return ordered[:8]

    def _graph_hits(self, query: str, top_k: int) -> List[RetrievalHit]:
        if self.graph_repo is None:
            return []

        query_tokens = _token_set(query)
        hits: List[RetrievalHit] = []
        seen_ids: set[str] = set()

        for term in self._graph_terms(query):
            try:
                entities = self.graph_repo.search_entities_by_name(term, limit=top_k)
            except Exception:
                continue

            for entity in entities:
                entity_id = getattr(entity, "id", None) or str(getattr(entity, "name", term))
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)

                name = getattr(entity, "name", term)
                entity_type = getattr(entity, "type", "ENTITY")
                entity_tokens = _token_set(name)
                overlap = len(query_tokens & entity_tokens) / max(1, len(query_tokens))
                hits.append(
                    RetrievalHit(
                        id=str(entity_id),
                        content=f"Entity: {name} ({entity_type})",
                        score=0.65 + 0.35 * overlap,
                        source="graph_entity",
                        metadata={"entity_name": name, "entity_type": entity_type},
                    )
                )

                try:
                    related_entities = self.graph_repo.get_related_entities(entity_id, limit=top_k)
                except Exception:
                    related_entities = []

                for related in related_entities:
                    related_id = getattr(related, "id", None) or str(getattr(related, "name", "related"))
                    if related_id in seen_ids:
                        continue
                    seen_ids.add(related_id)
                    related_name = getattr(related, "name", "")
                    related_type = getattr(related, "type", "ENTITY")
                    related_tokens = _token_set(related_name)
                    related_overlap = len(query_tokens & related_tokens) / max(1, len(query_tokens))
                    hits.append(
                        RetrievalHit(
                            id=str(related_id),
                            content=f"Related entity: {related_name} ({related_type}) via {name}",
                            score=0.45 + 0.25 * related_overlap,
                            source="graph_related",
                            metadata={"entity_name": related_name, "entity_type": related_type, "via": name},
                        )
                    )

        return hits
        
    def _fallback_graph_hits(self, query: str, top_k: int) -> List[RetrievalHit]:
        if self.graph_repo is None:
            return []
        lowered = (query or "").lower()
        entity_type = None
        if any(word in lowered for word in ["study", "studied", "education", "school", "university", "college", "degree"]):
            entity_type = "ORG"
        try:
            entities = self.graph_repo.list_entities(entity_type=entity_type, limit=top_k)
        except Exception:
            return []

        hits: List[RetrievalHit] = []
        for entity in entities:
            name = getattr(entity, "name", "")
            entity_type_value = getattr(entity, "type", "ENTITY")
            hits.append(
                RetrievalHit(
                    id=str(getattr(entity, "id", name)),
                    content=f"Entity: {name} ({entity_type_value})",
                    score=0.25,
                    source="graph_entity",
                    metadata={"entity_name": name, "entity_type": entity_type_value, "fallback": True},
                )
            )
        return hits

    def _merge_hits(self, hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
        merged: Dict[str, RetrievalHit] = {}
        for hit in hits:
            key = f"{hit.source}:{hit.id}"
            existing = merged.get(key)
            if existing is None or hit.score > existing.score:
                merged[key] = hit
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def _dedupe_preserve_order(self, hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
        seen: set[str] = set()
        ordered: List[RetrievalHit] = []
        for hit in hits:
            key = f"{hit.source}:{hit.id}"
            if key in seen:
                continue
            seen.add(key)
            ordered.append(hit)
        return ordered

    def _balanced_hits(self, vector_hits: List[RetrievalHit], graph_hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        if not graph_hits:
            return vector_hits[:top_k]
        if not vector_hits:
            return graph_hits[:top_k]

        vector_quota = max(1, top_k // 2)
        graph_quota = max(1, top_k - vector_quota)

        selected = list(vector_hits[:vector_quota]) + list(graph_hits[:graph_quota])
        if len(selected) < top_k:
            selected.extend(vector_hits[vector_quota:])
            selected.extend(graph_hits[graph_quota:])

        return self._dedupe_preserve_order(selected)[:top_k]

    def retrieve(self, query: str, top_k: int = 5, vector_top_k: Optional[int] = None, graph_top_k: Optional[int] = None, metadata_filter: dict | None = None) -> List[RetrievalHit]:
        vector_top_k = vector_top_k or top_k
        graph_top_k = graph_top_k or top_k
        vector_hits = self._vector_hits(query, top_k=vector_top_k, metadata_filter=metadata_filter)
        graph_hits = self._graph_hits(query, top_k=graph_top_k)
        if not graph_hits:
            graph_hits = self._fallback_graph_hits(query, top_k=graph_top_k)
        # Return a balanced mix so both sources are considered by the reranker.
        return self._balanced_hits(vector_hits, graph_hits, top_k=top_k)
