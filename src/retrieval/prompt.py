from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.retrieval.models import RetrievalHit


@dataclass
class BuiltPrompt:
    prompt: str
    citations: List[Dict[str, Any]]


class PromptBuilder:
    """Builds a context-rich prompt for the local LLM."""

    def __init__(self, max_context_chars: int = 7000):
        self.max_context_chars = max_context_chars

    def build(self, query: str, hits: List[RetrievalHit]) -> BuiltPrompt:
        citations: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        remaining = self.max_context_chars

        for index, hit in enumerate(hits, start=1):
            snippet = hit.content.strip().replace("\n", " ")
            if len(snippet) > 1200:
                snippet = snippet[:1200].rstrip() + "..."

            citation = {
                "citation_id": index,
                "source": hit.source,
                "id": hit.id,
                "title": hit.title,
                "metadata": hit.metadata,
                "score": hit.score,
            }
            citations.append(citation)

            block = (
                f"[{index}] source={hit.source} id={hit.id} score={hit.score:.4f}\n"
                f"{snippet}"
            )
            if len(block) + 2 > remaining:
                break
            context_parts.append(block)
            remaining -= len(block) + 2

        prompt = (
            "You are CiteCheck Pro, an academic RAG assistant.\n"
            "Answer the user's question only from the provided context.\n"
            "If the context does not contain the answer, say you do not have enough information.\n"
            "Include inline citations like [1], [2] for claims.\n\n"
            f"Question:\n{query}\n\n"
            "Context:\n"
            + "\n\n".join(context_parts)
            + "\n\n"
            "Return a concise, well-structured answer with citations."
        )
        return BuiltPrompt(prompt=prompt, citations=citations)
