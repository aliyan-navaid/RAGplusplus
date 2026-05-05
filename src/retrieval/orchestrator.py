from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.repositories.vector.chroma_repository import ChromaVectorRepository

try:
    from src.repositories.graph.neo4j_repository import Neo4jRepository
except Exception:  # pragma: no cover - graph backend may be unavailable during tests
    Neo4jRepository = None

from src.services.embeddings import EmbeddingModel

from .ensemble import EnsembleRetriever
from .models import RetrievalHit, RetrievalResponse
from .ollama_client import OllamaClient
from .prompt import PromptBuilder
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Coordinates hybrid retrieval, reranking, prompt construction, and generation."""

    def __init__(
        self,
        vector_repo: Any,
        graph_repo: Any = None,
        embedder: Any = None,
        retriever: Optional[EnsembleRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_client: Optional[OllamaClient] = None,
        model_name: str = "phi3-mini",
    ):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.embedder = embedder or EmbeddingModel()
        self.retriever = retriever or EnsembleRetriever(vector_repo=vector_repo, graph_repo=graph_repo, embedder=self.embedder)
        self.reranker = reranker or CrossEncoderReranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or OllamaClient(model=model_name)

    def answer(self, query: str, top_k: int = 5) -> RetrievalResponse:
        hits = self.retriever.retrieve(query, top_k=top_k)
        reranked_hits = self.reranker.rerank(query, hits, top_k=top_k)
        built_prompt = self.prompt_builder.build(query, reranked_hits)
        answer_text = self.llm_client.generate(built_prompt.prompt)
        return RetrievalResponse(
            query=query,
            answer=answer_text,
            prompt=built_prompt.prompt,
            citations=built_prompt.citations,
            hits=reranked_hits,
        )


def build_default_orchestrator(model_name: str = "phi3-mini") -> RetrievalOrchestrator:
    """Construct a default orchestrator using the persisted Chroma store and, if available, Neo4j."""
    vector_repo = ChromaVectorRepository()
    graph_repo = None
    if Neo4jRepository is not None:
        try:
            graph_repo = Neo4jRepository()
        except Exception as exc:  # pragma: no cover - depends on local Neo4j availability
            logger.warning("Neo4j unavailable, continuing without graph retrieval: %s", exc)
            graph_repo = None

    return RetrievalOrchestrator(
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        embedder=EmbeddingModel(),
        llm_client=OllamaClient(model=model_name),
        model_name=model_name,
    )
