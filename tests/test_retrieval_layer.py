import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import dataclass

from src.retrieval.ensemble import EnsembleRetriever
from src.retrieval.models import RetrievalHit
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.retrieval.prompt import PromptBuilder
from src.retrieval.reranker import CrossEncoderReranker


class FakeVectorRepo:
    def search(self, query_embedding, top_k=5):
        return [
            ("chunk-1", 1.8, {"chunk_index": 1}, "A different passage about systems and databases."),
            ("chunk-0", 1.1, {"chunk_index": 0}, "Higher Secondary Certificate (HSC) Adamjee Govt. Science College"),
        ][:top_k]


@dataclass
class FakeEntity:
    id: str
    name: str
    type: str = "CONCEPT"


class FakeGraphRepo:
    def search_entities_by_name(self, name: str, limit: int = 10):
        if "adamjee" in name.lower():
            return [FakeEntity(id="entity-1", name="Adamjee Govt.")]
        return []

    def get_related_entities(self, entity_id: str, limit: int = 10):
        return [FakeEntity(id="entity-2", name="Science College")]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeLLMClient:
    def generate(self, prompt: str, system: str | None = None) -> str:
        return "FAKE ANSWER"


def test_ensemble_retriever_merges_vector_and_graph_hits():
    retriever = EnsembleRetriever(vector_repo=FakeVectorRepo(), graph_repo=FakeGraphRepo(), embedder=FakeEmbedder())
    hits = retriever.retrieve("Adamjee Govt.", top_k=5)

    sources = [hit.source for hit in hits]
    assert "vector" in sources
    assert "graph_entity" in sources


def test_reranker_prioritizes_exact_phrase():
    reranker = CrossEncoderReranker()
    hits = [
        RetrievalHit(id="chunk-1", content="Systems content", score=0.5, source="vector"),
        RetrievalHit(id="chunk-0", content="Higher Secondary Certificate (HSC) Adamjee Govt. Science College", score=0.4, source="vector"),
    ]
    reranked = reranker.rerank("Adamjee Govt.", hits)
    assert reranked[0].id == "chunk-0"


def test_prompt_builder_includes_citations():
    builder = PromptBuilder()
    built = builder.build(
        "Adamjee Govt.",
        [
            RetrievalHit(id="chunk-0", content="Higher Secondary Certificate (HSC) Adamjee Govt. Science College", score=0.9, source="vector"),
        ],
    )

    assert "Question:" in built.prompt
    assert "Context:" in built.prompt
    assert "[1]" in built.prompt
    assert built.citations[0]["id"] == "chunk-0"


def test_orchestrator_answers_with_citations():
    orchestrator = RetrievalOrchestrator(
        vector_repo=FakeVectorRepo(),
        graph_repo=FakeGraphRepo(),
        embedder=FakeEmbedder(),
        llm_client=FakeLLMClient(),
    )

    response = orchestrator.answer("Adamjee Govt.", top_k=5)
    assert response.answer == "FAKE ANSWER"
    assert response.citations
    assert response.hits[0].id == "chunk-0"
