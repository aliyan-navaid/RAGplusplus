import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.retrieval.models import RetrievalResponse, RetrievalHit


class FakeOrchestrator:
    def answer(self, query: str, top_k: int = 5):
        return RetrievalResponse(
            query=query,
            answer="FAKE API ANSWER",
            prompt="FAKE PROMPT",
            citations=[{"citation_id": 1, "source": "vector", "id": "chunk-0"}],
            hits=[RetrievalHit(id="chunk-0", content="Adamjee Govt.", score=1.0, source="vector")],
        )


def test_query_endpoint_returns_rich_response():
    app = create_app(orchestrator=FakeOrchestrator())
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "Adamjee Govt.", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "FAKE API ANSWER"
    assert payload["citations"][0]["id"] == "chunk-0"
    assert payload["hits"][0]["id"] == "chunk-0"
