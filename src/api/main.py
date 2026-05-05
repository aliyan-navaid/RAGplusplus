from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.orchestrator import RetrievalOrchestrator, build_default_orchestrator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    prompt: str
    citations: list[dict[str, Any]]
    hits: list[dict[str, Any]]


def create_app(orchestrator: Optional[RetrievalOrchestrator] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = orchestrator or build_default_orchestrator()
        yield
        llm_client = getattr(app.state.orchestrator, "llm_client", None)
        close = getattr(llm_client, "close", None)
        if callable(close):
            close()
        graph_repo = getattr(app.state.orchestrator, "graph_repo", None)
        close_graph = getattr(graph_repo, "close", None)
        if callable(close_graph):
            close_graph()

    app = FastAPI(title="CiteCheck Pro API", version="0.1.0", lifespan=lifespan)

    @app.get("/status")
    def status() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        active_orchestrator = getattr(app.state, "orchestrator", None)
        if active_orchestrator is None:
            raise HTTPException(status_code=500, detail="Retrieval orchestrator is not available")

        result = active_orchestrator.answer(request.query, top_k=request.top_k)
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            prompt=result.prompt,
            citations=result.citations,
            hits=[hit.to_dict() for hit in result.hits],
        )

    return app


app = create_app()
