from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel, Field

from src.retrieval.orchestrator import RetrievalOrchestrator, build_default_orchestrator
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


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

    # mount static frontend
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def root_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "ok"}

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

    class UploadResponse(BaseModel):
        source: str
        ids: list[str]

    @app.post('/upload_pdf', response_model=UploadResponse)
    async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
        """Upload a PDF, parse -> convert -> index into vector store. Returns stored ids."""
        active_orchestrator = getattr(app.state, 'orchestrator', None)
        if active_orchestrator is None:
            raise HTTPException(status_code=500, detail='Retrieval orchestrator is not available')

        # save upload to temp file
        import tempfile
        from pathlib import Path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        tmp_path = Path(tmp.name)

        # parse PDF -> markdown
        markdown = None
        try:
            from src.ingestion.parsers.docling_parser import DoclingParser
            from src.ingestion.parsers.markdown_converter import MarkdownConverter
            parser = DoclingParser(use_ocr=False)
            parsed = parser.parse(tmp_path)
            converter = MarkdownConverter()
            converted = converter.convert(parsed, source_path=str(file.filename))
            markdown = converted.markdown_content
            source_name = converted.source or file.filename
        except Exception:
            # fallback using PyPDF2 if Docling unavailable
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(tmp_path))
                pages = [p.extract_text() or '' for p in reader.pages]
                markdown = '\n\n'.join(pages)
                source_name = file.filename
            except Exception as e:
                raise HTTPException(status_code=500, detail=f'PDF parsing failed: {e}')

        # index into vector repo
        repo = getattr(active_orchestrator, 'vector_repo', None)
        embedder = getattr(active_orchestrator, 'embedder', None)
        if repo is None or embedder is None:
            raise HTTPException(status_code=500, detail='Orchestrator missing required components')

        from src.services.vector_indexer import index_markdown
        ids = index_markdown(markdown, repo, model=embedder, source=source_name)

        return UploadResponse(source=source_name, ids=ids)

    class QueryPDFRequest(BaseModel):
        query: str = Field(..., min_length=1)
        top_k: int = Field(default=5, ge=1, le=20)
        source: Optional[str] = None

    from fastapi import Request as FastAPIRequest

    @app.post('/query_pdf', response_model=QueryResponse)
    async def query_pdf(req: FastAPIRequest) -> QueryResponse:
        active_orchestrator = getattr(app.state, 'orchestrator', None)
        if active_orchestrator is None:
            raise HTTPException(status_code=500, detail='Retrieval orchestrator is not available')

        try:
            payload = await req.json()
        except Exception:
            raise HTTPException(status_code=422, detail='Invalid or missing JSON body')

        try:
            request_model = QueryPDFRequest.parse_obj(payload)
        except Exception as exc:
            # return a clearer 422 for missing/invalid fields
            raise HTTPException(status_code=422, detail=str(exc))

        # use retriever with optional metadata filter
        retriever = getattr(active_orchestrator, 'retriever')
        reranker = getattr(active_orchestrator, 'reranker')
        prompt_builder = getattr(active_orchestrator, 'prompt_builder')
        llm_client = getattr(active_orchestrator, 'llm_client')

        metadata_filter = {'source': request_model.source} if request_model.source else None
        hits = retriever.retrieve(request_model.query, top_k=request_model.top_k, metadata_filter=metadata_filter)
        reranked = reranker.rerank(request_model.query, hits, top_k=request_model.top_k)
        built = prompt_builder.build(request_model.query, reranked)
        answer = llm_client.generate(built.prompt)

        return QueryResponse(
            query=request_model.query,
            answer=answer,
            prompt=built.prompt,
            citations=built.citations,
            hits=[h.to_dict() for h in reranked],
        )

    return app


app = create_app()
