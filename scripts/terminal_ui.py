#!/usr/bin/env python3
"""Polished terminal UI for RAGplusplus.

Features:
- Structured lifecycle output
- Progress and status labels
- Execution time and word counts
- Streaming Ollama output
- Rich UI when installed, ASCII fallback otherwise
"""

from __future__ import annotations

import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.orchestrator import RetrievalOrchestrator, build_default_orchestrator
from src.services.embeddings import EmbeddingModel
from src.services.vector_indexer import index_markdown
from src.services.text_splitter import split_markdown_by_words
import uuid
import math

try:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH = True
    console = Console()
except Exception:  # pragma: no cover - optional UI dependency
    RICH = False
    console = None
    Align = Panel = Table = Text = None  # type: ignore[assignment]


DETECTED_MODEL: str | None = None
VERBOSE = False


def now() -> str:
    return time.strftime("%H:%M:%S")


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def rule(title: str) -> None:
    if RICH:
        rich_console = console
        assert rich_console is not None and Panel is not None and Align is not None and Text is not None
        rich_console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{'=' * 10} {title} {'=' * 10}\n")


def panel(title: str, body: str) -> None:
    if RICH:
        rich_console = console
        assert rich_console is not None and Panel is not None and Align is not None and Text is not None
        rich_console.print(Panel(Align.left(Text(body, no_wrap=False)), title=f"[bold]{title}[/bold]"))
    else:
        print(f"--- {title} ---")
        print(body)


def status(label: str, message: str = "", kind: str = "info") -> None:
    symbol = {"info": "ℹ", "ok": "✓", "warn": "⚠", "error": "✗"}.get(kind, "•")
    line = f"[{now()}] {symbol} {label}"
    if message:
        line += f": {message}"
    if RICH:
        style = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(kind, "white")
        rich_console = console
        assert rich_console is not None
        rich_console.print(f"[{style}]{line}[/{style}]")
    else:
        print(line)


def parse_pdf_to_markdown(path: Path) -> tuple[str, str]:
    try:
        from src.ingestion.parsers.docling_parser import DoclingParser
        from src.ingestion.parsers.markdown_converter import MarkdownConverter

        parser = DoclingParser(use_ocr=False)
        parsed = parser.parse(path)
        converter = MarkdownConverter()
        converted = converter.convert(parsed, source_path=str(path.name))
        return converted.markdown_content, converted.source or path.name
    except Exception:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages), path.name
        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF: {exc}") from exc


def detect_model() -> str | None:
    try:
        import ollama

        client = ollama.Client()
        response = client.list()
        models = getattr(response, "models", None)
        if models:
            first = models[0]
            return getattr(first, "model", None) or (first.get("model") if isinstance(first, dict) else None)
    except Exception:
        return None
    return None


def build_orchestrator(model_name: str | None) -> RetrievalOrchestrator:
    try:
        return build_default_orchestrator(model_name=model_name or "phi3:mini")
    except Exception as exc:
        status("Orchestrator", f"fallback in use ({exc})", "warn")
        from src.repositories.vector.in_memory_vector_repository import InMemoryVectorRepository

        return RetrievalOrchestrator(
            vector_repo=InMemoryVectorRepository(),
            embedder=EmbeddingModel(),
            model_name=model_name or "phi3:mini",
        )


def show_metrics(rows: list[tuple[str, str]]) -> None:
    if RICH:
        rich_console = console
        assert rich_console is not None and Table is not None and Text is not None
        table = Table(title="Metrics", show_header=False, box=None)
        table.add_column("Name", style="bold cyan")
        table.add_column("Value", style="white")
        for name, value in rows:
            table.add_row(name, value)
        rich_console.print(table)
    else:
        print("Metrics:")
        for name, value in rows:
            print(f"  {name}: {value}")


def show_hits(hits) -> None:
    if RICH:
        rich_console = console
        assert rich_console is not None and Table is not None and Text is not None
        table = Table(title="Top Retrieval Hits", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Score", width=9)
        table.add_column("Source", width=20)
        table.add_column("Excerpt")
        for index, hit in enumerate(hits, start=1):
            # Use Rich Text objects to avoid markup parsing errors when content
            # contains bracketed tokens like [linkedin ...] which Rich would
            # otherwise interpret as markup and raise MarkupError.
            excerpt_text = Text((hit.content or "").replace("\n", " ")[:220])
            source_text = Text(hit.source or "vector")
            score_text = Text(f"{hit.score:.3f}")
            table.add_row(Text(str(index)), score_text, source_text, excerpt_text)
        rich_console.print(table)
    else:
        print("--- Top Retrieval Hits ---")
        for index, hit in enumerate(hits, start=1):
            excerpt = (hit.content or "").replace("\n", " ")[:220]
            print(f"{index}. ({hit.score:.3f}) [{hit.source}] {excerpt}")


def stream_llm_answer(prompt: str, model_name: str) -> str:
    import ollama

    client = ollama.Client()
    stream = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}], stream=True)

    parts: list[str] = []
    started = time.time()
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        parts.append(piece)
        if piece:
            if RICH:
                rich_console = console
                assert rich_console is not None and Text is not None
                # Print as Text to avoid markup parsing in streamed pieces.
                rich_console.print(Text(piece), end="")
            else:
                print(piece, end="", flush=True)
    if not RICH:
        print()
    status("LLM Completed", f"{time.time() - started:.2f}s", "ok")
    return "".join(parts)


def run_query(query: str, orchestrator: RetrievalOrchestrator, call_llm: bool = True) -> None:
    started_total = time.time()

    rule("Query Lifecycle")
    panel("User Prompt", query)

    status("Retrieval", "searching vector store...", "info")
    started = time.time()
    hits = orchestrator.retriever.retrieve(query, top_k=5)
    retrieval_seconds = time.time() - started
    status("Retrieval", f"completed in {retrieval_seconds:.2f}s with {len(hits)} hits", "ok")

    if VERBOSE:
        # Show raw retriever hits
        lines = []
        for i, h in enumerate(hits[:10], start=1):
            excerpt = (h.content or "").replace("\n", " ")[:300]
            lines.append(f"#{i} id={h.id} score={h.score:.4f} source={h.source} title={h.title} excerpt={excerpt}")
        panel("Retriever - Raw Hits", "\n".join(lines) or "(no hits)")

    status("Reranking", "applying cross-encoder...", "info")
    started = time.time()
    reranked = orchestrator.reranker.rerank(query, hits, top_k=5)
    rerank_seconds = time.time() - started
    status("Reranking", f"completed in {rerank_seconds:.2f}s", "ok")

    if VERBOSE:
        # Attempt to show cross-encoder raw scores if model available
        try:
            model_obj = orchestrator.reranker._load_model()
            if model_obj is not None:
                pairs = [(query, h.content) for h in hits]
                raw_scores = model_obj.predict(pairs)
            else:
                raw_scores = None
        except Exception:
            raw_scores = None

        rerank_lines = []
        for i, h in enumerate(reranked, start=1):
            orig_idx = next((idx for idx, hh in enumerate(hits) if hh.id == h.id), None)
            orig_score = hits[orig_idx].score if orig_idx is not None else 0.0
            raw = f"{raw_scores[orig_idx]:.4f}" if raw_scores is not None and orig_idx is not None else "n/a"
            rerank_lines.append(f"#{i} id={h.id} orig_score={orig_score:.4f} reranker_raw={raw} combined={h.score:.4f} source={h.source}")
        panel("Reranker - Results", "\n".join(rerank_lines) or "(no reranked results)")

    status("Prompt Build", "assembling citations and instructions...", "info")
    started = time.time()
    built = orchestrator.prompt_builder.build(query, reranked)
    prompt_seconds = time.time() - started
    status("Prompt Build", f"completed in {prompt_seconds:.2f}s", "ok")

    if VERBOSE:
        # Show prompt internals: citations and prompt preview
        try:
            citations_text = "\n".join([f"[{c['citation_id']}] id={c['id']} source={c['source']} score={c.get('score'):.4f}" for c in built.citations])
        except Exception:
            citations_text = str(built.citations)
        panel("Prompt Citations", citations_text or "(no citations)")
        panel("Assembled Prompt (preview)", built.prompt[:8000] + ("\n...truncated..." if len(built.prompt) > 8000 else ""))
        show_metrics([
            ("Prompt words", str(word_count(built.prompt))),
            ("Citations", str(len(built.citations)))
        ])

    show_metrics([
        ("Input words", str(word_count(query))),
        ("Prompt words", str(word_count(built.prompt))),
        ("Retrieval time", f"{retrieval_seconds:.2f}s"),
        ("Rerank time", f"{rerank_seconds:.2f}s"),
        ("Prompt build time", f"{prompt_seconds:.2f}s"),
    ])

    show_hits(reranked)
    panel("Intermediate Prompt", built.prompt)

    if call_llm:
        status("LLM", "streaming answer...", "info")
        model_name = DETECTED_MODEL or getattr(orchestrator.llm_client, "model", None) or "phi3:mini"
        try:
            if RICH:
                rich_console = console
                assert rich_console is not None and Panel is not None
                rich_console.print(Panel("", title="Final Answer", subtitle=f"model={model_name}"))
            else:
                print("--- Final Answer ---")
            answer = stream_llm_answer(built.prompt, model_name)
            if not answer.strip():
                status("LLM", "no text returned", "warn")
            panel("Final Answer", answer or "(no text returned)")
        except Exception as exc:
            status("LLM", f"failed: {exc}", "error")
            panel("Final Answer", f"(LLM failed: {exc})")

    total_seconds = time.time() - started_total
    show_metrics([
        ("Total lifecycle time", f"{total_seconds:.2f}s"),
        ("Prompt words", str(word_count(built.prompt))),
        ("Answer words", "streamed above" if call_llm else "n/a"),
    ])


def main() -> None:
    global DETECTED_MODEL, VERBOSE

    rule("RAGplusplus — Terminal UI")
    DETECTED_MODEL = detect_model()
    status("Model Detected", DETECTED_MODEL or "(none)", "ok" if DETECTED_MODEL else "warn")

    orchestrator = build_orchestrator(DETECTED_MODEL)
    status("Ready", "type 'help' for commands", "ok")

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            break

        if not raw:
            continue
        if raw in {"exit", "quit"}:
            break

        if raw == "help":
            help_text = textwrap.dedent(
                    """
                    Commands:
                        index /path/to/file.pdf   parse and index a PDF
                        query <question>          retrieval + rerank + streamed LLM answer
                        queryraw <question>       retrieval + rerank + prompt only
                        verbose on/off            toggle verbose pipeline logging
                        models                    show detected Ollama model
                        help                      show this message
                        exit                      quit
                    """
            ).strip()
            panel("Help", help_text)
            continue

        if raw == "models":
            panel("Detected Model", DETECTED_MODEL or "(none)")
            continue

        if raw == "verbose on":
            VERBOSE = True
            status("Verbose", "enabled", "ok")
            continue

        if raw == "verbose off":
            VERBOSE = False
            status("Verbose", "disabled", "ok")
            continue

        if raw.startswith("index "):
            pdf_path = Path(raw[len("index "):].strip())
            if not pdf_path.exists():
                status("Indexing", f"file not found: {pdf_path}", "error")
                continue

            status("Indexing", f"parsing {pdf_path.name}...", "info")
            try:
                started = time.time()
                markdown, source = parse_pdf_to_markdown(pdf_path)
                # Verbose instrumentation: show converted markdown, splitting, embeddings
                if VERBOSE:
                    panel("Converted Markdown (preview)", markdown[:4000] + ("\n...truncated..." if len(markdown) > 4000 else ""))
                    show_metrics([
                        ("Markdown chars", str(len(markdown))),
                        ("Markdown words", str(word_count(markdown)))
                    ])
                    # Split into chunks and show previews
                    chunks = split_markdown_by_words(markdown)
                    panel("Text Splitter", f"chunks={len(chunks)} chunk_size_words=... overlap_words=...")
                    # show first few chunk previews
                    preview_lines = []
                    for i, c in enumerate(chunks[:10]):
                        txt = c.get('text', '').replace('\n', ' ')[:400]
                        preview_lines.append(f"#{i} start={c['meta']['start_word']} end={c['meta']['end_word']} preview={txt}")
                    panel("Chunk Previews", "\n\n".join(preview_lines) or "(no chunks)")

                    # Compute embeddings for preview and show summaries
                    texts = [c['text'] for c in chunks]
                    embeddings = orchestrator.embedder.embed(texts)
                    emb_lines = []
                    for i, emb in enumerate(embeddings[:10]):
                        dim = len(emb)
                        norm = math.sqrt(sum(x * x for x in emb)) if dim else 0.0
                        sample = ", ".join(f"{v:.4f}" for v in emb[:6])
                        emb_lines.append(f"#{i} dim={dim} norm={norm:.4f} sample=[{sample}]")
                    panel("Embeddings (summary)", "\n".join(emb_lines) if emb_lines else "(no embeddings)")

                    # Build docs and index using precomputed embeddings to avoid double work
                    docs = []
                    for c, emb in zip(chunks, embeddings):
                        doc_id = str(uuid.uuid4())
                        meta = dict(c.get('meta', {}) or {})
                        if source:
                            meta['source'] = source
                        docs.append({
                            'id': doc_id,
                            'text': c['text'],
                            'embedding': emb,
                            'metadata': meta,
                        })
                    chunk_ids = orchestrator.vector_repo.add_documents(docs)
                    status("Indexing", f"indexed {len(chunk_ids)} chunks in {time.time() - started:.2f}s", "ok")
                    panel("Indexing Complete", f"Source: {source}\nChunks: {len(chunk_ids)}")
                else:
                    chunk_ids = index_markdown(markdown, orchestrator.vector_repo, model=orchestrator.embedder, source=source)
                    status("Indexing", f"indexed {len(chunk_ids)} chunks in {time.time() - started:.2f}s", "ok")
                    panel("Indexing Complete", f"Source: {source}\nChunks: {len(chunk_ids)}")
            except Exception as exc:
                status("Indexing", str(exc), "error")
                panel("Chroma Fix", "Set CHROMA_PERSIST_DIRECTORY to a writable path or fix permissions in data/chroma.")
            continue

        call_llm = True
        query = None
        if raw.startswith("queryraw "):
            query = raw[len("queryraw "):].strip()
            call_llm = False
        elif raw.startswith("query "):
            query = raw[len("query "):].strip()

        if query is not None:
            if not query:
                status("Input", "empty query", "warn")
                continue
            run_query(query, orchestrator, call_llm=call_llm)
            continue

        status("Input", f"unknown command: {raw}", "warn")


if __name__ == "__main__":
    main()