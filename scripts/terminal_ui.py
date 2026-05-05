#!/usr/bin/env python3
"""Minimal terminal UI for testing indexing and queries.

Usage:
  python scripts/terminal_ui.py

Commands:
  index /path/to/file.pdf   - parse and index PDF into vector store
  query Your question here  - run retrieval + rerank + prompt + (try) generate
  exit                      - quit

This tool bypasses the HTTP API and calls the orchestrator directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.orchestrator import build_default_orchestrator, RetrievalOrchestrator
from src.services.vector_indexer import index_markdown
from src.services.embeddings import EmbeddingModel
import time
import json
import concurrent.futures


def _now():
    return time.strftime('%H:%M:%S')


# Global to store detected model for use in query command
_detected_model = None


def parse_pdf_to_markdown(path: Path) -> tuple[str, str]:
    """Return (markdown, source_name)."""
    try:
        from src.ingestion.parsers.docling_parser import DoclingParser
        from src.ingestion.parsers.markdown_converter import MarkdownConverter
        parser = DoclingParser(use_ocr=False)
        parsed = parser.parse(path)
        conv = MarkdownConverter()
        converted = conv.convert(parsed, source_path=str(path.name))
        return converted.markdown_content, converted.source or path.name
    except Exception:
        # fallback
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = [p.extract_text() or '' for p in reader.pages]
            return '\n\n'.join(pages), path.name
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {e}")


def main():
    print("Terminal UI — RAGplusplus")
    orch = None
    detected_model = None
    try:
        # Detect available model first
        try:
            import ollama
            client = ollama.Client()
            resp = client.list()
            if hasattr(resp, 'models') and resp.models:
                detected_model = resp.models[0].model
                print(f"Detected model: {detected_model}")
        except Exception as e:
            print(f"Could not detect model: {e}")
        
        # Build orchestrator with detected model
        orch = build_default_orchestrator(model_name=detected_model or "phi3:mini")
    except Exception as e:
        print(f"Warning: build_default_orchestrator failed: {e}\nFalling back to partial orchestrator.")
        try:
            from src.repositories.vector.in_memory_vector_repository import InMemoryVectorRepository
            emb = EmbeddingModel()
            repo = InMemoryVectorRepository()
            orch = RetrievalOrchestrator(vector_repo=repo, embedder=emb)
        except Exception as e2:
            print(f"Failed to create fallback orchestrator: {e2}")
            sys.exit(1)

    print("Orchestrator ready. type 'help' for commands.")
    
    # Store detected model in global scope for query command
    global _detected_model
    _detected_model = detected_model

    while True:
        try:
            raw = input('> ').strip()
        except EOFError:
            break
        if not raw:
            continue
        if raw in ('exit', 'quit'):
            break
        if raw == 'help':
            print("Commands:\n  index /path/to/file.pdf\n  query your question         # retrieve+rerank+prompt+LLM (streaming)\n  queryraw your question     # retrieve+rerank+prompt only\n  models                     # list ollama models\n  test                       # test ollama\n  exit")
            continue

        if raw.startswith('index '):
            path = raw[len('index '):].strip()
            p = Path(path)
            if not p.exists():
                print(f"File not found: {p}")
                continue
            print(f"Parsing {p} ...")
            try:
                markdown, source = parse_pdf_to_markdown(p)
                ids = index_markdown(markdown, orch.vector_repo, model=orch.embedder, source=source)
                print(f"Indexed {len(ids)} chunks for source: {source}")
            except Exception as e:
                print(f"Indexing failed: {e}")
            continue

        if raw.startswith('models'):
            # diagnostic: list ollama models if ollama is installed
            try:
                import ollama
                client = ollama.Client()
                resp = client.list()
                
                # resp is ollama._types.ListResponse with a .models attribute
                if hasattr(resp, 'models'):
                    models_list = resp.models
                elif isinstance(resp, dict):
                    models_list = resp.get('models', [])
                else:
                    models_list = []
                
                model_names = []
                for m in models_list:
                    if hasattr(m, 'model'):
                        model_names.append(m.model)
                    elif isinstance(m, dict) and 'model' in m:
                        model_names.append(m['model'])
                    elif isinstance(m, str):
                        model_names.append(m)
                
                print('Available models:')
                if model_names:
                    print('\n'.join(model_names))
                else:
                    print('(no models found)')
            except Exception as e:
                print(f"Could not list models: {e}")
            continue

        if raw.startswith('test'):
            # Test ollama connectivity with a simple ping
            try:
                import ollama
                client = ollama.Client()
                print(f"[{_now()}] Testing Ollama connectivity...")
                start = time.time()
                # Try a very short generate call (empty message)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(client.chat, model='phi3:mini', messages=[{"role": "user", "content": "hello"}])
                    resp = fut.result(timeout=5)
                elapsed = time.time() - start
                print(f"[{_now()}] Ollama responded in {elapsed:.2f}s")
                print(f"[{_now()}] Response: {str(resp)[:200]}")
            except concurrent.futures.TimeoutError:
                print(f"[{_now()}] Ollama TIMEOUT (5s)")
            except Exception as e:
                print(f"[{_now()}] Ollama test failed: {e}")
            continue

        if raw.startswith('queryraw '):
            q = raw[len('queryraw '):].strip()
            call_llm = False
        elif raw.startswith('query '):
            q = raw[len('query '):].strip()
            call_llm = True
        else:
            q = None

        if q is not None:
            if not q:
                print('Empty query')
                continue
            print(f"[{_now()}] Query: {q}")
            try:
                print(f"[{_now()}] retrieving...")
                start = time.time()
                hits = orch.retriever.retrieve(q, top_k=5)
                print(f"[{_now()}] retrieved {len(hits)} hits (t={time.time()-start:.2f}s)")

                print(f"[{_now()}] reranking...")
                start = time.time()
                reranked = orch.reranker.rerank(q, hits, top_k=5)
                print(f"[{_now()}] reranked {len(reranked)} hits (t={time.time()-start:.2f}s)")

                print(f"[{_now()}] building prompt...")
                built = orch.prompt_builder.build(q, reranked)
                print(f"[{_now()}] prompt built (len={len(built.prompt)})")

                if not call_llm:
                    print('\n--- PROMPT ---')
                    print(built.prompt[:2000])
                    print('\n--- HITS ---')
                    for h in reranked:
                        print(f"- ({h.score:.3f}) [{h.source}] {h.content[:200].replace('\n',' ')}")
                    continue

                # Call Ollama with streaming
                print(f"[{_now()}] calling LLM (streaming)...")
                try:
                    import ollama
                    client = ollama.Client()
                    model_to_use = _detected_model or "phi3:mini"
                    
                    print('\n--- ANSWER ---')
                    stream = client.chat(
                        model=model_to_use,
                        messages=[{"role": "user", "content": built.prompt}],
                        stream=True,
                    )
                    
                    start = time.time()
                    for chunk in stream:
                        content = chunk.get('message', {}).get('content', '')
                        print(content, end='', flush=True)
                    print(f"\n[{_now()}] LLM done (t={time.time()-start:.2f}s)")
                except Exception as e:
                    print(f"\n[{_now()}] LLM call FAILED: {e}")
                    print("(see prompt above for manual testing with: ollama run phi3:mini)")
                
                print('\n--- PROMPT ---')
                print(built.prompt[:2000])
                print('\n--- HITS ---')
                for h in reranked:
                    print(f"- ({h.score:.3f}) [{h.source}] {h.content[:200].replace('\n',' ')}")

            except Exception as e:
                print(f"Query failed: {e}\nAttempting partial retrieval...")
                try:
                    print('Dumping prompt/hits for inspection...')
                    print('\n--- PROMPT ---')
                    print(built.prompt[:2000])
                except Exception:
                    pass
            continue

        print("Unknown command. Type 'help'.")


if __name__ == '__main__':
    main()
