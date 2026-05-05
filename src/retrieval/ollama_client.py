from __future__ import annotations

from typing import Any, Optional
import concurrent.futures
import logging

try:
    import ollama
except Exception:  # pragma: no cover - optional runtime dependency
    ollama = None


class OllamaClient:
    """Thin wrapper around Ollama chat completion."""

    def __init__(self, model: str = "phi3-mini", host: Optional[str] = None, client: Any = None):
        self.model = model
        self.host = host
        self._client = client

    def _resolve_client(self):
        if self._client is not None:
            return self._client
        if ollama is None:
            raise RuntimeError("ollama package is not installed")
        if self.host:
            return ollama.Client(host=self.host)
        return ollama.Client()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a response from Ollama using the ollama library with timeout."""
        import sys
        
        if ollama is None:
            raise RuntimeError("ollama package is not installed")
        
        client = self._resolve_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        def _chat_call(model_name: str):
            print(f"[ollama] chat call starting with model={model_name}", file=sys.stderr, flush=True)
            result = client.chat(model=model_name, messages=messages)
            print(f"[ollama] chat call completed", file=sys.stderr, flush=True)
            return result

        timeout_seconds = 120
        try:
            print(f"[ollama] submitting to thread pool with timeout={timeout_seconds}s", file=sys.stderr, flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_chat_call, self.model)
                response = fut.result(timeout=timeout_seconds)
            print(f"[ollama] got response", file=sys.stderr, flush=True)
        except concurrent.futures.TimeoutError:
            print(f"[ollama] TIMEOUT after {timeout_seconds}s", file=sys.stderr, flush=True)
            raise RuntimeError(f"Ollama call timed out after {timeout_seconds}s. Is the model loading? Try running again.")
        except Exception as exc:
            print(f"[ollama] Exception {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise

        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        return getattr(getattr(response, "message", None), "content", "")
