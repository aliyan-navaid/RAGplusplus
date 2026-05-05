from __future__ import annotations

from typing import Any, Optional

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
        client = self._resolve_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(model=self.model, messages=messages)
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        return getattr(getattr(response, "message", None), "content", "")
