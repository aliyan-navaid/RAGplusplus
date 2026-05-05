from typing import List
import hashlib
import math


class EmbeddingModel:
    """Simple embedding model abstraction.

    It uses `sentence_transformers` when available and falls back to a
    deterministic hashing-based embedding only when the transformer backend
    cannot be imported or initialized.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        self._model = None
        self._backend = 'fallback'
        try:
            from sentence_transformers import SentenceTransformer
            if device is None:
                self._model = SentenceTransformer(model_name)
            else:
                self._model = SentenceTransformer(model_name, device=device)
            self._backend = 'sentence_transformers'
        except Exception:
            self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            arr = self._model.encode(texts, show_progress_bar=False)
            return [list(map(float, a)) for a in arr]
        # deterministic fallback: hash each text and produce a small vector
        out = []
        dim = 64
        for t in texts:
            h = hashlib.sha256(t.encode('utf8')).digest()
            # convert bytes to list of ints
            vec = [float(b) for b in h]
            if len(vec) < dim:
                vec = vec + [0.0] * (dim - len(vec))
            else:
                vec = vec[:dim]
            # normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm == 0:
                out.append([0.0] * dim)
            else:
                out.append([x / norm for x in vec])
        return out
