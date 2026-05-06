from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os

try:
    import chromadb
except Exception:  # pragma: no cover - handled at runtime
    chromadb = None

from .base_repository import VectorRepository


class ChromaVectorRepository(VectorRepository):
    """ChromaDB-backed implementation of VectorRepository.

    Usage:
        repo = ChromaVectorRepository(collection_name='papers', persist_directory='./chroma')
    """

    def __init__(self, collection_name: str = "default", persist_directory: Optional[str] = None, client: Optional[Any] = None):
        if chromadb is None:
            raise RuntimeError('chromadb is required for ChromaVectorRepository')

        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY")
            if not persist_directory:
                persist_directory = str(Path(__file__).resolve().parents[3] / "data" / "chroma")

        allow_in_memory = os.getenv("CHROMA_ALLOW_IN_MEMORY", "0") == "1"

        # If no client provided, try to create a persistent client when a
        # persist_directory is specified. If the directory exists but is not
        # writable (e.g., mounted read-only or permission issue), either
        # fallback to in-memory if explicitly allowed or raise a clear error.
        if client is None:
            if persist_directory:
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
                # quick writability test
                try:
                    test_path = Path(persist_directory) / ".chroma_write_test"
                    with open(test_path, "w") as tf:
                        tf.write("ok")
                    test_path.unlink()
                    client = chromadb.PersistentClient(path=persist_directory)
                except Exception as e:
                    if allow_in_memory:
                        # Fallback to ephemeral in-memory client when persistent
                        # storage isn't usable. Print a concise warning so CLI/TUI
                        # users see the cause.
                        print(f"[WARN] Chroma persist_directory '{persist_directory}' not writable: {e}. Using in-memory store.")
                        client = chromadb.Client()
                    else:
                        raise RuntimeError(
                            f"Chroma persist_directory not writable: {persist_directory}. "
                            "Fix permissions or set CHROMA_PERSIST_DIRECTORY to a writable path."
                        ) from e
            else:
                client = chromadb.Client()

        self._client = client
        self._collection_name = collection_name

        # get or create collection
        self._col = self._client.get_or_create_collection(name=collection_name)

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        ids = [d['id'] for d in docs]
        documents = [d.get('text') for d in docs]
        embeddings = [d.get('embedding') for d in docs]
        metadatas = [d.get('metadata') for d in docs]

        self._col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return ids

    def search(self, query_embedding, top_k: int = 5, metadata_filter: dict | None = None) -> List[Tuple[str, float, Dict[str, Any], str]]:
        # Chroma returns lists under keys; query expects list of embeddings
        # note: 'ids' is not a valid include value for some chroma versions
        try:
            # If metadata_filter is provided, try to pass it as 'where' (supported in newer chroma)
            if metadata_filter:
                res = self._col.query(query_embeddings=[query_embedding], n_results=top_k, include=["metadatas", "documents", "distances"], where=metadata_filter)
            else:
                res = self._col.query(query_embeddings=[query_embedding], n_results=top_k, include=["metadatas", "documents", "distances"])
        except TypeError:
            # Older chroma versions may not support 'where' argument; fall back to post-filtering
            res = self._col.query(query_embeddings=[query_embedding], n_results=top_k, include=["metadatas", "documents", "distances"])

        ids = res.get('ids', [[]])[0]
        docs = res.get('documents', [[]])[0]
        metas = res.get('metadatas', [[]])[0]
        dists = res.get('distances', [[]])[0]

        out: List[Tuple[str, float, Dict[str, Any], str]] = []
        for _id, dist, meta, doc in zip(ids, dists, metas, docs):
            out.append((_id, float(dist), meta, doc))

        # If we had to fallback (or chroma didn't filter), apply metadata_filter locally
        if metadata_filter and out:
            filtered = []
            for item in out:
                meta = item[2] or {}
                ok = True
                for k, v in metadata_filter.items():
                    if meta.get(k) != v:
                        ok = False
                        break
                if ok:
                    filtered.append(item)
            return filtered[:top_k]

        return out
