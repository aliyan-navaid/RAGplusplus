from pathlib import Path
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.services.text_splitter import split_markdown_by_words
from src.services.embeddings import EmbeddingModel
from src.repositories.vector.chroma_repository import ChromaVectorRepository


PERSIST_DIR = os.path.join(ROOT, 'data', 'chroma')
COLLECTION_NAME = 'manual_debug_sample_md'
MD_PATH = os.path.join(ROOT, 'tests', 'sample.md')


def normalize_text(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else ' ')
    return ' '.join(''.join(cleaned).split())


def token_set(value: str):
    normalized = normalize_text(value)
    if not normalized:
        return set()
    return set(normalized.split())


def build_docs():
    text = Path(MD_PATH).read_text(encoding='utf-8')
    chunks = split_markdown_by_words(text, chunk_size_words=200, overlap_words=40)
    model = EmbeddingModel()
    vectors = model.embed([chunk['text'] for chunk in chunks])

    docs = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        docs.append({
            'id': f'chunk-{index}',
            'text': chunk['text'],
            'embedding': vector,
            'metadata': {
                'chunk_index': index,
                'start_word': chunk['meta']['start_word'],
                'end_word': chunk['meta']['end_word'],
                'source': 'sample.md',
            },
        })
    return model, chunks, vectors, docs


def rerank_exact_match(results, q_text, alpha: float = 0.5):
    """Rerank Chroma results using exact phrase and token overlap signals.

    results: list of (id, distance, metadata, document)
    Returns reranked list in same tuple format.
    """
    if not results:
        return results
    query_norm = normalize_text(q_text)
    query_tokens = token_set(q_text)
    max_dist = max(r[1] for r in results)
    scored = []
    for r in results:
        _id, dist, meta, doc = r
        doc_norm = normalize_text(doc)
        doc_tokens = token_set(doc)
        sim = (max_dist - dist)
        exact_phrase = 1.0 if query_norm and query_norm in doc_norm else 0.0
        overlap = (len(query_tokens & doc_tokens) / len(query_tokens)) if query_tokens else 0.0
        combined = (2.0 * exact_phrase) + (1.5 * overlap) + (alpha * sim)
        scored.append((exact_phrase, overlap, combined, r))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [r for (_, _, _, r) in scored]


def write_mode():
    model, chunks, vectors, docs = build_docs()
    repo = ChromaVectorRepository(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR)

    print('backend:', model._backend)
    print('persist_dir:', PERSIST_DIR)
    print('collection:', COLLECTION_NAME)
    print('chunks:', len(chunks))
    print('vectors:', len(vectors))
    print('vector_dim:', len(vectors[0]) if vectors else 0)

    ids = repo.add_documents(docs)
    print('stored_ids:', ids)
    print('collection_count_after_write:', repo._col.count())

    query_text = 'Adamjee Govt.'
    query_vector = model.embed([query_text])[0]
    results = repo.search(query_vector, top_k=5)

    print('\nquery:', query_text)
    print('results:')
    for item_id, score, metadata, document in results:
        print('id:', item_id)
        print('score:', score)
        print('metadata:', metadata)
        print('snippet:', document[:220].replace('\n', ' '))
        print('---')

    # apply exact-match re-ranker and show reranked output
    reranked = rerank_exact_match(results, query_text, alpha=0.5)
    print('\nre-ranked results:')
    for item_id, score, metadata, document in reranked:
        print('id:', item_id)
        print('score:', score)
        print('metadata:', metadata)
        print('snippet:', document[:220].replace('\n', ' '))
        print('---')


def query_mode():
    model = EmbeddingModel()
    repo = ChromaVectorRepository(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR)
    print('backend:', model._backend)
    print('persist_dir:', PERSIST_DIR)
    print('collection:', COLLECTION_NAME)
    print('collection_count_on_reload:', repo._col.count())

    query_text = 'Adamjee Govt.'
    query_vector = model.embed([query_text])[0]
    results = repo.search(query_vector, top_k=5)

    print('\nquery:', query_text)
    print('results:')
    for item_id, score, metadata, document in results:
        print('id:', item_id)
        print('score:', score)
        print('metadata:', metadata)
        print('snippet:', document[:220].replace('\n', ' '))
        print('---')

    reranked = rerank_exact_match(results, query_text, alpha=0.5)
    print('\nre-ranked results:')
    for item_id, score, metadata, document in reranked:
        print('id:', item_id)
        print('score:', score)
        print('metadata:', metadata)
        print('snippet:', document[:220].replace('\n', ' '))
        print('---')


if __name__ == '__main__':
    mode = 'write'
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == 'write':
        write_mode()
    elif mode == 'query':
        query_mode()
    else:
        raise SystemExit("Usage: python tests/debug_chroma.py [write|query]")
