import os
import sys
import math
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.services.text_splitter import split_markdown_by_words
from src.services.embeddings import EmbeddingModel
from src.repositories.vector.chroma_repository import ChromaVectorRepository


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    md = Path(ROOT) / 'tests' / 'sample.md'
    text = md.read_text(encoding='utf-8')
    chunks = split_markdown_by_words(text, chunk_size_words=200, overlap_words=40)

    model = EmbeddingModel()
    chunk_texts = [c['text'] for c in chunks]
    chunk_embs = model.embed(chunk_texts)

    query = 'adamjee'
    q_emb = model.embed([query])[0]

    print('model backend:', getattr(model, '_backend', 'unknown'))
    print('query:', query)
    print()

    for i, (c, emb) in enumerate(zip(chunks, chunk_embs)):
        sim = cosine(q_emb, emb)
        print(f'chunk {i} meta:', c['meta'])
        print('sim (cosine) with query:', sim)
        print('snippet:', c['text'][:320].replace('\n', ' '))
        print('embedding sample (first 8):', emb[:8])
        print('-' * 60)

    # Query Chroma if available
    try:
        repo = ChromaVectorRepository(collection_name='manual_debug_sample_md', persist_directory=str(Path(ROOT) / 'data' / 'chroma'))
        chroma_res = repo.search(q_emb, top_k=10)
        print('\nChroma query results (id, distance, metadata):')
        for r in chroma_res:
            print(r[0], r[1], r[2])
    except Exception as e:
        print('\nChroma query skipped (error):', e)


if __name__ == '__main__':
    main()
