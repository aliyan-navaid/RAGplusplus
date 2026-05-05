import os, sys, uuid, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.services.text_splitter import split_markdown_by_words
from src.services.embeddings import EmbeddingModel
from src.repositories.vector.chroma_repository import ChromaVectorRepository
# fallback if chroma isn't available you can swap to InMemoryVectorRepository for dry-run
from src.repositories.vector.in_memory_vector_repository import InMemoryVectorRepository

MD_PATH = os.path.join(os.path.dirname(__file__), 'sample.md')

def main():
    text = open(MD_PATH, 'r', encoding='utf8').read()
    print('--- Source length (chars):', len(text))

    # 1) Split
    chunks = split_markdown_by_words(text, chunk_size_words=200, overlap_words=40)
    print('Chunks produced:', len(chunks))
    for i, c in enumerate(chunks[:3]):
        print(f'--- chunk {i} meta: {c["meta"]}')
        print(c['text'][:200].replace('\\n',' ') + ('...' if len(c['text'])>200 else ''))
    print()

    # 2) Embed
    model = EmbeddingModel()
    texts = [c['text'] for c in chunks]
    embeddings = model.embed(texts)
    print('Embedding dimension:', len(embeddings[0]) if embeddings else 0)
    print('First vector sample (first 8 values):', embeddings[0][:8])

    # 3) Store to Chroma (or in-memory)
    try:
        persist_dir = tempfile.mkdtemp(prefix='chroma_')
        repo = ChromaVectorRepository(collection_name=f'debug_{uuid.uuid4().hex[:6]}', persist_directory=persist_dir)
        print('Using Chroma at persist dir:', persist_dir)
    except Exception as e:
        print('Chroma not available or failed to init — using InMemoryVectorRepository:', e)
        repo = InMemoryVectorRepository()

    docs = []
    for i, (txt, emb) in enumerate(zip(texts, embeddings)):
        docs.append({
            'id': f'chunk-{i}',
            'text': txt,
            'embedding': emb,
            'metadata': {'chunk_index': i, 'source': os.path.basename(MD_PATH)}
        })
    ids = repo.add_documents(docs)
    print('Stored IDs:', ids[:5], '... total', len(ids))

    # Query: use a short phrase (or the first chunk) and print top results
    q_text = 'banana' if 'banana' in text else chunks[0]['text'].split()[:10]
    if isinstance(q_text, list):
        q_text = ' '.join(q_text)
    q_emb = model.embed([q_text])[0]
    results = repo.search(q_emb, top_k=5)
    print('\\nTop results for query:', q_text)
    for rid, score, meta, doc in results:
        print('id=', rid, 'score=', score, 'meta=', meta)
        print('snippet:', doc[:200].replace('\\n',' '), '...\\n')

if __name__ == '__main__':
    main()