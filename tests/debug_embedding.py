from pathlib import Path
from src.services.text_splitter import split_markdown_by_words
from src.services.embeddings import EmbeddingModel

text = Path("tests/sample.md").read_text(encoding="utf-8")
chunks = split_markdown_by_words(text, chunk_size_words=200, overlap_words=40)

model = EmbeddingModel()
vectors = model.embed([c["text"] for c in chunks])

print("backend:", model._backend)
print("chunks:", len(chunks))
print("vectors:", len(vectors))
print("vector_dim:", len(vectors[0]) if vectors else 0)

for i, vec in enumerate(vectors):
    print(f"\n--- chunk {i} ---")
    print("meta:", chunks[i]["meta"])
    print("first_12_values:", vec[:12])