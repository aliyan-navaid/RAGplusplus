./venv/bin/python - <<'PY'
from pathlib import Path
from src.services.text_splitter import split_markdown_by_words

text = Path("tests/sample.md").read_text(encoding="utf-8")
words = text.split()
chunks = split_markdown_by_words(text, chunk_size_words=200, overlap_words=40)

print("original_word_count:", len(words))
covered = set()

for chunk in chunks:
    start = chunk["meta"]["start_word"]
    end = chunk["meta"]["end_word"]
    covered.update(range(start, end))
    print("\n--- chunk ---")
    print(chunk["meta"])
    print(chunk["text"])  # print full chunk, not a preview

print("\ncovered_word_count:", len(covered))
print("missing_word_positions:", [i for i in range(len(words)) if i not in covered])
PY