from typing import List, Dict, Any


def split_markdown_by_words(markdown: str, chunk_size_words: int = 400, overlap_words: int = 50) -> List[Dict[str, Any]]:
    """Split a markdown string into overlapping chunks by word count.

    Returns list of dicts: {"text": str, "meta": {"chunk_index": int, ...}}
    """
    if chunk_size_words <= 0:
        raise ValueError('chunk_size_words must be > 0')
    if overlap_words < 0:
        raise ValueError('overlap_words must be >= 0')

    words = markdown.split()
    out: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    n = len(words)
    step = chunk_size_words - overlap_words if chunk_size_words > overlap_words else chunk_size_words
    if step <= 0:
        step = chunk_size_words

    while start < n:
        end = min(start + chunk_size_words, n)
        chunk_words = words[start:end]
        text = ' '.join(chunk_words)
        out.append({
            'text': text,
            'meta': {'chunk_index': idx, 'start_word': start, 'end_word': end}
        })
        idx += 1
        start += step

    return out
