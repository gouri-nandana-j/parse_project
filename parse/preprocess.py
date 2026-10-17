"""Text helpers.

clean_text is used by the lexical (TF-IDF) baseline. SBERT receives raw text:
transformer models work best on natural sentences.
"""

import re

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9\s\-]")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def split_chunks(text: str, max_words: int = 120) -> list[str]:
    """Group consecutive sentences into chunks of at most ~max_words words.

    SBERT models only read the first few hundred tokens, so a long pasted idea is
    embedded chunk by chunk instead of being silently truncated.
    """
    chunks, cur, n = [], [], 0
    for s in split_sentences(text):
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]
