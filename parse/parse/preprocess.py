"""Text cleaning used by the lexical (TF-IDF) baseline.

SBERT receives the raw text: transformer models work best on natural sentences.
"""

import re

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9\s\-]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()
