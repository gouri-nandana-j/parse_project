"""PARSE settings. Secrets come from a git-ignored ".env" file (or real environment variables).

Setup: copy ".env.example" to ".env" and fill in the placeholders there. Nothing in this file
needs editing.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> dict:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


_ENV = _load_dotenv(Path(__file__).resolve().parent / ".env")


def _get(name: str) -> str:
    value = os.environ.get(name) or _ENV.get(name, "")
    return "" if value.startswith("PASTE_YOUR") else value  # untouched placeholder == not set


SEMANTIC_SCHOLAR_API_KEY = _get("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_EMAIL = _get("OPENALEX_EMAIL")

# Sentence-BERT model (downloaded automatically on first run, ~90 MB, then cached by HuggingFace).
SBERT_MODEL = "all-MiniLM-L6-v2"

# How many candidate papers to pull from each source per search.
CANDIDATES_PER_SOURCE = 60
