"""Paper data model and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    url: str = ""
    source: str = ""  # "arxiv", "semantic_scholar", "sample"
    topic: str = ""  # optional label, used only for evaluation

    @property
    def text(self) -> str:
        """Text that gets indexed: title + abstract."""
        return f"{self.title}. {self.abstract}".strip()


def save_corpus(papers: list[Paper], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in papers], f, indent=2, ensure_ascii=False)


def load_corpus(path: str | Path) -> list[Paper]:
    with Path(path).open(encoding="utf-8") as f:
        return [Paper(**d) for d in json.load(f)]


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Drop duplicates by normalised title; keep papers that have an abstract."""
    seen: set[str] = set()
    out: list[Paper] = []
    for p in papers:
        key = " ".join(p.title.lower().split())
        if not p.abstract.strip() or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
