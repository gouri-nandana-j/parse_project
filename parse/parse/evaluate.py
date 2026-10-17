"""Information-retrieval metrics and a comparison harness.

Relevance is defined by topic labels: a paper is relevant to a query if its
`topic` is in the query's `relevant_topics`. For a corpus fetched from the APIs,
label papers by the seed query that retrieved them (see build_corpus.py --label).
"""

from __future__ import annotations

import json
from pathlib import Path


def precision_at_k(relevance: list[bool], k: int) -> float:
    top = relevance[:k]
    return sum(top) / k if k else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def load_queries(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def evaluate(retriever, queries: list[dict], ks=(1, 3, 5)) -> dict:
    per_query = []
    for q in queries:
        rel_topics = set(q["relevant_topics"])
        hits = retriever.search(q["query"], k=max(ks))
        rel = [h.paper.topic in rel_topics for h in hits]
        row = {"query": q["query"], "rr": reciprocal_rank(rel)}
        for k in ks:
            row[f"p@{k}"] = precision_at_k(rel, k)
        per_query.append(row)
    n = len(per_query)
    summary = {f"P@{k}": sum(r[f"p@{k}"] for r in per_query) / n for k in ks}
    summary["MRR"] = sum(r["rr"] for r in per_query) / n
    return {"summary": summary, "per_query": per_query}


def compare(retrievers: list, queries: list[dict], ks=(1, 3, 5)) -> str:
    results = {r.name: evaluate(r, queries, ks) for r in retrievers}
    cols = [f"P@{k}" for k in ks] + ["MRR"]
    lines = ["| Method | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{res['summary'][c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)
