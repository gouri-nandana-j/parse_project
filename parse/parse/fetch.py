"""Fetch paper metadata and abstracts from arXiv and Semantic Scholar."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests

from .corpus import Paper, dedupe

ARXIV_URL = "http://export.arxiv.org/api/query"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
HEADERS = {"User-Agent": "PARSE-student-project/1.0"}


def fetch_arxiv(query: str, max_results: int = 50, timeout: int = 30) -> list[Paper]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    r = requests.get(ARXIV_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    papers = []
    for e in root.findall("a:entry", ATOM):
        link = (e.findtext("a:id", "", ATOM) or "").strip()
        published = e.findtext("a:published", "", ATOM) or ""
        papers.append(
            Paper(
                id="arxiv:" + link.rsplit("/", 1)[-1],
                title=" ".join((e.findtext("a:title", "", ATOM) or "").split()),
                abstract=" ".join((e.findtext("a:summary", "", ATOM) or "").split()),
                authors=[a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                url=link,
                source="arxiv",
            )
        )
    return papers


def fetch_semantic_scholar(query: str, limit: int = 50, timeout: int = 30) -> list[Paper]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,authors,year,url,externalIds",
    }
    r = requests.get(S2_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    papers = []
    for d in r.json().get("data", []):
        if not d.get("abstract"):
            continue
        papers.append(
            Paper(
                id="s2:" + d["paperId"],
                title=d.get("title") or "",
                abstract=" ".join(d["abstract"].split()),
                authors=[a.get("name", "") for a in d.get("authors", [])],
                year=d.get("year"),
                url=d.get("url") or "",
                source="semantic_scholar",
            )
        )
    return papers


def build_corpus(queries: list[str], per_query: int = 50, sources=("arxiv", "s2"),
                 delay: float = 3.0) -> list[Paper]:
    """Run each query against each source and merge. `delay` keeps us polite to the APIs."""
    papers: list[Paper] = []
    for q in queries:
        for src in sources:
            try:
                if src == "arxiv":
                    papers += fetch_arxiv(q, per_query)
                elif src == "s2":
                    papers += fetch_semantic_scholar(q, per_query)
            except requests.RequestException as exc:
                print(f"[warn] {src} failed for {q!r}: {exc}")
            time.sleep(delay)
    return dedupe(papers)
