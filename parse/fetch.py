"""Fetch paper metadata and abstracts from arXiv, Semantic Scholar and OpenAlex.

PARSE cannot download "every paper on the web" - nobody can. Instead it asks these large
indexes for a pool of keyword-matched candidates (several searches per pasted idea), and the
SBERT model then ranks that pool by meaning.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .corpus import Paper, dedupe
from .preprocess import split_sentences

ARXIV_URL = "https://export.arxiv.org/api/query"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
HEADERS = {"User-Agent": "PARSE-student-project/1.0"}

# Words that are common in research writing but say nothing about the topic.
_FILLER = {"paper", "propose", "proposed", "approach", "method", "methods", "use", "using",
           "used", "based", "study", "work", "show", "results", "idea", "project", "system",
           "want", "need", "like", "also", "new", "novel", "good", "make", "build"}

# Verbs/filler from phrasing like "I want to build a system that uses ..." - never topic words.
_QUERY_NOISE = {"use", "uses", "using", "used", "want", "wants", "need", "needs", "like",
                "idea", "build", "make", "makes", "help", "helps", "find", "finds", "finding"}


def keyword_terms(text: str, max_terms: int = 8) -> list[str]:
    """Content words of a query (stop-words removed, order kept)."""
    seen, out = set(), []
    for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower()):
        if w in ENGLISH_STOP_WORDS or w in _QUERY_NOISE or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out[:max_terms]


def _top_terms(text: str, n: int) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
             if w not in ENGLISH_STOP_WORDS and w not in _FILLER and w not in _QUERY_NOISE]
    return [w for w, _ in Counter(words).most_common(n)]


def _phrase_runs(text: str, max_words: int = 4) -> list[str]:
    """Runs of consecutive content words, e.g. 'procedural content generation'. Stop-words,
    filler verbs and punctuation break a run. Longest phrases first."""
    runs, cur = [], []
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]*|[.,;:!?()]", text.lower()):
        if len(tok) < 3 or tok in ENGLISH_STOP_WORDS or tok in _QUERY_NOISE:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if len(cur) >= 2:
        runs.append(cur)
    runs.sort(key=len, reverse=True)  # stable: ties keep text order
    return [" ".join(r[:max_words]) for r in runs]


def build_search_queries(text: str, max_queries: int = 4) -> list[str]:
    """Turn a topic or pasted idea into keyword searches covering different parts of it:
    the whole text's most frequent terms, the key phrases in it, and one per section."""
    text = " ".join(text.split())
    runs = _phrase_runs(text)
    if len(text.split()) <= 12:
        qs = [text] + runs
    else:
        sents = split_sentences(text) or [text]
        n_groups = min(3, len(sents))
        size = -(-len(sents) // n_groups)  # ceil division
        groups = [" ".join(_top_terms(" ".join(sents[i:i + size]), 5))
                  for i in range(0, len(sents), size)]
        qs = [" ".join(_top_terms(text, 6))] + runs[:1] + groups + runs[1:]
    out: list[str] = []
    for q in qs:
        q = q.strip()
        words = set(q.split())
        # skip empty searches and near-duplicates of one already chosen (they add no coverage)
        if q and all(len(words & set(o.split())) / len(words | set(o.split())) < 0.6 for o in out):
            out.append(q)
    return out[:max_queries]


def arxiv_search_query(text: str) -> str:
    """Broad (OR-joined) arXiv query so natural-language input still returns candidates;
    SBERT then does the precise ranking."""
    terms = keyword_terms(text)
    if not terms:
        return f'all:"{text.strip()}"'
    parts = [f"all:{t}" for t in terms]
    if 2 <= len(terms) <= 4 and len(text.split()) <= 4:  # a short typed topic: also try the phrase
        parts.insert(0, 'all:"' + " ".join(terms) + '"')
    return " OR ".join(parts)


def parse_arxiv(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
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


def fetch_arxiv(query: str, max_results: int = 50, timeout: int = 30,
                broad: bool = False) -> list[Paper]:
    params = {
        "search_query": arxiv_search_query(query) if broad else f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    r = requests.get(ARXIV_URL, params=params, headers=HEADERS, timeout=timeout)
    if r.status_code == 429:  # arXiv throttles bursts; wait and retry once
        time.sleep(6)
        r = requests.get(ARXIV_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return parse_arxiv(r.text)


class _RateLimiter:
    """Thread-safe limiter: guarantees a minimum gap between the *starts* of requests.

    Semantic Scholar allows 1 request/second cumulative across all endpoints, so every call to
    their API goes through one shared limiter (searches, deep-search expansion, build_corpus.py,
    and several browser sessions at once all share it).
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_ok - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_ok = now + self.min_interval


# 1.25 s gap = 0.8 requests/second, safely below Semantic Scholar's 1 request/second limit.
S2_LIMITER = _RateLimiter(1.25)


def fetch_semantic_scholar(query: str, limit: int = 50, timeout: int = 30,
                           api_key: str = "") -> list[Paper]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,authors,year,url,externalIds",
    }
    headers = dict(HEADERS)
    if api_key:
        headers["x-api-key"] = api_key  # header name is case-sensitive
    for attempt in range(4):
        S2_LIMITER.wait()
        r = requests.get(S2_URL, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429 and attempt < 3:  # still throttled: back off, then retry
            retry_after = r.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 3 * (attempt + 1))
            continue
        break
    r.raise_for_status()
    papers = []
    for d in r.json().get("data") or []:
        if not d.get("abstract") or not d.get("paperId"):
            continue
        papers.append(
            Paper(
                id="s2:" + d["paperId"],
                title=d.get("title") or "",
                abstract=" ".join(d["abstract"].split()),
                authors=[a.get("name") or "" for a in d.get("authors") or []],
                year=d.get("year"),
                url=d.get("url") or "",
                source="semantic_scholar",
            )
        )
    return papers


def abstract_from_inverted(index: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the plain text."""
    if not index:
        return ""
    slots: dict[int, str] = {}
    for word, positions in index.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def parse_openalex(data: dict) -> list[Paper]:
    papers = []
    for d in data.get("results", []):
        abstract = abstract_from_inverted(d.get("abstract_inverted_index"))
        if not abstract:
            continue
        loc = d.get("primary_location") or {}
        url = d.get("doi") or loc.get("landing_page_url") or d.get("id") or ""
        papers.append(
            Paper(
                id="oa:" + (d.get("id") or "").rsplit("/", 1)[-1],
                title=" ".join((d.get("title") or d.get("display_name") or "").split()),
                abstract=abstract,
                authors=[(a.get("author") or {}).get("display_name", "")
                         for a in d.get("authorships", [])],
                year=d.get("publication_year"),
                url=url,
                source="openalex",
            )
        )
    return papers


def fetch_openalex(query: str, limit: int = 50, timeout: int = 30, email: str = "") -> list[Paper]:
    params = {
        "search": query,
        "filter": "has_abstract:true",
        "per-page": min(limit, 100),
        "select": "id,title,display_name,abstract_inverted_index,authorships,"
                  "publication_year,doi,primary_location",
    }
    if email:  # optional: OpenAlex gives faster service to identified ("polite") users
        params["mailto"] = email
    r = requests.get(OPENALEX_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return parse_openalex(r.json())


def _run_source(name: str, queries: list[str], per_query: int, pause: float, call,
                has_key: bool = False) -> tuple[list[Paper], list[str]]:
    """Run every query against one source, politely spaced; one failure doesn't stop the rest."""
    papers: list[Paper] = []
    warnings: list[str] = []
    for i, q in enumerate(queries):
        if i:
            time.sleep(pause)
        try:
            papers += call(q, per_query)
        except Exception as exc:
            hint = ""
            if name == "Semantic Scholar" and "429" in str(exc):
                hint = " - rate limited, try again in a minute" if has_key else " - add an API key (see README)"
            warnings.append(f"{name} unavailable: {exc}{hint}")
            break  # if a source is failing/limiting us, don't hammer it with more queries
    return papers, warnings


def fetch_queries(queries: list[str], per_source: int = 50, api_key: str = "", email: str = "",
                  sources=("arxiv", "s2", "openalex")) -> tuple[list[Paper], list[str]]:
    """Run the given searches on every source (sources in parallel, queries spaced politely).

    Returns (papers, warnings). A failing source yields a warning, not an exception.
    """
    per_query = max(20, -(-per_source // max(len(queries), 1)))
    jobs = []
    if "arxiv" in sources:
        jobs.append(("arXiv", 3.0, lambda q, n: fetch_arxiv(q, n, broad=True)))
    if "s2" in sources:
        jobs.append(("Semantic Scholar", 0.0,  # spacing is enforced by S2_LIMITER
                     lambda q, n: fetch_semantic_scholar(" ".join(keyword_terms(q)) or q, n, api_key=api_key)))
    if "openalex" in sources:
        jobs.append(("OpenAlex", 0.3, lambda q, n: fetch_openalex(" ".join(keyword_terms(q)) or q, n, email=email)))

    papers: list[Paper] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
        futures = [pool.submit(_run_source, name, queries, per_query, pause, call, bool(api_key))
                   for name, pause, call in jobs]
        for f in futures:
            p, w = f.result()
            papers += p
            warnings += w
    return dedupe(papers), warnings


def fetch_live(text: str, per_source: int = 50, api_key: str = "", email: str = "",
               sources=("arxiv", "s2", "openalex"), max_queries: int = 4
               ) -> tuple[list[Paper], list[str], list[str]]:
    """Candidate pool for a typed topic or pasted idea: (papers, warnings, queries_used)."""
    queries = build_search_queries(text, max_queries)
    papers, warnings = fetch_queries(queries, per_source, api_key, email, sources)
    return papers, warnings, queries


def build_corpus(queries: list[str], per_query: int = 50, sources=("arxiv", "s2"),
                 delay: float = 3.0, api_key: str = "", email: str = "") -> list[Paper]:
    """Run each query against each source and merge (used by build_corpus.py)."""
    papers: list[Paper] = []
    for q in queries:
        for src in sources:
            try:
                if src == "arxiv":
                    papers += fetch_arxiv(q, per_query)
                elif src == "s2":
                    papers += fetch_semantic_scholar(q, per_query, api_key=api_key)
                elif src == "openalex":
                    papers += fetch_openalex(q, per_query, email=email)
            except requests.RequestException as exc:
                print(f"[warn] {src} failed for {q!r}: {exc}")
            time.sleep(delay)
    return dedupe(papers)