"""Download a paper corpus from arXiv / Semantic Scholar.

Example:
    python build_corpus.py --queries "graph neural networks" "protein structure prediction" \
        --per-query 60 --out data/corpus.json

With --label, each paper's `topic` is set to the query that retrieved it, so the
corpus can be used with evaluate_cli.py (relevant_topics = list of seed queries).
"""

import argparse

import config
from parse.corpus import save_corpus
from parse.fetch import build_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--sources", nargs="+", default=["arxiv", "s2"], choices=["arxiv", "s2", "openalex"])
    ap.add_argument("--out", default="data/corpus.json")
    ap.add_argument("--label", action="store_true", help="tag papers with their seed query")
    args = ap.parse_args()

    papers = []
    for q in args.queries:
        got = build_corpus([q], args.per_query, tuple(args.sources),
                           api_key=config.SEMANTIC_SCHOLAR_API_KEY, email=config.OPENALEX_EMAIL)
        if args.label:
            for p in got:
                p.topic = q
        papers += got
    # de-duplicate across queries, keeping first label
    seen, unique = set(), []
    for p in papers:
        key = p.title.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    save_corpus(unique, args.out)
    print(f"Saved {len(unique)} papers to {args.out}")


if __name__ == "__main__":
    main()