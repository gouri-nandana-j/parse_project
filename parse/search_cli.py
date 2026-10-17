"""Command-line search and evaluation.

    python search_cli.py search "how do machines learn from few examples" -k 5
    python search_cli.py eval
"""

import argparse

from parse.corpus import load_corpus
from parse.evaluate import compare, load_queries
from parse.retrievers import SbertRetriever, TfidfRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["search", "eval"])
    ap.add_argument("query", nargs="?")
    ap.add_argument("--corpus", default="data/sample_papers.json")
    ap.add_argument("--queries", default="data/eval_queries.json")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    papers = load_corpus(args.corpus)
    tfidf = TfidfRetriever(papers)
    sbert = SbertRetriever(papers, args.model, cache_dir="data/cache")

    if args.command == "eval":
        print(compare([tfidf, sbert], load_queries(args.queries)))
        return
    if not args.query:
        ap.error("search needs a query")
    for r in (sbert, tfidf):
        print(f"\n== {r.name} ==")
        for h in r.search(args.query, args.k):
            print(f"{h.rank:>2}. [{h.score:.3f}] {h.paper.title} ({h.paper.year})")


if __name__ == "__main__":
    main()
