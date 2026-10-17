"""Diagnose Semantic Scholar access.   Run:  python check_s2.py"""
import time

import requests

import config

key = config.SEMANTIC_SCHOLAR_API_KEY
print(f"API key loaded from .env: {'YES' if key else 'NO'}" + (f" (length {len(key)}, starts '{key[:4]}...')" if key else ""))
url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {"query": "semantic search research papers", "limit": 1, "fields": "title"}
for label, headers in (("WITH key", {"x-api-key": key}), ("WITHOUT key", {})):
    if label == "WITH key" and not key:
        continue
    time.sleep(1.5)  # stay under 1 request/second
    r = requests.get(url, params=params, headers=headers, timeout=30)
    print(f"\n[{label}] HTTP {r.status_code}")
    print("  Retry-After:", r.headers.get("Retry-After"))
    print("  body:", r.text[:200].replace("\n", " "))
print("\n200 = works | 403 = key rejected (re-copy it) | 429 = rate limited (wait a minute, don't run other searches)")