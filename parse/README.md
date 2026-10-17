# PARSE – Paper Abstract Retrieval using Semantic Embeddings

Semantic search over research-paper abstracts. Queries and abstracts are embedded with
Sentence-BERT and ranked by cosine similarity; a TF-IDF retriever is the keyword baseline.
Both are evaluated with Precision@K and MRR.

Team: Gagandeep M (229), Gouri Nandana J (232), Hrishikesh V M (234), Negha R (249) · Batch R7A · Guide: Prof. Rejimoan R

## Layout

| File | Purpose |
|---|---|
| `parse/fetch.py` | arXiv and Semantic Scholar API clients |
| `parse/corpus.py` | `Paper` model, JSON save/load, de-duplication |
| `parse/preprocess.py` | text cleaning for TF-IDF |
| `parse/retrievers.py` | `TfidfRetriever` (baseline), `SbertRetriever` (semantic, embedding cache) |
| `parse/evaluate.py` | Precision@K, MRR, comparison table |
| `build_corpus.py` | download a real corpus |
| `search_cli.py` | command-line search / evaluation |
| `app.py` | Streamlit search interface |
| `make_sample_data.py` | regenerates the offline demo data |
| `tests/` | pytest suite |

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace.

## Usage

```bash
# 1. (optional) build a real corpus; --label tags papers with their seed query
python build_corpus.py --queries "few-shot learning" "graph neural networks" \
    "protein structure prediction" --per-query 60 --label --out data/corpus.json

# 2. search from the terminal (SBERT and TF-IDF side by side)
python search_cli.py search "how can a model recognise objects it has barely seen" --corpus data/corpus.json

# 3. evaluate (P@1/3/5 and MRR)
python search_cli.py eval

# 4. web interface
streamlit run app.py
```

## Evaluation

Relevance is defined by topic labels: a paper is relevant to a query if its `topic` is listed
in the query's `relevant_topics` (`data/eval_queries.json`). For a corpus from `build_corpus.py --label`,
the topic is the seed query, so write queries with `relevant_topics` set to those seed strings.

`data/sample_papers.json` is a small **demo corpus of 32 illustrative abstracts written for this
project (not real publications)**, so the pipeline runs offline. Its queries are mostly paraphrases
with little keyword overlap, which is where semantic retrieval should beat TF-IDF. For
your report, evaluate on a real corpus with more papers and queries.

TF-IDF on the demo set: P@1 0.600, P@3 0.367, P@5 0.280, MRR 0.650.
