# PARSE – Paper Abstract Retrieval using Semantic Embeddings

Semantic search over research-paper abstracts. Queries and abstracts are embedded with
Sentence-BERT and ranked by cosine similarity; a TF-IDF retriever is the keyword baseline.
Both are evaluated with Precision@K and MRR.

## Quick start (web app)

1. **Install** (PowerShell, inside this folder):
   ```powershell
   pip install -r requirements.txt
   ```
2. **Create your settings file.** Copy `.env.example` to a new file named exactly `.env`:
   ```powershell
   copy .env.example .env
   ```
3. **Add your Semantic Scholar API key (placeholder 1).** Open `.env` and replace
   `PASTE_YOUR_SEMANTIC_SCHOLAR_KEY_HERE` with your key (no quotes). Get a free key at
   https://www.semanticscholar.org/product/api#api-key-form . Without it the app still works but
   Semantic Scholar is often rate-limited (HTTP 429) and the app falls back to arXiv + OpenAlex.
   `OPENALEX_EMAIL` in the same file is optional (placeholder 2): an email you're happy to share
   with OpenAlex for faster service.
4. **Run:**
   ```powershell
   python -m streamlit run app.py
   ```
   Open the address it prints (usually http://localhost:8501). The first run downloads the SBERT
   model (~90 MB), so it needs internet and takes a minute.
5. **Use it:** paste a paragraph describing your research idea (or type a topic) and press
   **Find papers**. You get a ranked list with similarity scores, abstracts and an **Open paper**
   link for each result.

`.env` is in `.gitignore`, so your key is never committed.

### How it finds papers
PARSE can't download "every paper on the web". For your text it (1) derives several keyword
searches covering different parts of the idea, (2) pulls candidate papers with abstracts from
arXiv, Semantic Scholar and OpenAlex, then (3) ranks them by SBERT meaning-similarity to your
full text (long ideas are embedded in chunks). The TF-IDF keyword ranking is shown alongside
for comparison. The searches it ran are listed in the app under "Searches PARSE ran".

## Layout

| File | Purpose |
|---|---|
| `parse/fetch.py` | arXiv, Semantic Scholar and OpenAlex clients |
| `parse/corpus.py` | `Paper` model, JSON save/load, de-duplication |
| `parse/preprocess.py` | text cleaning for TF-IDF |
| `parse/retrievers.py` | `TfidfRetriever` (baseline), `SbertRetriever` (semantic, embedding cache) |
| `parse/evaluate.py` | Precision@K, MRR, comparison table |
| `build_corpus.py` | download a real corpus |
| `search_cli.py` | command-line search / evaluation |
| `app.py` | Streamlit web app (live search or local corpus) |
| `config.py` | reads settings and the API key from `.env` |
| `.env.example` | template for `.env` (copy it; holds the API-key placeholder) |
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
