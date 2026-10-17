# PARSE — Paper Abstract Retrieval using Semantic Embeddings

PARSE is an NLP-based research-paper discovery tool. A user describes a research idea, or types a
topic, and PARSE returns a ranked list of relevant papers from open academic sources, each with a
similarity score, its abstract and a link to the paper.

Traditional literature search depends on exact keyword overlap, so relevant work is missed whenever
it uses different terminology for the same concept. PARSE addresses this by comparing the *meaning*
of the user's text with the meaning of each paper abstract, using Sentence-BERT embeddings and cosine
similarity. A TF-IDF keyword retriever is included as a baseline so the two approaches can be compared
directly, both in the web app and through Precision@K and Mean Reciprocal Rank (MRR).

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Transformer model](#transformer-model)
4. [Project structure](#project-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the web app](#running-the-web-app)
8. [Using the app](#using-the-app)
9. [Command-line tools](#command-line-tools)
10. [Evaluation](#evaluation)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Limitations](#limitations)

---

## Features

- **Semantic search.** Queries and abstracts are embedded with Sentence-BERT (`all-MiniLM-L6-v2`) and
  ranked by cosine similarity, so relevant papers surface even when they share no keywords with the query.
- **Free-form input.** Accepts a short topic or a full pasted paragraph. Long text is split into chunks and
  embedded piece by piece instead of being silently truncated.
- **Multiple open sources.** Candidate papers are gathered from arXiv, Semantic Scholar and OpenAlex and
  de-duplicated by title.
- **Keyword baseline.** A TF-IDF retriever can be shown beside the SBERT ranking for side-by-side comparison.
- **Adjustable search depth.** *Quick*, *Standard* and *Deep* modes trade speed for coverage.
- **Sorting and export.** Results can be ordered by relevance, newest first or oldest first, and downloaded
  as CSV (including abstracts).
- **Insights.** A similarity-by-rank chart and a clickable papers-by-year chart that lists the papers of any
  selected year.
- **Offline mode.** A small local corpus lets the whole pipeline run without internet access (after the
  model has been downloaded once).
- **Evaluation harness.** Precision@K and MRR over labelled queries, for SBERT and TF-IDF.

## How it works

PARSE cannot download "every paper on the web". Instead it follows a retrieve-then-rank pipeline:

1. **Query building.** The input text is turned into several keyword searches that cover different parts of
   the idea (key phrases and the most informative terms per section).
2. **Candidate retrieval.** Each search is sent to arXiv, Semantic Scholar and OpenAlex. Papers without an
   abstract are discarded and duplicates are removed.
3. **Semantic ranking.** Every abstract (title + abstract) and the user's full text are encoded with SBERT.
   Papers are ranked by cosine similarity to the query. For long inputs, the score combines the best-matching
   and the average-matching chunk.
4. **Baseline ranking.** The same candidates are scored with TF-IDF (unigrams and bigrams) for comparison.
5. **Deep mode (optional).** The titles of the three best matches are used as extra searches, which finds
   additional related papers.

## Transformer model

PARSE uses **`all-MiniLM-L6-v2`**, a pretrained Sentence-BERT model distributed through the
`sentence-transformers` library (Hugging Face id `sentence-transformers/all-MiniLM-L6-v2`). The model name is
set in one place, `SBERT_MODEL` in `config.py`.

| Property | Value |
|---|---|
| Architecture | MiniLM transformer encoder, 6 layers |
| Output | One 384-dimensional embedding per text |
| Size | About 22 million parameters (roughly a 90 MB download) |
| Training | Contrastive fine-tuning on over 1 billion sentence pairs, so texts with similar meaning get nearby vectors |
| Input limit | About 256 word pieces; longer text is truncated |
| Similarity measure | Cosine similarity (embeddings are normalised, so the dot product equals the cosine) |

**How PARSE uses it**

1. Each paper's title and abstract is encoded into a single vector.
2. The user's idea is encoded the same way. Long ideas are split into chunks and each chunk is encoded
   separately.
3. Papers are ranked by the cosine similarity between their vector and the query vector.

**Why this model.** It is small and fast enough to embed 200 or more abstracts per search on an ordinary
laptop CPU, and it is a widely used, well-tested general-purpose model for semantic search.

**Known limitations.** Abstracts longer than the input limit are truncated, and the model is trained on
general text rather than scientific papers specifically.

**Changing the model.** Edit `SBERT_MODEL` in `config.py` for the web app, or pass `--model` to
`search_cli.py`. Cached embeddings are stored per model name, so switching models never mixes vectors. The new
model is downloaded on first use. Two alternatives worth comparing with `python search_cli.py eval`:

- `all-mpnet-base-v2` – generally more accurate, but larger and slower.
- `allenai-specter` – trained on scientific papers and their citation links, which suits this domain.
  It has not been tested with this project.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `config.py` | Reads settings and API keys from `.env` |
| `parse/fetch.py` | arXiv, Semantic Scholar and OpenAlex clients, query building, Semantic Scholar rate limiter |
| `parse/corpus.py` | `Paper` data model, JSON save/load, de-duplication |
| `parse/preprocess.py` | Text cleaning and chunking |
| `parse/retrievers.py` | `TfidfRetriever` (baseline) and `SbertRetriever` (semantic, with embedding cache) |
| `parse/evaluate.py` | Precision@K, MRR and the comparison table |
| `search_cli.py` | Command-line search and evaluation |
| `build_corpus.py` | Downloads a real corpus from the APIs |
| `make_sample_data.py` | Regenerates the offline demo corpus and evaluation queries |
| `check_s2.py` | Diagnoses Semantic Scholar API access |
| `data/` | Demo corpus (`sample_papers.json`) and evaluation queries (`eval_queries.json`) |
| `tests/` | Automated test suite |
| `.streamlit/config.toml` | App theme (colours) |
| `.env.example` | Template for the `.env` settings file |

## Installation

**Requirements:** Python 3.10 or newer and an internet connection (for the first model download and for
live search).

```powershell
# inside the project folder
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first time the app runs it downloads the `all-MiniLM-L6-v2` model (about 90 MB) from Hugging Face and
caches it for later runs.

> Streamlit 1.46 or newer is required (the interface uses themed sidebars and chart selection events).

## Configuration

Settings are stored in a `.env` file in the project root. The file is listed in `.gitignore` and must
never be committed or shared.

1. Copy the template:
   ```powershell
   copy .env.example .env          # macOS / Linux: cp .env.example .env
   ```
2. Open `.env` and fill in the values:

| Variable | Required | Description |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | Free key from <https://www.semanticscholar.org/product/api#api-key-form>. Paste it without quotes. |
| `OPENALEX_EMAIL` | Optional | An email address you are comfortable sharing with OpenAlex; identified users receive faster service. |

Leaving a value as the `PASTE_YOUR...` placeholder is treated as "not set".

**Semantic Scholar rate limit.** Semantic Scholar allows 1 request per second, cumulative across all
endpoints, and expects the key in the `x-api-key` header. PARSE sends the key in that header and routes every
Semantic Scholar request through a shared limiter that keeps at least 1.25 seconds between requests. A
throttled request (HTTP 429) is retried with a growing delay. Without a key the app still works, but
Semantic Scholar is frequently rate limited and results come mainly from arXiv and OpenAlex.

To verify your key at any time:

```powershell
python check_s2.py
```

`200` means the key works, `403` means it was rejected (re-copy it), and `429` means you are being throttled
(wait a minute).

## Running the web app

```powershell
python -m streamlit run app.py
```

Open the address printed in the terminal (normally <http://localhost:8501>).

## Using the app

1. **Describe your idea.** Type a topic or paste a paragraph into the search box and press **Find papers**.
2. **Read the summary row.** It shows how many papers were ranked, which sources contributed, the best
   similarity score and the number of searches run. Click **Searches run** to see the exact keyword searches
   PARSE issued.
3. **Browse the results.** The *Top results* tab lists the best matches with their similarity score, authors,
   year, source, an expandable abstract and a link to the paper. The *All papers* tab shows every candidate
   in a sortable table. Use **Sort** to order results by relevance, newest first or oldest first, and
   **Download CSV** to export them.
4. **Explore the Insights tab.**
   - *Similarity by rank* plots the SBERT score (dark line) and the TF-IDF score (coral line) for the top
     papers. A steep early drop means only a few strong matches; a gentle slope means many papers are about
     equally relevant. Where the TF-IDF line is low but the SBERT line is high, a paper matches the meaning
     of the idea without sharing its wording.
   - *Papers by year* counts the fetched papers per publication year. Click a bar to list that year's papers.

**Sidebar settings**

| Setting | Effect |
|---|---|
| Live web search | On: fetch fresh abstracts from arXiv, Semantic Scholar and OpenAlex. Off: search the saved corpus in `data/`. |
| Compare with TF-IDF | Show the keyword ranking beside the SBERT ranking. |
| Results to show | Number of top papers displayed as cards. |
| Search depth | **Quick** – 3 searches, fastest. **Standard** – 5 searches, balanced. **Deep** – 8 searches plus searches built from the titles of the 3 best matches (30–60 s). |
| Papers per source | Candidates pulled from each source before ranking. More gives a wider net but a slower search. |

## Command-line tools

```powershell
# Build a real corpus; --label tags each paper with the query that retrieved it
python build_corpus.py --queries "few-shot learning" "graph neural networks" "protein structure prediction" --per-query 60 --label --out data/corpus.json

# Search a corpus from the terminal (SBERT and TF-IDF side by side)
python search_cli.py search "how can a model recognise objects it has barely seen" --corpus data/corpus.json -k 5

# Evaluate Precision@K and MRR
python search_cli.py eval
```

`build_corpus.py` reads the Semantic Scholar key and OpenAlex email from `.env`; `--sources` accepts
`arxiv`, `s2` and `openalex`.

## Evaluation

Retrieval quality is measured with two standard information-retrieval metrics:

- **Precision@K** – the fraction of the top K results that are relevant.
- **Mean Reciprocal Rank (MRR)** – the average of 1 / (rank of the first relevant result) over all queries.

Relevance is defined through topic labels: a paper is relevant to a query when its `topic` appears in that
query's `relevant_topics` (see `data/eval_queries.json`). For a corpus created with
`build_corpus.py --label`, the topic is the seed query that retrieved the paper, so evaluation queries should
list those seed strings.

`data/sample_papers.json` is a small demo corpus of 32 illustrative abstracts written for this project; they
are **not** real publications. The 10 evaluation queries are mostly paraphrases with little keyword overlap,
which is where semantic retrieval is expected to outperform keyword matching. For the project report,
evaluate on a larger real corpus with more queries.

Keyword baseline on the demo set:

| Method | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.600 | 0.367 | 0.280 | 0.650 |

Run `python search_cli.py eval` to produce the SBERT row alongside the TF-IDF row.

## Testing

```powershell
pip install pytest
python -m pytest
```

The tests cover the metrics, de-duplication, query building, the arXiv and OpenAlex parsers, failure
handling when a source is unavailable, and the Semantic Scholar header and rate limit. They run offline and
do not need the SBERT model.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "Semantic Scholar unavailable … 429" | Rate limited. Run `python check_s2.py`, confirm the key is in `.env`, wait a minute and search again. The app continues with arXiv and OpenAlex in the meantime. |
| First search is slow | The SBERT model is downloading (about 90 MB, once only). |
| "Could not load the SBERT model" | No internet connection on first run. Connect and reload. |
| Sidebar is hidden | Click the small arrow at the top left to open it. |
| Few or no results | Try different wording, a higher search depth or more papers per source. |
| Fonts look different | The block-letter font loads from Google Fonts; offline, a system fallback font is used. |

## Limitations

- PARSE ranks a pool of keyword-matched candidates; a relevant paper that none of the keyword searches
  retrieve cannot be ranked.
- Only papers whose source provides an abstract are considered.
- Similarity scores are not probabilities and are best compared within a single search.
- The bundled demo corpus is synthetic and intended only to demonstrate the pipeline offline.
- User accounts and saved search history are planned but not yet implemented.

## Acknowledgements

Paper metadata is provided by the [arXiv API](https://info.arxiv.org/help/api/index.html), the
[Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) and [OpenAlex](https://openalex.org/).
Embeddings use the [Sentence-Transformers](https://www.sbert.net/) library and the
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.