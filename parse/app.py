"""Streamlit search interface:  streamlit run app.py"""

import json
from pathlib import Path

import streamlit as st

from parse.corpus import load_corpus
from parse.retrievers import SbertRetriever, TfidfRetriever

st.set_page_config(page_title="PARSE", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading corpus and embedding model...")
def load(corpus_path: str):
    papers = load_corpus(corpus_path)
    return papers, SbertRetriever(papers, cache_dir="data/cache"), TfidfRetriever(papers)


st.title("PARSE")
st.caption("Paper Abstract Retrieval using Semantic Embeddings")

with st.sidebar:
    corpus_files = sorted(str(p) for p in Path("data").glob("*.json") if "eval" not in p.name)
    corpus_path = st.selectbox("Corpus", corpus_files or ["data/sample_papers.json"])
    method = st.radio("Retrieval method", ["SBERT (semantic)", "TF-IDF (keyword)", "Compare both"])
    k = st.slider("Results", 3, 20, 5)

papers, sbert, tfidf = load(corpus_path)
st.sidebar.write(f"{len(papers)} papers indexed")

query = st.text_input("Research topic or question",
                      placeholder="e.g. teaching models to recognise new classes from very few examples")


def show(retriever):
    st.subheader(retriever.name)
    for h in retriever.search(query, k):
        p = h.paper
        with st.container(border=True):
            title = f"[{p.title}]({p.url})" if p.url else p.title
            st.markdown(f"**{h.rank}. {title}**")
            st.caption(f"Similarity: {h.score:.3f} · {', '.join(p.authors[:3])} · {p.year or ''}")
            with st.expander("Abstract"):
                st.write(p.abstract)


if query:
    if method.startswith("Compare"):
        c1, c2 = st.columns(2)
        with c1:
            show(sbert)
        with c2:
            show(tfidf)
    else:
        show(sbert if method.startswith("SBERT") else tfidf)
