import numpy as np
from parse.corpus import load_corpus, Paper, dedupe
from parse.evaluate import precision_at_k, reciprocal_rank, evaluate, load_queries
from parse.retrievers import TfidfRetriever, SbertRetriever


class FakeModel:
    """Deterministic bag-of-characters embedder standing in for SBERT (offline tests)."""
    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def test_metrics():
    assert precision_at_k([True, False, True], 3) == 2 / 3
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([False]) == 0.0


def test_dedupe():
    ps = [Paper("1", "A  Title", "x"), Paper("2", "a title", "y"), Paper("3", "B", "")]
    assert [p.id for p in dedupe(ps)] == ["1"]


def test_tfidf_finds_keyword_match():
    papers = load_corpus("data/sample_papers.json")
    hits = TfidfRetriever(papers).search("speech recognition", k=3)
    assert hits[0].paper.topic == "asr"
    assert hits[0].score >= hits[1].score


def test_sbert_pipeline_with_stub(tmp_path):
    papers = load_corpus("data/sample_papers.json")
    r = SbertRetriever(papers, model=FakeModel(), cache_dir=tmp_path)
    assert r.embeddings.shape[0] == len(papers)
    r2 = SbertRetriever(papers, model=FakeModel(), cache_dir=tmp_path)  # cache hit
    assert np.allclose(r.embeddings, r2.embeddings)
    assert len(r.search("graph", k=4)) == 4


def test_evaluate_runs():
    papers = load_corpus("data/sample_papers.json")
    res = evaluate(TfidfRetriever(papers), load_queries("data/eval_queries.json"))
    assert 0 <= res["summary"]["MRR"] <= 1
