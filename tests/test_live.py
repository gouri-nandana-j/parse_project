import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unittest import mock
from parse import fetch
from parse.corpus import Paper
from parse.preprocess import split_chunks

XML = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/1234.5678v1</id><published>2020-03-01T00:00:00Z</published>
<title>A  Test
Paper</title><summary> Some   abstract. </summary>
<author><name>Ann</name></author><author><name>Bob</name></author></entry></feed>"""

IDEA = ("I want to build a tool that helps students find research papers. "
        "Keyword search misses papers that use different terminology for the same concept. "
        "So the tool should compare the meaning of my idea with paper abstracts using sentence embeddings. "
        "It should then rank the abstracts by similarity and give links to the papers.")


def test_keyword_terms_drops_stopwords():
    assert fetch.keyword_terms("how can a model recognise new classes") == ["model", "recognise", "new", "classes"]


def test_arxiv_query_is_broad():
    q = fetch.arxiv_search_query("speech recognition")
    assert 'all:"speech recognition"' in q and "all:speech" in q and " OR " in q


def test_short_input_is_one_query():
    assert fetch.build_search_queries("graph neural networks") == ["graph neural networks"]


def test_long_idea_gives_several_queries():
    qs = fetch.build_search_queries(IDEA, max_queries=4)
    assert 2 <= len(qs) <= 4 and len(set(qs)) == len(qs)
    assert all(q.split() for q in qs)
    assert any("keyword" in q or "search" in q for q in qs)


def test_parse_arxiv():
    p = fetch.parse_arxiv(XML)[0]
    assert (p.id, p.title, p.abstract, p.year) == ("arxiv:1234.5678v1", "A Test Paper", "Some abstract.", 2020)
    assert p.authors == ["Ann", "Bob"]


def test_openalex_abstract_and_link():
    data = {"results": [
        {"id": "https://openalex.org/W1", "title": "T", "publication_year": 2021,
         "doi": "https://doi.org/10.1/x",
         "abstract_inverted_index": {"semantic": [1], "Dense": [0], "retrieval": [2]},
         "authorships": [{"author": {"display_name": "Ann"}}]},
        {"id": "https://openalex.org/W2", "title": "no abstract", "abstract_inverted_index": None},
    ]}
    papers = fetch.parse_openalex(data)
    assert len(papers) == 1
    assert papers[0].abstract == "Dense semantic retrieval"
    assert papers[0].url == "https://doi.org/10.1/x" and papers[0].authors == ["Ann"]


def test_split_chunks():
    chunks = split_chunks(IDEA, max_words=20)
    assert len(chunks) > 1 and " ".join(chunks).split() == IDEA.split()


def test_fetch_live_survives_failures():
    with mock.patch.object(fetch, "fetch_arxiv", return_value=[Paper("1", "T", "abs")]), \
         mock.patch.object(fetch, "fetch_openalex", return_value=[Paper("2", "T2", "abs2")]), \
         mock.patch.object(fetch, "fetch_semantic_scholar", side_effect=Exception("429 Too Many")), \
         mock.patch.object(fetch.time, "sleep"):
        papers, warns, queries = fetch.fetch_live(IDEA)
    assert {p.id for p in papers} == {"1", "2"}
    assert len(warns) == 1 and "API key" in warns[0] and len(queries) >= 2


def test_semantic_scholar_sends_key_and_respects_rate_limit():
    class Resp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self): return {"data": [{"paperId": "x", "title": "T", "abstract": "a b",
                                           "authors": [{"name": "A"}], "year": 2020, "url": "u"}]}
    seen = []
    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append((fetch.time.monotonic(), headers.get("x-api-key")))
        return Resp()
    with mock.patch.object(fetch.requests, "get", fake_get):
        fetch.fetch_semantic_scholar("q", 5, api_key="KEY")
        fetch.fetch_semantic_scholar("q", 5, api_key="KEY")
    assert all(k == "KEY" for _, k in seen)
    assert seen[1][0] - seen[0][0] > 1.0  # Semantic Scholar limit: 1 request per second