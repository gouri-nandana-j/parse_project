"""PARSE web app.   Run:  python -m streamlit run app.py"""

import html
import time
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
from parse.corpus import dedupe, load_corpus
from parse.fetch import fetch_live, fetch_queries
from parse.retrievers import SbertRetriever, TfidfRetriever

st.set_page_config(page_title="PARSE", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

DARK, CORAL, GREEN, PASTEL = "#272736", "#FA8A6E", "#9BE29B", "#EAF6F0"
BAR_H = 96  # height (px) of the white top bar

# search depth -> (number of searches per source, also search with titles of the best matches?)
DEPTHS = {"Quick": (3, False), "Standard": (5, False), "Deep": (8, True)}
DEPTH_INFO = {"Quick": "3 searches, fastest",
              "Standard": "5 searches, balanced",
              "Deep": "8 searches plus searches built from your 3 best matches (30-60 s)"}
SORTS = ["Relevance", "Newest first", "Oldest first"]

# ---------------------------------------------------------------- look & feel
_ICONS = {  # minimal line icons (Feather style)
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
}


def icon(name: str, color: str = CORAL, size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')


BLOCK = "'Bebas Neue', Impact, 'Arial Narrow', sans-serif"
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');
.stApp {{ font-family:'Inter', sans-serif; }}

/* the container that carries this <style> block must not take up space above the brand bar */
[data-testid="stElementContainer"]:has(style), .element-container:has(style) {{ display:none !important; }}

/* LAYER 0 - pastel green, behind everything */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:{PASTEL} !important; }}
/* LAYER 1 - white strip across the very top (the sidebar sits above it, so it only shows beside the sidebar) */
.stApp::before {{ content:''; position:fixed; top:0; left:0; right:0; height:{BAR_H}px; background:#fff;
    box-shadow:0 1px 6px rgba(39,39,54,.10); z-index:999980; }}
header[data-testid="stHeader"] {{ position:fixed !important; top:0; left:0; right:0; height:0 !important; min-height:0 !important;
    overflow:visible !important; background:transparent !important; }}
[data-testid="stDecoration"] {{ display:none !important; }}
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDeployButton"], .stDeployButton,
#MainMenu, footer {{ display:none !important; }}   /* NB: the sidebar open/close arrow is left alone */
section[data-testid="stSidebar"] {{ background:{DARK} !important; z-index:999991 !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding:{BAR_H + 4}px 2rem 3rem !important; max-width:100% !important; }}

/* brand bar: sits on the white strip and stays there while scrolling */
.topbar {{ height:0; margin:0; padding:0; }}
.tl, .acct {{ position:fixed; top:0; height:{BAR_H}px; display:flex; align-items:center; gap:12px; z-index:999985; }}
.tl {{ left:auto; margin-left:1rem; }}   /* left:auto = stays at the left edge of the main area, wherever the sidebar is */
.acct {{ right:2rem; left:auto; }}
.logo {{ position:relative; width:44px; height:44px; flex:none; }}
.logo:before {{ content:''; position:absolute; inset:0; border-radius:50%; background:{CORAL}; }}
.logo span {{ position:absolute; left:0; top:0; width:40px; height:44px; border-radius:50%; background:{DARK}; color:{CORAL};
    font:400 27px {BLOCK}; display:flex; align-items:center; justify-content:center; }}
.brand {{ font:400 36px/1 {BLOCK}; letter-spacing:.04em; color:{DARK}; }}
.tag {{ font-size:12px; color:#7B8794; }}
.avatar {{ width:40px; height:40px; border-radius:50%; background:{CORAL}; display:flex; align-items:center; justify-content:center; }}
.who {{ font:400 20px/1 {BLOCK}; letter-spacing:.04em; color:{DARK}; }}
.soon {{ font-size:11px; color:#7B8794; letter-spacing:.04em; }}

/* LAYER 2 - white panels (matched by container key, so they work in every Streamlit version) */
.st-key-search_panel, .st-key-how_panel, .st-key-results_panel, [class*="st-key-icard"] {{
    background:#fff; border:1px solid #DCEBE5; border-radius:10px; padding:1rem; overflow:hidden;
    box-shadow:0 1px 6px rgba(39,39,54,.07); }}
[class*="st-key-rcard"] {{ background:#F6FBF8; border:1px solid #DCEBE5; border-radius:10px; padding:.8rem 1rem; }}
.card-h {{ background:{DARK}; color:#fff; padding:8px 18px; font:400 19px {BLOCK}; letter-spacing:.04em;
    margin:-1rem -1rem .8rem; }}
.card-h.coral {{ background:{CORAL}; color:{DARK}; }}
.sec-h {{ font:400 26px {BLOCK}; letter-spacing:.04em; color:{DARK}; margin:2px 0 6px; }}
.rk {{ color:{CORAL}; font:400 26px {BLOCK}; margin-right:8px; }}
.ptitle {{ font-size:16px; font-weight:600; color:{DARK}; }}
.ptitle a {{ color:{DARK}; text-decoration:none; }} .ptitle a:hover {{ color:{CORAL}; }}
.pill {{ background:{GREEN}; color:{DARK}; border-radius:20px; padding:2px 10px; font-size:12px; font-weight:600; margin-right:8px; }}
button[data-baseweb="tab"] p {{ font:400 21px {BLOCK}; letter-spacing:.05em; }}

/* spinner / loading messages in a small rounded rectangle */
[data-testid="stSpinner"] {{ display:flex !important; align-items:center; gap:10px; width:fit-content; background:#fff; border:1px solid #DCEBE5;
    border-left:4px solid {CORAL}; border-radius:10px; padding:8px 16px; box-shadow:0 1px 6px rgba(39,39,54,.10); }}
[data-testid="stSpinner"] * {{ color:{DARK} !important; }}

/* dropdowns: pick from the list only - no text cursor, no typing */
div[data-baseweb="select"] input {{ caret-color:transparent !important; pointer-events:none !important; user-select:none; }}
div[data-baseweb="select"], div[data-baseweb="select"] * {{ cursor:pointer !important; }}

/* papers-by-year: the side list only exists after a bar is clicked; until then the chart takes the whole row */
.st-key-year_row [data-testid="stColumn"]:first-child, .st-key-year_row [data-testid="column"]:first-child {{ flex:1 1 auto !important; }}
.st-key-year_row [data-testid="stColumn"]:not(:has([class*="st-key-icard_list"])):last-child,
.st-key-year_row [data-testid="column"]:not(:has([class*="st-key-icard_list"])):last-child {{ display:none !important; }}

/* sort control: sized to its text, block font */
[data-testid="stButtonGroup"] button p {{ font:400 17px {BLOCK}; letter-spacing:.04em; }}
.st-key-results_panel [data-testid="stTabs"] {{ margin-top:-4px; }}
.lbl {{ font:400 18px {BLOCK}; letter-spacing:.04em; color:#6B7785; white-space:nowrap; }}
/* sort: a compact button that opens a small menu of choices, sized to its text */
.st-key-sort_pop, .st-key-sort_pop [data-testid="stPopover"] {{ width:fit-content !important; }}
.st-key-sort_pop button p, [data-testid="stPopoverBody"] button p {{ font:400 17px {BLOCK}; letter-spacing:.04em; }}
[data-testid="stPopoverBody"] {{ width:auto !important; min-width:0 !important; max-width:240px; padding:.6rem; }}
[data-testid="stPopoverBody"] button {{ width:100%; justify-content:flex-start; }}

/* metrics chips (white) + "searches run" chip whose list overlays the page */
.metrics {{ position:relative; z-index:60; display:flex; flex-wrap:wrap; align-items:flex-start; gap:10px; margin:2px 0; }}
.m {{ background:#fff; border-radius:10px; padding:6px 16px; display:flex; align-items:baseline; gap:10px;
    border-left:4px solid {CORAL}; box-shadow:0 1px 5px rgba(39,39,54,.07); }}
.m.g {{ border-left-color:{GREEN}; }} .m.d {{ border-left-color:{DARK}; }}
.m b {{ font:400 28px {BLOCK}; letter-spacing:.04em; color:{DARK}; }}
.m span {{ font:400 15px {BLOCK}; letter-spacing:.04em; color:#6B7785; }}
details.srch {{ display:block; position:relative; }}
details.srch summary {{ display:flex; align-items:baseline; gap:10px; cursor:pointer; list-style:none; }}
details.srch summary::-webkit-details-marker {{ display:none; }}
.arr {{ align-self:center; width:8px; height:8px; margin-left:4px; border-right:2.5px solid {CORAL}; border-bottom:2.5px solid {CORAL};
    transform:rotate(-45deg); transition:transform .15s; }}
details[open] .arr {{ transform:rotate(45deg); }}
.slist, .steps, .modes {{ list-style:none; margin:0; padding:0; }}
.slist {{ position:absolute; top:calc(100% + 8px); left:0; z-index:100; min-width:380px; max-width:600px; padding:10px 18px;
    background:#fff; border-radius:10px; box-shadow:0 8px 28px rgba(39,39,54,.20); }}
.slist li {{ position:relative; padding-left:20px; margin:6px 0; font-size:13px; color:#3B3B4F; }}
.slist li:before, .modes li:before {{ content:''; position:absolute; left:0; top:.5em; width:8px; height:8px; border-radius:50%; background:{CORAL}; }}
.steps {{ counter-reset:s; }}
.steps li {{ counter-increment:s; position:relative; padding:2px 0 2px 46px; margin:14px 0; color:#3B3B4F; }}
.steps li:before {{ content:counter(s); position:absolute; left:0; top:0; width:32px; height:32px; border-radius:50%; background:{CORAL};
    color:{DARK}; font:400 20px {BLOCK}; display:flex; align-items:center; justify-content:center; }}
.steps b {{ display:block; font:400 21px {BLOCK}; letter-spacing:.04em; color:{DARK}; }}

/* sidebar */
.nav-i {{ display:flex; align-items:center; gap:14px; padding:7px 4px; }}
.nav-i span {{ font:400 20px {BLOCK}; letter-spacing:.04em; color:#fff; }}
.nav-i em {{ font:400 11px {BLOCK}; font-style:normal; background:{CORAL}; color:{DARK}; padding:1px 7px; border-radius:10px; }}
.nav-i.off span {{ color:#9A9AAE; }}
.side-h {{ font:400 18px {BLOCK}; letter-spacing:.04em; color:{CORAL}; margin:14px 0 8px; border-top:1px solid #3A3A4D; padding-top:14px; }}
.modes li {{ position:relative; padding-left:18px; margin:5px 0; font:400 16px/1.15 {BLOCK}; letter-spacing:.04em; color:#9A9AAE; }}
.modes li.on {{ color:#fff; }} .modes li.on b {{ color:{CORAL}; }} .modes li b {{ font-weight:400; margin-right:6px; }}
section[data-testid="stSidebar"] :is(p, label, li, span, div[data-baseweb="select"] div):not([data-testid="stIconMaterial"]):not([class*="material"]) {{
    font-family:{BLOCK} !important; letter-spacing:.04em; }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stToggle"] p {{ font-size:19px; color:#fff; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ font-size:16px; line-height:1.15; color:#A9A9BD; }}
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ font-size:17px; color:{CORAL}; }}
section[data-testid="stSidebar"] div[role="slider"] {{ background-color:{CORAL} !important; box-shadow:none !important; }}
div[data-baseweb="popover"] li {{ font:400 18px {BLOCK}; letter-spacing:.04em; }}

/* never touch Streamlit's own icon font (otherwise icons show as words like "keyboard_double_arrow_right") */
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {{
    font-family:'Material Symbols Rounded', 'Material Icons' !important; letter-spacing:normal !important; }}
</style>
"""


def card_head(title: str, tone: str = "") -> None:
    st.markdown(f'<div class="card-h {tone}">{html.escape(title)}</div>', unsafe_allow_html=True)


def metrics_html(papers, sbert, query: str, queries: list[str]) -> str:
    sources = sorted({p.source.replace("_", " ") for p in papers})
    top = float(np.max(sbert.scores(query)))
    chips = [("", str(len(papers)), "PAPERS RANKED"), ("g", ", ".join(sources) or "-", "SOURCES"),
             ("", f"{top:.2f}", "BEST MATCH")]
    out = "".join(f'<div class="m {c}"><b>{html.escape(v)}</b><span>{lab}</span></div>' for c, v, lab in chips)
    if queries:  # click the chip to show the keyword searches PARSE ran (opens as an overlay)
        items = "".join(f"<li>{html.escape(q)}</li>" for q in queries)
        out += (f'<details class="m d srch"><summary><b>{len(queries)}</b><span>SEARCHES RUN</span>'
                f'<i class="arr"></i></summary><ul class="slist">{items}</ul></details>')
    else:
        out += '<div class="m d"><b>LOCAL</b><span>CORPUS - NO WEB SEARCHES</span></div>'
    return f'<div class="metrics">{out}</div>'


# ---------------------------------------------------------------- data / models
@st.cache_resource(show_spinner=False)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.SBERT_MODEL)


@st.cache_resource
def _fetch_cache() -> dict:
    return {}


def _cached(key, fn):
    """Remember a fetch for an hour - but a fetch with warnings (e.g. a 429) only for 60 s, so a
    temporary Semantic Scholar failure is never replayed from the cache."""
    store = _fetch_cache()
    hit = store.get(key)
    if hit and time.time() < hit[0]:
        return hit[1]
    result = fn()
    store[key] = (time.time() + (60 if result[1] else 3600), result)
    return result


def live_candidates(text: str, per_source: int, max_queries: int):
    return _cached(("live", text, per_source, max_queries),
                   lambda: fetch_live(text, per_source, api_key=config.SEMANTIC_SCHOLAR_API_KEY,
                                      email=config.OPENALEX_EMAIL, max_queries=max_queries))


def expansion_candidates(titles: tuple, per_source: int):
    return _cached(("expand", titles, per_source),
                   lambda: fetch_queries(list(titles), per_source, api_key=config.SEMANTIC_SCHOLAR_API_KEY,
                                         email=config.OPENALEX_EMAIL))


@st.cache_resource
def _rank_cache() -> dict:
    return {}


def build_retrievers(papers, model):
    """Embedding ~200 abstracts takes a few seconds, so do it once per paper pool, not on every click."""
    store, key = _rank_cache(), tuple(p.id for p in papers)
    if key not in store:
        if len(store) > 6:
            store.clear()
        store[key] = (SbertRetriever(papers, model=model), TfidfRetriever(papers))
    return store[key]


@st.cache_resource(show_spinner=False)
def local_index(path: str):
    papers = load_corpus(path)
    return papers, SbertRetriever(papers, model=get_model(), cache_dir="data/cache"), TfidfRetriever(papers)


def _set_sort(option: str) -> None:
    st.session_state["sort_menu"] = option


def sort_hits(hits, sort: str):
    """Order a result list by year (papers without a year always go last); relevance order is the default."""
    if sort.endswith("Newest first"):
        return sorted(hits, key=lambda h: (h.paper.year is None, -(h.paper.year or 0)))
    if sort.endswith("Oldest first"):
        return sorted(hits, key=lambda h: (h.paper.year is None, h.paper.year or 0))
    return hits


def paper_card(key: str, rank, title, url, meta: list[str], score, abstract: str) -> None:
    with st.container(key=key):
        t = html.escape(title)
        t = f'<a href="{html.escape(url)}" target="_blank">{t}</a>' if url else t
        st.markdown(f'<span class="rk">{rank}</span><span class="ptitle">{t}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">{score:.3f}</span><span style="color:#7B8794;font-size:13px">'
                    f'{html.escape(" · ".join(m for m in meta if m))}</span>', unsafe_allow_html=True)
        with st.expander("Abstract"):
            st.write(abstract)


def show(retriever, query: str, k: int, sort: str):
    st.markdown(f'<div class="sec-h">{retriever.name}</div>', unsafe_allow_html=True)
    for h in sort_hits(retriever.search(query, k), sort):
        p = h.paper
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        paper_card(f"rcard_{retriever.name}_{h.rank}", h.rank, p.title, p.url,
                   [authors, str(p.year or ""), p.source], h.score, p.abstract)


def results_table(sbert, tfidf, query: str) -> pd.DataFrame:
    """Every paper in the pool, best SBERT match first (column '#' = that relevance rank)."""
    s, t = sbert.scores(query), tfidf.scores(query)
    rows = []
    for rank, i in enumerate(np.argsort(-s), start=1):
        p = sbert.papers[i]
        rows.append({
            "#": rank,
            "SBERT score": round(float(s[i]), 3),
            "TF-IDF score": round(float(t[i]), 3),
            "Title": p.title,
            "Authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
            "Year": str(p.year or ""),
            "Source": p.source,
            "Link": p.url,
            "Abstract": p.abstract,
        })
    return pd.DataFrame(rows)


def sort_table(df: pd.DataFrame, sort: str) -> pd.DataFrame:
    if sort.endswith("Relevance"):
        return df
    year = pd.to_numeric(df["Year"], errors="coerce")
    order = year.sort_values(ascending=sort.endswith("Oldest first"), na_position="last", kind="stable").index
    return df.loc[order]


def score_chart(df: pd.DataFrame) -> alt.Chart:
    top = df.head(30)
    long = pd.concat([pd.DataFrame({"rank": top["#"], "score": top["SBERT score"], "method": "SBERT"}),
                      pd.DataFrame({"rank": top["#"], "score": top["TF-IDF score"], "method": "TF-IDF"})])
    base = alt.Chart(long).encode(
        x=alt.X("rank:Q", title="Rank (1 = best SBERT match)"),
        y=alt.Y("score:Q", title="Similarity score"),
        color=alt.Color("method:N", title=None,
                        scale=alt.Scale(domain=["SBERT", "TF-IDF"], range=[DARK, CORAL]),
                        legend=alt.Legend(orient="top")))
    return (base.mark_line() + base.mark_point(filled=True, size=55)).properties(height=260, width="container")


def year_chart(df: pd.DataFrame):
    counts = Counter(y for y in df["Year"] if y)
    if not counts:
        return None
    data = pd.DataFrame(sorted(counts.items())[-15:], columns=["Year", "Papers"])
    pick = alt.selection_point(name="year_sel", fields=["Year"])
    return (alt.Chart(data).mark_bar(size=24, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, cursor="pointer")
            .encode(x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("Papers:Q"),
                    color=alt.condition(pick, alt.value(CORAL), alt.value(DARK)),
                    tooltip=["Year", "Papers"])
            .add_params(pick)
            .properties(height=340, width="container", background=GREEN)
            .configure_view(stroke=None)
            .configure_axis(labelColor=DARK, titleColor=DARK, domainColor=DARK, tickColor=DARK, gridColor="#8BD48B"))


# ---------------------------------------------------------------- sidebar
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div class="nav">'
        f'<div class="nav-i">{icon("home")}<span>HOME</span></div>'
        f'<div class="nav-i off">{icon("clock", "#9A9AAE")}<span>HISTORY</span><em>SOON</em></div>'
        f'<div class="nav-i off">{icon("user", "#9A9AAE")}<span>ACCOUNT</span><em>SOON</em></div>'
        '</div><div class="side-h">SETTINGS</div>', unsafe_allow_html=True)
    live = st.toggle("Live web search", value=True)
    st.caption("On: fetches fresh abstracts from arXiv, Semantic Scholar and OpenAlex. "
               "Off: searches the saved papers in the data folder.")
    compare = st.toggle("Compare with TF-IDF", value=False)
    st.caption("Shows the classic keyword ranking beside SBERT, so you can see what meaning-based search adds.")
    k = st.slider("Results to show", 3, 50, 10)
    st.caption("How many top papers appear as cards.")
    if live:
        depth = st.select_slider("Search depth", list(DEPTHS), value="Standard")
        st.markdown('<ul class="modes">' + "".join(
            f'<li class="{"on" if d == depth else ""}"><b>{d.upper()}</b>{txt}</li>'
            for d, txt in DEPTH_INFO.items()) + "</ul>", unsafe_allow_html=True)
        per_source = st.slider("Papers per source", 20, 100, config.CANDIDATES_PER_SOURCE, step=10)
        st.caption("Candidates pulled from each source before ranking. More means a wider net but a slower search.")
        if not config.SEMANTIC_SCHOLAR_API_KEY:
            st.caption("No Semantic Scholar key found in .env.")
    else:
        files = sorted(str(p) for p in Path("data").glob("*.json") if "eval" not in p.name)
        corpus_path = st.selectbox("Corpus file", files or ["data/sample_papers.json"])
        st.caption("The saved paper collection to search.")
    if not st.session_state.get("_sidebar_opened"):  # open by default; the arrow still toggles it afterwards
        st.session_state["_sidebar_opened"] = True
        components.html("""<script>
        const d = window.parent.document; let tries = 0;
        const timer = setInterval(() => {
          const sb = d.querySelector('section[data-testid="stSidebar"]');
          const btn = d.querySelector('[data-testid="stExpandSidebarButton"] button')
                   || d.querySelector('[data-testid="stExpandSidebarButton"]')
                   || d.querySelector('[data-testid="collapsedControl"] button')
                   || d.querySelector('[data-testid="collapsedControl"]');
          if (sb && sb.getAttribute('aria-expanded') === 'true') clearInterval(timer);
          else if (sb && btn) { btn.click(); clearInterval(timer); }
          if (++tries > 25) clearInterval(timer);
        }, 150);
        </script>""", height=0)

# ---------------------------------------------------------------- header + search
st.markdown(
    '<div class="topbar"><div class="tl"><div class="logo"><span>P</span></div><div><div class="brand">PARSE</div>'
    '<div class="tag">Paper Abstract Retrieval using Semantic Embeddings</div></div></div>'
    f'<div class="acct"><div class="avatar">{icon("user", DARK, 20)}</div>'
    '<div><div class="who">GUEST</div><div class="soon">SIGN-IN COMING SOON</div></div></div></div>',
    unsafe_allow_html=True)

with st.container(key="search_panel"):
    card_head("FIND PAPERS FOR YOUR IDEA", "coral")
    with st.form("search", border=False):
        text = st.text_area("Your research idea or topic", height=120, label_visibility="collapsed",
                            placeholder="Describe your research idea or type a topic")
        submitted = st.form_submit_button("Find papers", type="primary")

if submitted and text.strip():
    st.session_state["query"] = text.strip()
query = st.session_state.get("query", "")
status = st.container()  # loading messages appear here and push the content below them down

# ---------------------------------------------------------------- results
if not query:
    with st.container(key="how_panel"):
        st.markdown(
            '<div class="sec-h">HOW PARSE WORKS</div><ol class="steps">'
            '<li><b>Describe your idea</b>Paste a paragraph or type a topic in the box above.</li>'
            '<li><b>PARSE gathers candidates</b>Keyword searches pull paper abstracts from arXiv, '
            'Semantic Scholar and OpenAlex.</li>'
            '<li><b>SBERT ranks by meaning</b>Every abstract is scored against your idea, so relevant papers '
            'surface even when they share no keywords.</li></ol>', unsafe_allow_html=True)
    st.stop()

try:
    with status, st.spinner("Loading the SBERT model (first run downloads ~90 MB)..."):
        model = get_model()
except Exception as exc:
    st.error(f"Could not load the SBERT model: {exc}\n\nCheck your internet connection "
             "(the model downloads once on first run), then reload.")
    st.stop()

queries_used: list[str] = []
if live:
    max_queries, expand = DEPTHS[depth]
    with status, st.spinner("Searching arXiv, Semantic Scholar and OpenAlex..."):
        papers, warnings, queries_used = live_candidates(query, per_source, max_queries)
    warnings = list(warnings)
    queries_used = list(queries_used)
    if papers and expand:
        with status, st.spinner("Deep search: finding papers related to your best matches..."):
            first_pass = build_retrievers(papers, model)[0]
            top_titles = tuple(h.paper.title for h in first_pass.search(query, 3))
            extra, extra_warnings = expansion_candidates(top_titles, per_source)
            papers = dedupe(papers + extra)
            warnings += [w for w in extra_warnings if w not in warnings]
            queries_used += [f"(from a top match) {t}" for t in top_titles]
    for w in warnings:
        st.warning(w)
    if not papers:
        st.error("No papers could be fetched. Check your internet connection or try different wording.")
        st.stop()
    with status, st.spinner(f"Ranking {len(papers)} abstracts by meaning..."):
        sbert, tfidf = build_retrievers(papers, model)
else:
    with status, st.spinner("Indexing local corpus..."):
        papers, sbert, tfidf = local_index(corpus_path)

st.markdown(metrics_html(papers, sbert, query, queries_used), unsafe_allow_html=True)
df = results_table(sbert, tfidf, query)

with st.container(key="results_panel"):
    h1, h2, h3 = st.columns([1.1, 3, 1.3], vertical_alignment="center")
    h1.markdown('<div class="sec-h">RESULTS</div>', unsafe_allow_html=True)
    sort = st.session_state.get("sort_menu", SORTS[0])
    with h2.container(key="sort_pop"):
        with st.popover(f"Sort: {sort}"):  # click the button -> small menu of the three choices
            for option in SORTS:
                st.button(option, key=f"sort_opt_{option}", on_click=_set_sort, args=(option,),
                          type="primary" if option == sort else "secondary")
    df_view = sort_table(df, sort)
    h3.download_button("Download CSV", df_view.to_csv(index=False).encode("utf-8-sig"),
                       file_name="parse_results.csv", mime="text/csv")

    tab_top, tab_all, tab_info = st.tabs(["TOP RESULTS", f"ALL {len(papers)} PAPERS", "INSIGHTS"])

    with tab_top:
        st.caption(f"The {min(k, len(papers))} best matches" +
                   ("" if sort.endswith("Relevance") else ", re-ordered by year") +
                   ". The coral number is the relevance rank.")
        if compare:
            r1, r2 = st.columns(2)
            with r1:
                show(sbert, query, k, sort)
            with r2:
                show(tfidf, query, k, sort)
        else:
            show(sbert, query, k, sort)

    with tab_all:
        st.caption("Every paper that was fetched. The '#' column is the SBERT relevance rank; use the sort menu "
                   "above, or click any column header.")
        st.dataframe(
            df_view.drop(columns=["Abstract"]),
            hide_index=True,
            height=600,
            column_config={
                "Title": st.column_config.TextColumn(width="large"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open ↗"),
            },
        )

    with tab_info:
        with st.container(key="icard_score"):
            card_head("SIMILARITY BY RANK")
            st.altair_chart(score_chart(df), theme=None)
        with st.container(key="year_row"):
            c_year, c_list = st.columns([1.5, 1])
            year = ""
            with c_year:
                with st.container(key="icard_year"):
                    card_head("PAPERS BY YEAR")
                    chart = year_chart(df)
                    if chart is None:
                        st.caption("No publication years available for these papers.")
                    else:
                        event = st.altair_chart(chart, theme=None, on_select="rerun", key="year_chart")
                        picked = (event.get("selection", {}) if event else {}).get("year_sel") or []
                        year = str(picked[0].get("Year", "")) if picked else ""
                        st.caption("Click a bar to list that year's papers.")
            with c_list:
                if year:
                    with st.container(key="icard_list"):
                        card_head(f"PAPERS FROM {year}")
                        sub = df[df["Year"] == year]
                        st.caption(f"{len(sub)} paper(s), best match first.")
                        with st.container(height=430):
                            for i, r in enumerate(sub.to_dict("records")):
                                paper_card(f"rcard_year_{i}", r["#"], r["Title"], r["Link"],
                                           [r["Authors"], r["Source"]], r["SBERT score"], r["Abstract"])