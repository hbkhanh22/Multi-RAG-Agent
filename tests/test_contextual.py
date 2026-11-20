"""Contextual chunk headers, index versioning and cross-lingual expansion (no network)."""
import json

import pytest
from langchain_core.documents import Document


def test_chunk_header_and_prefix():
    from src.ingestion.contextualizer import add_chunk_headers, chunk_header
    meta = {"doc_title": "Laws of the Game", "section": "Law 11 - Offside", "page_summary": "Offside position."}
    assert chunk_header(meta) == "Laws of the Game › Law 11 - Offside\nOffside position."
    assert chunk_header({}) == ""
    chunks = add_chunk_headers([Document(page_content="body", metadata=dict(meta)),
                                Document(page_content="plain", metadata={})])
    assert chunks[0].page_content.startswith("Laws of the Game › Law 11") and chunks[0].page_content.endswith("body")
    assert chunks[1].page_content == "plain"


@pytest.fixture
def contextualizer(tmp_path, monkeypatch):
    import src.ingestion.contextualizer as cz
    monkeypatch.setattr(cz, "CACHE_FILE", tmp_path / "ctx.json")
    c = cz.PageContextualizer()
    calls = []

    def fake_ask(text, prev_section, title):
        calls.append((text, prev_section, title))
        if "Law 11" in text:
            return {"title": "Laws of the Game", "section": "Law 11 - Offside", "summary": f"about {text}"}
        return {"title": "", "section": "", "summary": f"about {text}"}  # no heading on this page

    monkeypatch.setattr(c, "_ask", fake_ask)
    return c, calls, cz


def _pages(*texts):
    return [Document(page_content=t, metadata={"page": i, "filename": "laws.pdf"}) for i, t in enumerate(texts)]


def test_contextualize_carries_section_and_title_forward(contextualizer):
    c, calls, _ = contextualizer
    pages = _pages("Law 11 offside position", "continued text", "")
    c.contextualize(pages, "data/raw/sport/laws.pdf", 10, 1.0)
    assert [p.metadata["section"] for p in pages] == ["Law 11 - Offside"] * 3   # inherited
    assert all(p.metadata["doc_title"] == "Laws of the Game" for p in pages)
    assert calls[1][1] == "Law 11 - Offside" and calls[1][2] == "Laws of the Game"  # prev section passed on
    assert len(calls) == 2  # the empty page is not sent to the LLM


def test_contextualize_uses_cache_and_invalidates_on_change(contextualizer):
    c, calls, cz = contextualizer
    c.contextualize(_pages("Law 11 a", "b"), "k", 10, 1.0)
    assert len(calls) == 2 and json.loads(cz.CACHE_FILE.read_text(encoding="utf-8"))
    c.contextualize(_pages("Law 11 a", "b"), "k", 10, 1.0)
    assert len(calls) == 2            # same file version: served from cache
    c.contextualize(_pages("Law 11 a", "b"), "k", 11, 1.0)
    assert len(calls) == 4            # file changed (size): asked again


def test_contextualize_survives_llm_errors(contextualizer, monkeypatch):
    c, _, _ = contextualizer
    monkeypatch.setattr(c, "_ask", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    pages = _pages("x")
    c.contextualize(pages, "k2", 1, 1.0)
    assert pages[0].metadata["doc_title"] == "laws" and pages[0].metadata["section"] == ""


def test_index_version_gets_its_own_store(monkeypatch):
    from config.settings import settings
    from src.retrieval.vector_store import embedding_slug
    monkeypatch.setattr(settings, "index_version", "")
    assert embedding_slug() == "bge-m3"
    monkeypatch.setattr(settings, "index_version", "v2")
    assert embedding_slug() == "bge-m3-v2"


def test_crosslingual_instruction_toggle(modules, monkeypatch):
    from config.settings import settings
    expander = modules["expander"]
    seen = []
    monkeypatch.setattr(expander, "_invoke_llm", lambda messages: seen.append(messages[0].content) or '["a", "b"]')
    monkeypatch.setattr(settings, "query_expansion_crosslingual", True)
    assert expander.multi_query_expansion("Luật việt vị là gì?") == ["Luật việt vị là gì?", "a", "b"]
    assert "translation of the query into English" in seen[-1]
    monkeypatch.setattr(settings, "query_expansion_crosslingual", False)
    expander.multi_query_expansion("q")
    assert "translation" not in seen[-1]


def test_detect_lang_and_pick_query():
    from src.retrieval.lang import detect_lang, pick_query
    assert detect_lang("Vòng tròn giữa sân có bán kính bao nhiêu mét?") == "vi"
    assert detect_lang("The centre circle has a radius of 9.15 m") == "other"
    assert detect_lang("") == "other"
    variants = ["Vòng tròn giữa sân có bán kính bao nhiêu?", "What is the radius of the centre circle?"]
    assert pick_query(variants, "A circle with a radius of 9.15 m is marked") == variants[1]
    assert pick_query(variants, "Vòng tròn trung tâm có bán kính 9,15 m") == variants[0]
    assert pick_query(["only english query"], "Tài liệu tiếng Việt có dấu đầy đủ") == "only english query"


def test_reranker_scores_each_doc_with_same_language_variant(modules, monkeypatch):
    from src.retrieval import reranker
    seen = []
    monkeypatch.setattr(reranker.model, "predict", lambda pairs, **kw: seen.extend(pairs) or [0.9] * len(pairs))
    docs = [Document(page_content="A circle with a radius of 9.15 m"),
            Document(page_content="Vòng tròn trung tâm có bán kính 9,15 m")]
    reranker.rerank("Bán kính vòng tròn giữa sân?", docs, k=2,
                    query_variants=["What is the radius of the centre circle?"])
    assert seen[0][0] == "What is the radius of the centre circle?"
    assert seen[1][0] == "Bán kính vòng tròn giữa sân?"
