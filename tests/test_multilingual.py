"""Regression: Vietnamese queries must retrieve English docs; unrelated ones must not pass the threshold."""
from config.settings import settings


def best_score(modules, query):
    docs = modules["vsm"].hybrid_search("tech", query, k=10)
    from src.retrieval.reranker import reranker
    ranked = reranker.rerank(query, docs, k=4)
    return ranked[0][1] if ranked else 0.0


def test_vietnamese_query_matches_english_docs(modules, tech_ready):
    assert best_score(modules, "Model Context Protocol (MCP) là gì?") >= settings.relevance_threshold


def test_unrelated_vietnamese_query_below_threshold(modules, tech_ready):
    assert best_score(modules, "Cách nấu phở bò ngon") < settings.relevance_threshold
