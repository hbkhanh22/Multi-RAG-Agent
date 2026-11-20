"""
Minimal language tagging for cross-lingual reranking (Vietnamese vs everything else).

The cross-encoder scores a Vietnamese question against an English passage very low
(~0.02-0.05 even for the exact answer), while the same passage scores ~0.6-0.99
against an English rendering of the question. So each candidate is reranked with
the query variant written in the candidate's own language.
"""

from typing import List, Optional

# Letters that occur in Vietnamese but (practically) never in English text
_VI_CHARS = set(
    "ăâđêôơư"
    "àáạảãầấậẩẫằắặẳẵ"
    "èéẹẻẽềếệểễ"
    "ìíịỉĩ"
    "òóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữ"
    "ỳýỵỷỹ"
)


def detect_lang(text: str, sample: int = 2000) -> str:
    """'vi' if the text contains a meaningful share of Vietnamese letters, else 'other'."""
    letters = [ch for ch in (text or "")[:sample].lower() if ch.isalpha()]
    if not letters:
        return "other"
    vi = sum(1 for ch in letters if ch in _VI_CHARS)
    return "vi" if vi / len(letters) >= 0.03 else "other"


def pick_query(variants: List[str], doc_text: str, doc_lang: Optional[str] = None) -> str:
    """The first query variant (original first) in the document's language, else the original."""
    if not variants:
        return ""
    lang = doc_lang or detect_lang(doc_text)
    for q in variants:
        if detect_lang(q) == lang:
            return q
    return variants[0]
