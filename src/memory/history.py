"""Helpers that turn stored messages into prompt material."""
import re
import unicodedata
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import settings

_SOURCE_TAG = re.compile(r"\s*\[Source \d+\]")
PER_MESSAGE_CHARS = 500


def _clean(role: str, text: str) -> str:
    # Old [Source N] numbers must not leak into a new turn (they would collide with its own numbering)
    if role == "assistant":
        text = _SOURCE_TAG.sub("", text)
    text = text.strip()
    return text if len(text) <= PER_MESSAGE_CHARS else text[:PER_MESSAGE_CHARS].rstrip() + "…"


def _window(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Most recent messages that fit the turn and character budgets (oldest first)."""
    picked, used = [], 0
    for msg in reversed(history[-settings.memory_max_turns * 2:]):
        content = _clean(msg["role"], msg["content"])
        if picked and used + len(content) > settings.memory_max_chars:
            break
        picked.append({"role": msg["role"], "content": content})
        used += len(content)
    return list(reversed(picked))


def format_history(history: List[Dict[str, str]]) -> str:
    """Plain-text transcript for prompts ('' when there is no history)."""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in _window(history)
    )


def history_to_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in _window(history)
    ]


# Follow-up detection: only rewrite when the question probably depends on earlier turns.
_FOLLOW_UP = re.compile(
    r"\b(no|do|day|kia|nay|ay|vay|the con|con|tiep|them|ky hon|chi tiet hon|ro hon|vi du|"
    r"truoc do|o tren|vua roi|thu (?:nhat|hai|ba|tu|nam)|cai nay|cai do|cai kia|"
    r"it|its|that|this|those|they|them|more|also|again|another|former|latter|what about|why)\b"
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def needs_condense(query: str, history: List[Dict[str, str]]) -> bool:
    if not history:
        return False
    norm = _norm(query)
    return len(norm.split()) <= 8 or bool(_FOLLOW_UP.search(norm))


def condense_prompt(query: str, history: List[Dict[str, str]]) -> str:
    return (
        "Conversation so far:\n"
        f"{format_history(history)}\n\n"
        f"Latest user message: {query}\n\n"
        "Rewrite the latest user message as ONE standalone question or request that contains all context "
        "needed to understand it without the conversation (resolve pronouns and references). "
        "Keep the user's language. If it is already standalone, return it unchanged. "
        "Output only the rewritten message, nothing else. Never answer it."
    )
