from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document


def _coerce_token_estimate(value, text: str) -> int:
    if value is None:
        return approximate_token_count(text)
    try:
        return int(value)
    except (TypeError, ValueError):
        return approximate_token_count(text)


def approximate_token_count(text: str) -> int:
    """Rough heuristic: ~4 characters per token."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def extractive_summary(text: str, max_tokens: int = 120) -> str:
    """Deterministic summary using lead sentences until the token budget is hit."""
    normalized = text.strip()
    if not normalized:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    summary_parts: List[str] = []
    tokens_used = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = approximate_token_count(sentence)
        if sentence_tokens == 0:
            continue
        if tokens_used + sentence_tokens > max_tokens:
            break
        summary_parts.append(sentence)
        tokens_used += sentence_tokens
        if tokens_used >= max_tokens:
            break

    if not summary_parts:
        truncated = normalized[: max_tokens * 4].rstrip()
        return truncated

    return " ".join(summary_parts)


@dataclass
class ContextSnippet:
    text: str
    score: float
    mode: str  # "raw" or "summary"
    source: str
    doc_type: str
    tokens: int
    metadata: dict


@dataclass
class ContextAssemblyConfig:
    model_context_tokens: int = 8192
    prompt_reserved_tokens: int = 2000
    keep_raw_top_n: int = 3
    summarize_low_n: int = 7
    summary_max_tokens: int = 120

    @property
    def max_context_tokens(self) -> int:
        usable = self.model_context_tokens - self.prompt_reserved_tokens
        return max(512, usable)


def assemble_with_token_budget(
    docs: Sequence[Document],
    config: Optional[ContextAssemblyConfig] = None,
) -> List[ContextSnippet]:
    if not docs:
        return []

    config = config or ContextAssemblyConfig()
    entries = []
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        score = float(metadata.get("retrieval_score", 0.0))
        token_estimate = _coerce_token_estimate(metadata.get("chunk_length_tokens"), doc.page_content)
        entries.append((doc, score, token_estimate, metadata))

    entries.sort(key=lambda x: x[1], reverse=True)

    assembled: List[ContextSnippet] = []
    tokens_used = 0
    summaries_added = 0
    max_tokens = config.max_context_tokens

    for idx, (doc, score, token_estimate, metadata) in enumerate(entries):
        raw_mode = idx < config.keep_raw_top_n
        if raw_mode:
            snippet_text = doc.page_content.strip()
            snippet_tokens = token_estimate
            snippet_mode = "raw"
        else:
            if summaries_added >= config.summarize_low_n:
                break
            summary = extractive_summary(doc.page_content, max_tokens=config.summary_max_tokens)
            summary = summary.strip()
            if not summary:
                continue
            snippet_text = summary
            snippet_tokens = approximate_token_count(summary)
            snippet_mode = "summary"
            summaries_added += 1

        if snippet_tokens == 0:
            continue
        if tokens_used + snippet_tokens > max_tokens:
            break

        snippet = ContextSnippet(
            text=snippet_text,
            score=score,
            mode=snippet_mode,
            source=metadata.get("source", "unknown"),
            doc_type=metadata.get("doc_type", "unknown"),
            tokens=snippet_tokens,
            metadata=metadata,
        )
        assembled.append(snippet)
        tokens_used += snippet_tokens

    return assembled


def format_weighted_context(
    snippets: Sequence[ContextSnippet],
    *,
    include_header: bool = True,
) -> str:
    if not snippets:
        return "Context: [no supporting documents retrieved]"

    lines: List[str] = []
    if include_header:
        lines.append("Context (most relevant first):")

    for snippet in snippets:
        score_str = f"{snippet.score:.3f}"
        lines.append(
            f"[score={score_str} type={snippet.mode} doc_type={snippet.doc_type}] {snippet.text}"
        )

    return "\n\n".join(lines)

