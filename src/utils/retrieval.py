from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

@dataclass
class AdaptiveRetrieverConfig:
    base_k: int = 10
    mmr_k: int = 5
    mmr_lambda: float = 0.6
    score_threshold: float = 0.15
    min_high_score: int = 5
    dedupe_threshold: float = 0.88

class AdaptiveRetriever:
    """Retrieval helper that applies metadata filtering, dedupe, and MMR."""

    def __init__(
        self,
        vectorstore,
        embeddings: OpenAIEmbeddings,
        config: Optional[AdaptiveRetrieverConfig] = None,
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.config = config or AdaptiveRetrieverConfig()

    def invoke(self, query: str) -> List[Document]:
        ranked = self.adaptive_retrieval(
            query=query,
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            config=self.config,
        )
        documents: List[Document] = []
        for doc, score in ranked:
            metadata = dict(doc.metadata)
            metadata["retrieval_score"] = float(score)
            documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )
        return documents

    def adaptive_retrieval(
        self,
        query: str,
        vectorstore,
        embeddings: OpenAIEmbeddings,
        config: AdaptiveRetrieverConfig,
    ) -> List[Tuple[Document, float]]:
        candidates = _run_vectorstore_search(
            query=query,
            vectorstore=vectorstore,
            k=config.base_k,
        )
        candidates = [
            (doc, float(score))
            for doc, score in candidates
            if score is not None and float(score) >= config.score_threshold
        ]

        if len(candidates) < config.min_high_score:
            expanded = _run_vectorstore_search(
                query=query,
                vectorstore=vectorstore,
                k=config.base_k * 2,
            )
            merged = {id(doc): (doc, float(score)) for doc, score in candidates}
            for doc, score in expanded:
                merged[id(doc)] = (doc, float(score))
            candidates = list(merged.values())
            candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            return []

        deduped = deduplicate_chunks(
            candidates,
            embeddings=embeddings,
            sim_threshold=config.dedupe_threshold,
        )
        if not deduped:
            return []

        selected = mmr_selection(
            deduped,
            embeddings=embeddings,
            mmr_k=config.mmr_k,
            mmr_lambda=config.mmr_lambda,
        )
        return selected

def deduplicate_chunks(
    candidates: List[Tuple[Document, float]],
    embeddings: OpenAIEmbeddings,
    sim_threshold: float,
) -> List[Tuple[Document, float]]:
    if len(candidates) <= 1:
        return candidates

    texts = [doc.page_content for doc, _ in candidates]
    vectors = embed_batch(texts, embeddings)
    kept: List[Tuple[Document, float]] = []
    skipped = set()

    for i, (doc_i, score_i) in enumerate(candidates):
        if i in skipped:
            continue
        representative = (doc_i, score_i)
        for j in range(i + 1, len(candidates)):
            if j in skipped:
                continue
            similarity = cos_sim(vectors[i], vectors[j])
            if similarity >= sim_threshold:
                if candidates[j][1] > representative[1]:
                    representative = candidates[j]
                skipped.add(j)
        kept.append(representative)

    return kept

def mmr_selection(
    candidates: List[Tuple[Document, float]],
    embeddings: OpenAIEmbeddings,
    mmr_k: int,
    mmr_lambda: float,
) -> List[Tuple[Document, float]]:
    k = min(mmr_k, len(candidates))
    if k == 0:
        return []

    vectors = np.array(embed_batch([doc.page_content for doc, _ in candidates], embeddings))
    relevance = np.array([score for _, score in candidates])

    selected_indices: List[int] = [int(np.argmax(relevance))]

    while len(selected_indices) < k:
        mmr_scores = []
        for idx in range(len(candidates)):
            if idx in selected_indices:
                mmr_scores.append(-math.inf)
                continue
            diversity = max(
                cos_sim(vectors[idx], vectors[sel_idx])
                for sel_idx in selected_indices
            )
            score = mmr_lambda * relevance[idx] - (1 - mmr_lambda) * diversity
            mmr_scores.append(score)
        next_idx = int(np.argmax(mmr_scores))
        if mmr_scores[next_idx] == -math.inf:
            break
        selected_indices.append(next_idx)

    return [candidates[i] for i in selected_indices]

def embed_batch(texts: Sequence[str], embeddings: OpenAIEmbeddings) -> List[List[float]]:
    if not texts:
        return []
    return embeddings.embed_documents(list(texts))

def cos_sim(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _run_vectorstore_search(vectorstore, query: str, k: int):
    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    raw = vectorstore.similarity_search_with_score(query, k=k)
    normalized: List[Tuple[Document, float]] = []
    for doc, distance in raw:
        if distance is None:
            score = 0.0
        else:
            score = 1 / (1 + float(distance))
        normalized.append((doc, score))
    return normalized