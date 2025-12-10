from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

@dataclass
class AdaptiveRetrieverConfig:
    base_k: int = 10
    mmr_k: int = 5
    mmr_lambda: float = 0.6
    score_threshold: float = 0.15
    min_high_score: int = 5
    dedupe_threshold: float = 0.88
    # Hybrid retrieval settings
    use_hybrid: bool = True
    bm25_weight: float = 0.5  # 0.5 = equal weighting
    embedding_weight: float = 0.5

class AdaptiveRetriever:
    """Retrieval helper with hybrid BM25+embeddings, dedupe, and MMR."""

    def __init__(
        self,
        vectorstore,
        embeddings: OpenAIEmbeddings,
        config: Optional[AdaptiveRetrieverConfig] = None,
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.config = config or AdaptiveRetrieverConfig()
        
        # Initialize BM25 index if hybrid retrieval is enabled
        self.bm25 = None
        self.corpus_docs = []
        if self.config.use_hybrid:
            self._initialize_bm25()

    def _initialize_bm25(self):
        """Build BM25 index from vectorstore documents."""
        try:
            # Extract all documents from vectorstore
            if hasattr(self.vectorstore, 'docstore'):
                self.corpus_docs = list(self.vectorstore.docstore._dict.values())
            elif hasattr(self.vectorstore, 'similarity_search'):
                # Fallback: get a large sample
                self.corpus_docs = self.vectorstore.similarity_search("", k=1000)
            
            if self.corpus_docs:
                # Tokenize corpus for BM25
                tokenized_corpus = [
                    doc.page_content.lower().split() 
                    for doc in self.corpus_docs
                ]
                self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"Warning: BM25 initialization failed: {e}. Falling back to embedding-only retrieval.")
            self.bm25 = None

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
        
        if config.use_hybrid and self.bm25 is not None:
            candidates = self._hybrid_search(query, config)
        else:
            candidates = _run_vectorstore_search(
                query=query,
                vectorstore=vectorstore,
                k=config.base_k,
            )
        
        # Filter by threshold
        candidates = [
            (doc, float(score))
            for doc, score in candidates
            if score is not None and float(score) >= config.score_threshold
        ]

        # Expand if needed
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

        # Deduplicate
        deduped = deduplicate_chunks(
            candidates,
            embeddings=embeddings,
            sim_threshold=config.dedupe_threshold,
        )
        if not deduped:
            return []

        # MMR selection
        selected = mmr_selection(
            deduped,
            embeddings=embeddings,
            mmr_k=config.mmr_k,
            mmr_lambda=config.mmr_lambda,
        )
        return selected

    def _hybrid_search(
        self, 
        query: str, 
        config: AdaptiveRetrieverConfig
    ) -> List[Tuple[Document, float]]:
        """Combine BM25 and embedding scores using reciprocal rank fusion."""
        
        # Get embedding-based results
        embedding_results = _run_vectorstore_search(
            query=query,
            vectorstore=self.vectorstore,
            k=config.base_k * 2,  # Get more for fusion
        )
        
        # Get BM25 results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_normalized = bm25_scores / max_bm25
        
        # Create BM25 results list
        bm25_results = [
            (self.corpus_docs[i], float(bm25_normalized[i]))
            for i in range(len(self.corpus_docs))
        ]
        
        # Sort both by score
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_results = bm25_results[:config.base_k * 2]
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_rrf = 60  # RRF constant
        
        # Add embedding ranks
        for rank, (doc, score) in enumerate(embedding_results, start=1):
            doc_id = id(doc)
            rrf_scores[doc_id] = {
                'doc': doc,
                'score': rrf_scores.get(doc_id, {}).get('score', 0) + 
                         config.embedding_weight / (k_rrf + rank)
            }
        
        # Add BM25 ranks
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            doc_id = id(doc)
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['score'] += config.bm25_weight / (k_rrf + rank)
            else:
                rrf_scores[doc_id] = {
                    'doc': doc,
                    'score': config.bm25_weight / (k_rrf + rank)
                }
        
        # Convert to list and sort
        fused_results = [
            (item['doc'], item['score']) 
            for item in rrf_scores.values()
        ]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results[:config.base_k]


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