import numpy as np
import time

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .chunking_old import token_count
from ..utils.logger import JSONLLogger

@dataclass
class AdaptiveRetrieverConfig:
    rag_threshold: int = 10000
    base_k: int = 15
    rerank_top_k: int = 5     # Final number of documents to return
    rerank_threshold: int = -2
    max_total_chunks: int = 10 
    use_hybrid: bool = True
    bm25_weight: float = 0.5  # 0.5 = equal weighting
    embedding_weight: float = 0.5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class AdaptiveRetriever:
    """Retrieval helper with hybrid BM25+embeddings, dedupe, and MMR."""

    def __init__(
        self,
        vectorstore,
        documents: List,
        logger: JSONLLogger,
        config: Optional[AdaptiveRetrieverConfig] = None,
    ):
        self.vectorstore = vectorstore
        self.documents = documents
        self.logger = logger
        self.config = config or AdaptiveRetrieverConfig()
        self.reranker = CrossEncoder(self.config.reranker_model)
        
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

    def invoke(self, queries: List[str]) -> List[Document]:
        corpus_token_count = token_count("".join([doc.page_content for doc in self.documents]))
        timings = {}
        total_start = time.time()

        if corpus_token_count > self.config.rag_threshold:
            print(f"\n{'='*60}\nTotal Token Count in Corpus: {corpus_token_count}\nConfigurated RAG Token Threshold: {self.config.rag_threshold}\nRunning RAG")
            ranked, timings = self.adaptive_retrieval(
                queries=queries,
                config=self.config,
                track_timings=True,
            )        
            documents: List[Document] = []
            for doc, score in ranked:
                metadata = dict(doc.metadata)
                documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=metadata,
                    )
                )

            total_time = time.time() - total_start

            log_obj = {
                "vectorstore_search_per_query": timings.get("vectorstore_search_per_query", {}),
                "rerank_time": timings.get("rerank_time", None),
                "total_retrieval_time": total_time,
            }
            self.logger.log_event("retrieval_timings", log_obj)

        else:
            print(f"\n{'='*60}\nTotal Token Count in Corpus: {corpus_token_count}\nConfigurated RAG Token Threshold: {self.config.rag_threshold}\nNOT Running RAG")
            documents = self.documents

        
        char_count = len("".join([doc.page_content for doc in documents]))
        print(f"\t-Using {len(documents)} Documents\n\t-Total character count:{char_count}\n{'='*60}\n")

        return documents

    def adaptive_retrieval(
        self,
        queries: List[str],
        config: 'AdaptiveRetrieverConfig',
        track_timings: bool = False
    ) -> List[Tuple[Document, float]]:
        
        all_candidates = {}
        vectorstore_search_per_query = {}
        total_search_time = 0

        for query in queries:
            search_start = time.time()
            if config.use_hybrid and self.bm25 is not None:
                results = self._hybrid_search(query)
            else:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.config.base_k
                )
            search_time = time.time() - search_start
            vectorstore_search_per_query[query] = search_time
            total_search_time += search_time

            for doc, _ in results:
                doc_id = id(doc)
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = doc

        candidates = list(all_candidates.values())
        if not candidates:
            return ([], {"vectorstore_search_per_query": vectorstore_search_per_query})

        # De-duplicate by chunk_id
        seen_ids = set()
        unique_docs = []
        for doc in candidates:
            chunk_id = doc.metadata.get('chunk_id', id(doc))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_docs.append(doc)

        # Reranking timing
        rerank_start = time.time()
        ranked_docs = self._rerank_parallel(queries, unique_docs)
        rerank_time = time.time() - rerank_start

        timings = {
            "vectorstore_search_per_query": vectorstore_search_per_query,
            "total_vectorstore_search_time": total_search_time,
            "rerank_time": rerank_time
        }

        if track_timings:
            return ranked_docs[:self.config.rerank_top_k], timings
        
        return ranked_docs[:self.config.rerank_top_k]

    def _hybrid_search(self, query: str) -> List[Tuple[Document, float]]:
        """
        Combines BM25 and Vector search using Reciprocal Rank Fusion (RRF).
        This ensures we find exact keyword matches for specific requirements.
        """
        # A. Vector Search
        vector_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.config.base_k
        )
        
        # B. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.config.base_k]
        bm25_results = [(self.corpus_docs[i], bm25_scores[i]) for i in bm25_top_indices]

        # C. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60 # Standard RRF constant
        
        for rank, (doc, _) in enumerate(vector_results, 1):
            rrf_scores[id(doc)] = rrf_scores.get(id(doc), 0) + (self.config.embedding_weight / (k + rank))
            
        for rank, (doc, _) in enumerate(bm25_results, 1):
            rrf_scores[id(doc)] = rrf_scores.get(id(doc), 0) + (self.config.bm25_weight / (k + rank))

        # Re-map IDs to Documents
        id_to_doc = {id(doc): doc for doc, _ in (vector_results + bm25_results)}
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(id_to_doc[doc_id], score) for doc_id, score in sorted_ids[:self.config.base_k]]
    
    def _rerank_parallel(self, queries: List[str], documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score every document against every requirement.
        Uses Max-Pooling: A document's final score is its best match among all requirements.
        """
        if not documents or not queries:
            return []
        
        # Create N*M pairs
        pairs = [[q, doc.page_content] for doc in documents for q in queries]
 
        # Batch predict (Parallelized inference)
        scores = self.reranker.predict(pairs, batch_size=32)
        
        # Reshape and take the Max across the requirement dimension
        # Result: One 'Peak Relevance' score per document
        scores_matrix = scores.reshape(len(documents), len(queries))
        max_scores = np.max(scores_matrix, axis=1)

        final_candidates = []
        for i, doc in enumerate(documents):
            score = float(max_scores[i])
            
            if score >= self.config.rerank_threshold:
                doc.metadata["rerank_score"] = score
                final_candidates.append(doc)

        final_candidates.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return [(doc, doc.metadata["rerank_score"]) for doc in final_candidates]