from __future__ import annotations

from typing import List, Optional, Any

import numpy as np
from langchain_core.documents import Document

from .retrieval import AdaptiveRetriever
from ..logger import JSONLLogger

class QueryEnricher():
    def __init__(
            self,
            retriever: AdaptiveRetriever, 
            logger: JSONLLogger
            ):
        self.retriever = retriever
        self.logger = logger
        self.retrieved_docs: Optional[List[Any]] = None

    def enrich_with_context(self, query: str):
        """ Takes input query and enriches it with retrieved context"""
        docs = self.retriever.invoke(query)
        context = self.process_docs(docs)
        self.log_invocation()
        return  f"<context>\n{context}\n</context>\n\n" + f"<query>\n{query}\n</query>" 
    
    
    def process_docs(self, docs: list) -> str:
        if self.logger is not None:
            self.retrieved_docs = [
                {
                    **doc.metadata,
                    "text": doc.page_content,
                }   
                for doc in docs
            ]

        return format_docs(docs)
    
    def log_invocation(self):
        diagnostics = {}
        if self.retrieved_docs is not None:
            retriever_config = {}
        if hasattr(self.retriever, "search_type"):
            retriever_config["search_type"] = self.retriever.search_type
        if hasattr(self.retriever, "search_kwargs"):
            retriever_config["search_kwargs"] = self.retriever.search_kwargs
        if retriever_config:
            diagnostics["retriever_config"] = retriever_config

        diagnostics["retrieved_documents"] = self.retrieved_docs
        self.retrieved_docs = None
        self.logger.log(
            {
                "retrieval":
                {
                    "diagnostics" : diagnostics
                }
            }
        )

def format_docs(
    docs: List[Document],
    include_header: bool = True,
) -> str:
    if not docs:
        return "Context: [no supporting documents retrieved]"

    lines: List[str] = []
    if include_header:
        lines.append("Context (most relevant first):")

    for doc in docs:
        md = doc.metadata
        score_str = f"{md['retrieval_score']:.3f}"
        tokens = f"{md['chunk_length_tokens']:.3f}"

        lines.append(
            f"[score={score_str} tokens={tokens}] {doc.page_content}"
        )

    return "\n\n".join(lines)
