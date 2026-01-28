from __future__ import annotations

from typing import List, Optional, Any

from langchain_core.documents import Document
from dataclasses import asdict

from .retrieval import AdaptiveRetriever
from ..utils.logger import JSONLLogger
from ..factories.agent_factory import SpecialisedAgentFactory

class QueryEnricher():
    def __init__(
            self,
            retriever: AdaptiveRetriever, 
            logger: JSONLLogger
            ):
        self.retriever = retriever
        self.logger = logger
        self.agent_factory = SpecialisedAgentFactory()

    def get_retrieved_artifacts(self, query: str):
        """ Takes input query, aligns with CV/CoverLetter semantics, and returns retrieved context"""
        # Convert requirement language into retrieval language
        self.aligned_queries = self.align_query(query)

        docs = self.retriever.invoke(self.aligned_queries)
        retrieved_docs = self.process_docs(docs)
        self.log_invocation(retrieved_docs)
        return retrieved_docs
    
    def align_query(self, query) -> List[str]:
        agent = self.agent_factory.create_agent(name="Semantic_Alignment_Agent", temperature=0)
        agent_ouptut, _ = agent.invoke({"summary": query})
        self.logger.log_event(
            event_name="semantic_alignment", 
            event_metadata={
                "summary": query, 
                "aligned_query": agent_ouptut.requirement
            }
        )

        return agent_ouptut.requirements

    def process_docs(self, docs: list) -> str:
        retrieved_docs = [
            {
                **doc.metadata,
                "text": doc.page_content,
            }   
            for doc in docs
        ]
        return retrieved_docs
    
    def log_invocation(self, retrieved_docs):
        diagnostics = {}

        diagnostics["retriever_config"] = asdict(self.retriever.config)
        diagnostics["aligned_queries"] = self.aligned_queries
        # Log retrieved docs (if any)
        diagnostics["retrieved_documents"] = retrieved_docs

        self.logger.log(
            {
                "retrieval":
                {
                    "diagnostics" : diagnostics
                }
            }
        )