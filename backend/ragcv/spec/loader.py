from pathlib import Path
from typing import Any, List, Dict
from langchain_openai import OpenAIEmbeddings

from .models import GraphConfig, RetrievalConfig
from ..retrieval.retrieval import AdaptiveRetrieverConfig
from ..factories.agent_factory import SpecialisedAgentFactory
from ..graph.node import AgentNodeWrapper


def load_graph_config(
    path: str | Path,
    tools: Any,
    logger: Any
) -> List[Dict[str, Any]]:
    """
    Load graph configuration and construct Agent objects.
    """
    agent_factory = SpecialisedAgentFactory()
    # Load and validate the config using the class method
    graph_spec = GraphConfig.from_yaml(path)

    # Construct Agent objects
    agents = []
    for spec in graph_spec.agents:
    
        try:
            agent = agent_factory.create_agent(
                name=spec.name,
                tools=tools,
                logger=logger
            )

            node = AgentNodeWrapper(
                    agent=agent, 
                    agent_name=spec.name,
                    logger=logger
                )
            
        except Exception as e:
            raise RuntimeError(f"[Error constructing agent {spec.name}, {e}]")

        entry = {
            "name": spec.name,
            "node": node,
            "rank": spec.rank,
        }
        if spec.conditional_links:
            entry["conditional_links"] = spec.conditional_links
        if spec.hard_links:
            entry["hard_links"] = spec.hard_links

        agents.append(entry)

    return agents

def load_retrieval_config(path: str | Path) -> AdaptiveRetrieverConfig:
    """
    Loads configuration for retrieval and constructs dataclass object.
    """
    # Load and validate the config using the class method
    cfg = RetrievalConfig.from_yaml(path)

    return AdaptiveRetrieverConfig(**cfg.model_dump())