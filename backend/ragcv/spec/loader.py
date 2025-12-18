from pathlib import Path
from typing import Any, List, Dict
from ..agents import Agent               
from .models import GraphConfig, RetrievalConfig
from ..retrieval.retrieval import AdaptiveRetrieverConfig

def load_graph_config(
    path: str | Path,
    tools: Any,
    logger: Any
) -> List[Dict[str, Any]]:
    """
    Load graph configuration and construct Agent objects.
    """
    # Load and validate the config using the class method
    graph_spec = GraphConfig.from_yaml(path)

    # Construct Agent objects
    agents = []
    for spec in graph_spec.agents:
        node = Agent(
            name=spec.name,
            prompt_path=spec.prompt_path,
            logger=logger,
            tools=tools,
            temperature=spec.temperature,
        )

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