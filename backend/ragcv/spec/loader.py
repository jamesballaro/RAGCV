import yaml
from ..agents import Agent               
from .models import GraphSpec

def load_from_yaml(path, tools, logger):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    graph_spec = GraphSpec(**cfg_dict)

    agents = []
    for spec in graph_spec.agents:
        node = Agent(
            name=spec.name,
            prompt_path=spec.prompt_path,
            logger=logger,
            tools=tools,
            temperature=spec.temperature if hasattr(spec, 'temperature') else None,
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