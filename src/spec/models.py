from typing import Optional, List
from pydantic import BaseModel, field_validator

class AgentSpec(BaseModel):
    name: str
    prompt_path: str
    rank: int
    conditional_links: Optional[List[str]] = None
    hard_links: Optional[List[str]] = None

    @field_validator("conditional_links", "hard_links")
    def empty_list_to_none(cls, v):
        return v or None

class GraphSpec(BaseModel):
    agents: List[AgentSpec]

    @field_validator("agents")
    def validate_graph(cls, agents):
        names = {a.name for a in agents}

        # Validate links
        for a in agents:
            if a.conditional_links:
                unknown = set(a.conditional_links) - names
                if unknown:
                    raise ValueError(f"{a.name}: unknown conditional_links: {unknown}")

            if a.hard_links:
                unknown = set(a.hard_links) - names
                if unknown:
                    raise ValueError(f"{a.name}: unknown hard_links: {unknown}")

        # Validate ranks: must be contiguous (your code assumes this)
        ranks = list(set(sorted(a.rank for a in agents)))
        if ranks != list(range(min(ranks), max(ranks) + 1)):
            raise ValueError(f"Ranks must be contiguous integers. Got: {ranks}")

        return agents