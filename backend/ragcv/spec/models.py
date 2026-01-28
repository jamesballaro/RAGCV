from typing import Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, field_validator, model_validator

class AgentConfig(BaseModel):
    name: str
    rank: int
    prompt_path: Optional[str] = None
    temperature: Optional[float] = 0.8
    conditional_links: Optional[List[str]] = None
    hard_links: Optional[List[str]] = None

    @field_validator("temperature")
    def check_temperature_range(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("temperature must be between 0 and 1")
        return v

    @field_validator("conditional_links", "hard_links")
    def empty_list_to_none(cls, v):
        return v or None

class GraphConfig(BaseModel):
    agents: List[AgentConfig]

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

    @classmethod
    def from_yaml(cls, path: str) -> 'GraphConfig':
        """Load and validate GraphConfig from a YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Graph config file not found: {path}")

        try:
            with open(path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML from {path}: {e}") from e

        if not cfg_dict:
            raise ValueError(f"Empty configuration file: {path}")

        return cls(**cfg_dict)

class RetrievalConfig(BaseModel):
    rag_threshold: int = 10000
    base_k: int = 15
    rerank_top_k: int = 5     # Final number of documents to return
    rerank_threshold: int = -2
    max_total_chunks: int = 10 
    use_hybrid: bool = True
    bm25_weight: float = 0.5  # 0.5 = equal weighting
    embedding_weight: float = 0.5

    @field_validator("bm25_weight", "embedding_weight")
    @classmethod
    def validate_weight_range(cls, v: float, info) -> float:
        """Ensure weights are in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1, got {v}")
        return v

    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'RetrievalConfig':
        """Ensure BM25 and embedding weights sum to approximately 1.0."""
        if self.use_hybrid:
            weight_sum = self.bm25_weight + self.embedding_weight
            if not (0.95 <= weight_sum <= 1.05):  # Allow small floating point error
                raise ValueError(
                    f"bm25_weight + embedding_weight should sum to 1.0, "
                    f"got {weight_sum:.3f} ({self.bm25_weight} + {self.embedding_weight})"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> 'RetrievalConfig':
        """Load and validate RetrievalConfig from a YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Retrieval config file not found: {path}")

        try:
            with open(path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML from {path}: {e}") from e

        if not cfg_dict:
            raise ValueError(f"Empty configuration file: {path}")

        return cls(**cfg_dict)
