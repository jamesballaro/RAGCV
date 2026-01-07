from typing import Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, field_validator, model_validator

class AgentConfig(BaseModel):
    name: str
    prompt_path: str
    rank: int
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
    base_k: int = 10
    mmr_k: int = 5
    mmr_lambda: float = 0.6
    score_threshold: float = 0.15
    min_high_score: int = 5
    dedupe_threshold: float = 0.88
    # hybrid retrieval settings

    use_hybrid: bool = True
    bm25_weight: float = 0.5
    embedding_weight: float = 0.5



    @field_validator("base_k", "min_high_score", "mmr_k")
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        """Ensure k values are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator("score_threshold", "dedupe_threshold")
    @classmethod
    def validate_threshold_range(cls, v: float, info) -> float:
        """Ensure thresholds are in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1, got {v}")
        return v

    @field_validator("mmr_lambda")
    @classmethod
    def validate_mmr_lambda(cls, v: float) -> float:
        """Ensure MMR lambda is in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"mmr_lambda must be between 0 and 1, got {v}")
        return v

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

    @model_validator(mode='after')
    def validate_mmr_k_vs_base_k(self) -> 'RetrievalConfig':
        """Ensure MMR k doesn't exceed base k."""
        if self.mmr_k > self.base_k:
            raise ValueError(
                f"mmr_k ({self.mmr_k}) cannot exceed base_k ({self.base_k})"
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
