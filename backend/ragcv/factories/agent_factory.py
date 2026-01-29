from typing import Optional, Dict

from ..core.agent import Agent
from .prompt_factory import PromptFactory
from ..spec.output_models import *
from ..spec.registry import PYDANTIC_REGISTRY
from ..utils.logger import JSONLLogger


class BaseAgentFactory:
    def __init__(self):
        self.prompt_factory = PromptFactory()

    def create_agent(self) -> Agent:
        pass

class SpecialisedAgentFactory(BaseAgentFactory):
    def __init__(self):
        super().__init__()

    def determine_role(self, name: str) -> Dict:
        prompt = self.prompt_factory.create_prompt(name)
        output_parser = PYDANTIC_REGISTRY[name]
        
        return {
            "prompt": prompt,
            "output_parser": output_parser,
        }

    def create_agent(self, name: str, tools: list = None, **kwargs) -> Agent:
        role = self.determine_role(name)

        return Agent(
            name=name,
            prompt=role['prompt'],
            output_parser=role['output_parser'],
            tools=tools,
            **kwargs
        )
