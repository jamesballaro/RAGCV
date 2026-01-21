from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..graph.state import RouterGraphState

class StateProcessor:
    def __init__(self, name: str) -> None:
        self.name = name

    def prepare_input(self, state: RouterGraphState) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        if "Task_Agent" not in self.name:
            return state

        latest_message = state.get("latest_message") or {}
        status: Optional[str] = latest_message.get('status', "NEW")

        # Base payload for the task agent
        output = {
            'task_agent_input':{
                "status": status,
                "summary": state.get("summary"),
            },
             "retrieved_documents": state.get("retrieved_documents"),
        }

        # Retry-specific augmentation
        if status == "RETRY":
            output.update({
                "critique": latest_message.get("critique"),
                "specific_fix_instructions": latest_message.get("specific_fix_instructions"),
            })

        return output
    
    def prepare_output(self, agent_output, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return state

        output_state = state.copy()
        agent_data = agent_output.model_dump()

        # Copy agent keys to top-level
        for k, v in agent_data.items():
            output_state[k] = v  # now top-level 'task' will be updated

        output_state["latest_message"] = agent_data

        return output_state
