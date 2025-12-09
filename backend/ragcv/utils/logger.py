import json
import os
import threading
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, AnyMessage
from typing import Any, Dict, List, Annotated

class JSONLLogger:
    """Simple JSONL logger for agent diagnostics."""

    def __init__(self, log_path: str = "logs/agent_runs.jsonl"):
        self.log_path = log_path
        self._lock = threading.Lock()
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # Check if log file exists, if so, empty it
        if os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as log_file:
                pass  # This will empty the file

    def log(self, payload: Dict[str, Any]) -> None:
        """Write a payload as a JSON line with timestamp metadata."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        serialized = json.dumps(entry, indent=4, default=self._fallback_serializer)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")

    @staticmethod
    def _fallback_serializer(obj: Any) -> Any:
        """Ensure non-serializable objects degrade gracefully."""
        try:
            return str(obj)
        except Exception:
            return repr(obj)
        
    @staticmethod
    def serialize_langchain_message(message: BaseMessage) -> dict:
        return {
            "type": message.__class__.__name__,
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", []),
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
            # "response_metadata": getattr(message, "response_metadata", {}),
        }
    
    def log_agent_invocation(
        self,
        agent_name: str, 
        input_messages: List[AnyMessage], 
        output_messages: List[AnyMessage],
        tool_logs: List[dict]) -> None:

        serialized_input = [JSONLLogger.serialize_langchain_message(msg) for msg in input_messages]
        serialized_output = [JSONLLogger.serialize_langchain_message(msg) for msg in output_messages]
        
        self.log(payload={
                "agent_name": agent_name,
                "event": "agent_invocation",
                "input_messages": serialized_input,
                "output_messages": serialized_output,
                "tool_calls": tool_logs,
            }
        )

    def log_conversation(self, messages: List[AnyMessage]) -> None:
        formatted_messages = []
        for msg in messages:
            metadata = JSONLLogger.serialize_langchain_message(msg)

            name = "Unknown"
            additional_kwargs = metadata.get('additional_kwargs', {})
            if isinstance(additional_kwargs, dict):
                agent_name = additional_kwargs.get("agent_name")
                if agent_name:
                    name = agent_name

            formatted_messages.append(f"Agent name:{name},\n\n{metadata['content']}")

        self.log(payload={
                "full_conversation": formatted_messages
            }
        )