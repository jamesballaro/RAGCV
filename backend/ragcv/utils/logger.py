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
        self.conversation_log: List[Dict] = []

        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as log_file:
                pass  

    def log(self, payload: Dict[str, Any]) -> None:
        """Write a payload as a JSON line with timestamp metadata."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        # JSONL for easier parsing
        serialized = json.dumps(entry, default=self._fallback_serializer)
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
    
    
    def log_agent_invocation(
        self,
        agent_name: str, 
        input_message: List[AnyMessage] | None,
        output_message: List[AnyMessage],
        tool_logs: List[dict]) -> None:
        
        payload={
            "agent_name": agent_name,
            "event": "agent_invocation",
            
            # Take last message
            "input_message": input_message,
            "output_message": output_message,
            "tool_calls": tool_logs,
        }
        self.log(payload=payload)
        self.conversation_log.append(payload)
    
    def log_agent_error(self, agent_name, error_message):
        payload={
            "agent_name": agent_name,
            "event": "agent_invocation",
            "error_message": error_message
        }
        self.log(payload=payload)
        self.conversation_log.append(payload)

   
    def log_conversation(self) -> None:
        conversation = []
        for message in self.conversation_log:
            name = message['agent_name']
            content = message.get('output_message', None)
            if content is None: 
                content = message.get('error_message')
            conversation.append({
                'agent_name': name,
                'content': content
            })

        self.log(payload={
                "full_conversation": conversation
            }
        )
    def get_conversation_log(self) -> None:
        conversation = []
        for message in self.conversation_log:
            name = message['agent_name']
            content = message['output_message']
            conversation.append({
                'agent_name': name,
                'content': content
            })

        return conversation