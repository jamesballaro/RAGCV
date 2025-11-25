import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

class JSONLLogger:
    """Simple JSONL logger for agent diagnostics."""

    def __init__(self, log_path: str = "logs/agent_runs.jsonl"):
        self.log_path = log_path
        self._lock = threading.Lock()
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

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

