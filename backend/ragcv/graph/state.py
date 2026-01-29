from typing import TypedDict, List, Dict, Any

class RouterGraphState(TypedDict):
    # Graph messages
    latest_message: Dict | None

    # Document generation 
    retrieved_documents: List[Dict]
    job_description: str | None
    task: str | None
    summary: str | None
    blueprint: str | None
    document: str | None

