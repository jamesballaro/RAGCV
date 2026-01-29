import os
from typing import Callable, List, Any
from langchain.tools import tool

def build_registry(test_name: str) -> List[Any]:
    return [
        tool(
            name_or_callable="write_to_file",
            runnable=make_write_to_file(test_name=test_name),
            description="Writes content to a file in the output directory.",
        )
    ]

def make_write_to_file(test_name: str) -> Callable[[str, str], str]:
    def inner(content: str, filename: str = "output.txt") -> str:
        os.makedirs(os.path.join("output", test_name), exist_ok=True)
        filepath = os.path.join("output", test_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[FileTool] Output successfully written to {filepath}"
    return inner

