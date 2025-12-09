import os
from langchain.tools import tool
from functools import partial

def build_registry(test_name: str):
    return [
        tool(
            name_or_callable="write_to_file",
            runnable=make_write_to_file(test_name=test_name),
            description="Writes content to a file in the output directory.",
        )
    ]

#Factory method instead of functools.partial which is incompatible with langchain
def make_write_to_file(test_name: str):
    def inner(content: str, filename: str = "output.txt") -> str:
        os.makedirs(os.path.join("output", test_name), exist_ok=True)
        filepath = os.path.join("output", test_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[FileTool] Output successfully written to {filepath}"
    return inner

