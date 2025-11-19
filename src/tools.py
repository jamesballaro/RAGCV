import os

from langchain.tools import tool

@tool("write_to_file", return_direct=True)
def write_to_file(content: str, filename: str = "output.txt") -> str:
    """
    Writes text content to a file within the ./output directory.
    """
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"[FileTool] Output successfully written to {filepath}"
