import os 

def write_to_file(content: str, filename: str = "output.txt") -> str:
    filepath = os.path.join("output", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"[FileTool] Output successfully written to {filepath}"
