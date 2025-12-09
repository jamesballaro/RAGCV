from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import subprocess
import tempfile
import os
import shutil

# Import your existing logic
# Ensure these imports match your actual folder structure relative to server.py
from ragcv.loader import DataLoader
from ragcv.graph import RouterGraph
from ragcv.tools.tools import build_registry
from ragcv.utils.logger import JSONLLogger
from ragcv.retrieval.enricher import QueryEnricher
from ragcv.retrieval.retrieval import AdaptiveRetriever
from ragcv.spec.loader import load_from_yaml
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

class LaTeXRequest(BaseModel):
    latex: str

# Save test_name globally so it's accessible everywhere needed
TEST_NAME = f"web_request_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_PATH = f"logs/server_log_{TEST_NAME}.jsonl"
OUTPUT_PATH = f"output/{TEST_NAME}"

# Initialize components globally so we don't reload FAISS on every request
loader = DataLoader(data_path="data", db_path="data/db.faiss")
embeddings = OpenAIEmbeddings()
vectorstore = loader.load_vectorstore(embeddings=embeddings)
logger = JSONLLogger(log_path=LOG_PATH)
retriever = AdaptiveRetriever(vectorstore=vectorstore, embeddings=embeddings)
enricher = QueryEnricher(retriever=retriever, logger=logger)
tools_list = build_registry(test_name=TEST_NAME)
agents = load_from_yaml(path="config/graph.yml", tools=tools_list, logger=logger)
graph = RouterGraph(agents=agents, logger=logger)

@app.post("/query")
async def run_query(request: QueryRequest):
    # 1. Enrich query
    message = enricher.enrich_with_context(request.text)

    # 2. Invoke Graph
    output = graph.invoke({'messages': [HumanMessage(content=message)]})
    # 3. Choose output file based on test_name
    if TEST_NAME:
        output_path = f"{OUTPUT_PATH}/cl_output.txt"
    else:
        output_path = "output/cl_output.txt"
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        file_content = f"Error reading {output_path}: {str(e)}"
    return {"result": file_content}
# Add this to your server.py after the /query endpoint

@app.post("/compile_latex")
async def compile_latex(request: LaTeXRequest):
    """
    Compile LaTeX source to PDF using pdflatex.
    Returns the PDF as a binary response.
    """
    # Create a temporary directory for compilation
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Write LaTeX source to .tex file
        tex_file = os.path.join(temp_dir, "document.tex")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(request.latex)
        
        # Run pdflatex twice (for proper references/TOC if needed)
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
        
        # Check if PDF was generated
        pdf_file = os.path.join(temp_dir, "document.pdf")
        if not os.path.exists(pdf_file):
            # Try to get error from log
            log_file = os.path.join(temp_dir, "document.log")
            error_msg = "LaTeX compilation failed"
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    log_content = f.read()
                    # Extract first error line
                    for line in log_content.split('\n'):
                        if line.startswith('!'):
                            error_msg = line
                            break
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read PDF and return as binary
        with open(pdf_file, "rb") as f:
            pdf_content = f.read()
        
        return Response(content=pdf_content, media_type="application/pdf")
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="LaTeX compilation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compilation error: {str(e)}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
@app.get("/logs")
async def get_logs():
    """Return the contents of the server log file"""
    log_path = LOG_PATH
    try:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            return {"logs": log_content}
        else:
            return {"logs": "Log file not found. No requests processed yet."}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


