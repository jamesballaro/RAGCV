import os
import uuid
import asyncio
import tempfile
import subprocess
import shutil
import uvicorn
from typing import Dict
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

from ragcv.core.loader import DataLoader
from ragcv.graph.graph import RouterGraph
from ragcv.tools.tools import build_registry
from ragcv.utils.logger import JSONLLogger
from ragcv.retrieval.enricher import QueryEnricher
from ragcv.retrieval.retrieval import AdaptiveRetriever
from ragcv.spec.loader import load_graph_config, load_retrieval_config

load_dotenv()

# =====================================================================
# LIFESPAN (Resource Management)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles heavy resource loading exactly once. 
    State is shared with all routes via app.state.
    """
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    # 1. Initialize Vectorstore
    loader = DataLoader(data_path="data", db_path="data/db.faiss")
    embeddings = OpenAIEmbeddings()
    vectorstore = loader.load_vectorstore(embeddings=embeddings)
    
    # 2. Setup Retriever
    retrieval_cfg = load_retrieval_config(path="config/retrieval.yml")
    app.state.retriever = AdaptiveRetriever(
        vectorstore=vectorstore, 
        embeddings=embeddings, 
        config=retrieval_cfg
    )
    
    print("\n[System Ready] Vectorstore and Retriever initialized in app.state.")
    print("="*60 + "\n")
    
    yield  # Application runs here
    
    print("Shutting down system...")

# Initialize App with lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# MODELS & SHARED STORAGE
# =====================================================================

class QueryRequest(BaseModel):
    text: str

class LaTeXRequest(BaseModel):
    latex: str

# Persistent background task store
tasks: Dict[str, dict] = {}

def read_file_safe(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# =====================================================================
# CORE LOGIC (BACKGROUND TASKS)
# =====================================================================

async def run_graph_task(task_id: str, input_text: str, retriever: AdaptiveRetriever):
    """
    Executes the RAG pipeline. 
    Note: retriever is passed from app.state at the time of task creation.
    """
    log_path = f"logs/task_{task_id}.jsonl"
    output_dir = f"output/{task_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    tasks[task_id]["status"] = "processing"
    
    try:
        task_logger = JSONLLogger(log_path=log_path)
        enricher = QueryEnricher(retriever=retriever, logger=task_logger)
        tools_list = build_registry(test_name=task_id) 
        
        graph_cfg = load_graph_config(path="config/graph.yml", tools=tools_list, logger=task_logger)
        graph = RouterGraph(agents=graph_cfg, logger=task_logger)

        # Step 1: Enrichment
        retrieved_docs = await asyncio.to_thread(enricher.get_retrieved_artifacts, input_text)
         
        tasks[task_id]["artifacts"] = retrieved_docs

        graph_input = {
            'job_description': HumanMessage(content=input_text),
            'retrieved_documents': retrieved_docs,
        }

        # Step 2: Graph Execution
        await asyncio.to_thread(graph.invoke, graph_input)

        # Step 3: Collect Results
        tasks[task_id]["result"] = read_file_safe(f"{output_dir}/cl_output.txt")
        tasks[task_id]["summary"] = read_file_safe(f"{output_dir}/summary_output.txt")
        tasks[task_id]["status"] = "completed"
        
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        print(f"Task {task_id} failed: {e}")

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.post("/query")
async def start_query(request: QueryRequest, background_tasks: BackgroundTasks):
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks[task_id] = {
        "status": "pending", 
        "result": None, 
        "summary": None, 
        "artifacts": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Pass the retriever from app.state to the background task
    background_tasks.add_task(run_graph_task, task_id, request.text, app.state.retriever)
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/logs/{task_id}")
async def get_logs(task_id: str, tail: int = 50):
    log_path = f"logs/task_{task_id}.jsonl"
    if not os.path.exists(log_path):
        return {"logs": "Initializing task logs..."}
    
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-tail:])}
    except Exception as e:
        return {"logs": f"Error: {str(e)}"}

@app.post("/compile_latex")
async def compile_latex(request: LaTeXRequest):
    temp_dir = tempfile.mkdtemp()
    try:
        tex_file = os.path.join(temp_dir, "document.tex")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(request.latex)
        
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail=result.stderr.decode(errors="ignore")
                )
        
        pdf_file = os.path.join(temp_dir, "document.pdf")
        if not os.path.exists(pdf_file):
            raise HTTPException(status_code=400, detail="LaTeX compilation failed")
        
        with open(pdf_file, "rb") as f:
            return Response(content=f.read(), media_type="application/pdf")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# =====================================================================
# STATIC FILE SERVING
# =====================================================================

FRONTEND_BUILD_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_BUILD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_BUILD_DIR / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend not found. Run build in /frontend."}

# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    # 1. We use the "string import" format 'server:app' for reload to work
    # 2. We exclude logs and output from the watcher to prevent reload loops
    if not FRONTEND_BUILD_DIR.exists():
        print("\n" + "!"*60)
        print("WARNING: Frontend build folder 'dist' not found.")
        print("The API will work, but the web UI will not be served.")
        print("!"*60 + "\n")
    else:
        print("\n" + "="*60)
        print(f"Frontend build found: {FRONTEND_BUILD_DIR}")
        print("App will be available at: http://localhost:8000")
        print("="*60 + "\n")

    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        reload_excludes=["logs/*", "output/*", "**/__pycache__/*"],
        log_level="info"
    )