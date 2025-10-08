# RAG-Powered CV & Cover Letter Builder

This repository provides a retrieval-augmented generation (RAG) pipeline for quickly creating tailored CVs and cover letters using large language models. It leverages past documents and templates to generate content relevant to a given job description.

## Features
- **RAG-Based Document Tailoring**: Uses your existing CVs, cover letters, and templates to generate contextually relevant output.
- **Multiple Use Cases**:
  1. Generate a CV tailored to a job description.
  2. Generate a cover letter tailored to a job description.
  3. Identify the most relevant experiences to include for a given job.
- **Customizable Prompts**: Easily modify system prompts for each agent to adjust tone, format, or style.
- **Extensible Architecture**: Designed with a modular agent graph, enabling integration of additional agents or workflows.

**Note**: While designed for the three main use cases above, the system can adapt to other document-related tasks due to the flexible nature of LLMs.

## Setup
1. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate ragcv

2. Set your OpenAI API key:
   export OPENAI_API_KEY="sk-xxxx"

3. Add input data:
- Place your job query in input/query.txt.
- Add previous CVs to data/CVs and cover letters to data/coverletters.
- Optionally, add templates to data/templates.

4. Customize system prompts (optional) in the prompts/ folder.

## Running the Pipeline
python3 src/main.py

- The pipeline will load documents, build a vectorstore (or reuse an existing one), and invoke the agent graph to produce tailored outputs.
- Results are written to the output/ folder.

## Tools
- write_to_file: Writes text content to a file within the ./output directory.

## Architecture Overview

- DataLoader: Loads CVs, cover letters, and templates, and chunks them appropriately for RAG.
- RouterGraph: Routes queries between specialized agents (CV agent, Cover Letter agent, Latex agent, etc.).
- Agents: Encapsulate specialized LLM chains, optionally using retrieval, with support for tool integration.
- Main Pipeline: Orchestrates the full workflow from input query to output generation and file saving.

## File Structure
├─ data/                  # Input documents
│  ├─ CVs/
│  ├─ coverletters/
│  └─ templates/
├─ input/
│  └─ query.txt           # Job description or prompt
├─ output/                # Generated CV/cover letters
├─ prompts/               # Agent system prompts
├─ src/
│  ├─ main.py             # Entry point
│  ├─ loader.py           # Data loading & chunking
│  ├─ graph.py            # Agent graph
│  ├─ agents.py           # Agent definitions
│  └─ tools.py            # Utility tools
└─ environment.yml        # Conda environment


## Notes
The project was primarily built to gain hands-on experience with LangChain, LangGraph, and RAG workflows. Some components use custom implementations instead of pre-built abstractions.
System prompts and chunking strategies are tailored for CVs and cover letters but can be adjusted for other document types.
Ensure your documents are properly formatted; PDFs, .tex, and .txt files are supported.
