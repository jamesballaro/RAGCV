# Job Application Assistant - User Guide

AI-powered tool that generates tailored CVs and cover letters by retrieving relevant experiences from your document history.

## What It Does

Paste a job description → System finds your relevant experiences → Generates customized documents in seconds.

**Not a chatbot that makes things up**. All content comes from YOUR uploaded CVs and cover letters.

## How It Works

### 1. Smart Document Search

When you submit a job description, the system uses **hybrid retrieval**:

- **Semantic matching**: Finds conceptually similar content (e.g., "machine learning" matches "neural networks")
- **Keyword matching**: Catches exact terms (e.g., "React", "AWS", "SQL")
- **Diversity ranking**: Ensures variety (no redundant bullets about the same project)

Example: Job requires "Python ML experience"
- ✓ Retrieves: "Built PyTorch pipeline...", "Deployed TensorFlow model..."
- ✗ Filters out: Generic Python web dev unrelated to ML

### 2. Multi-Agent Workflow

Your query flows through specialized AI agents:

```
Job Description
    ↓
[Summary Agent] → Extracts key requirements
    ↓
[Router Agent] → Decides: CV, cover letter, or both?
    ↓
[Task Agent] → Plans which experiences to include
    ↓
[Creator Agent] → Generates the document
    ↓
[Quality Checker] → Reviews completeness
    ↓
Final Output
```

*Why multiple agents?** Each specializes in one task, producing better results than a single "do-everything" agent.

### 3. Context-Aware Generation

Every agent sees:
- Original job description
- Retrieved document chunks with relevance scores
- Conversation history from previous agents

Retrieved chunks are annotated:
```
[CV_ML.pdf | cv | score: 0.87]
Built production ML pipeline processing 10M+ transactions...
```

Agents prioritize high-score chunks (>0.7) and use low-score chunks (<0.3) only as inspiration.

## Using the Web Interface

### Input Panel

**Paste your job description** - the more detail, the better:

✅ **Good query**:
```
Senior Backend Engineer at Stripe

Requirements:
- 5+ years Python/Ruby
- Distributed systems experience
- Payment processing knowledge
- Strong API design

Responsibilities:
- Build microservices handling millions TPS
- Design REST/gRPC APIs
- Mentor junior engineers
```

❌ **Too vague**: "Software engineer job"

### Output Panel

Three tabs:

1. **Outputs**: Generated CV/cover letter text
2. **Logs**: Detailed execution trace showing agent decisions and retrieved documents
3. **LaTeX**: Live PDF preview with editable source code

### LaTeX Panel Features

- **Auto-compile**: Text → professional PDF instantly
- **Edit source**: Modify LaTeX and see changes in real-time
- **Download**: Save compiled PDF
- **Toggle view**: Switch between PDF preview and LaTeX source

### Understanding Logs

Logs show exactly what happened:

```json
{
  "agent_name": "Router_Agent",
  "output": "Routing to CV_Task_Agent based on technical requirements"
}
```

Look for:
- **Agent sequence**: Which agents ran, in order
- **Routing decisions**: Why Router chose CV vs cover letter
- **Retrieved docs**: What was pulled from your history (check early entries)
- **Tool calls**: When files were written
