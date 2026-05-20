# Multi-RAG Agent System

> Multi-agent Retrieval-Augmented Generation repo powered by Google Gemini, LangChain, LangGraph, and Chroma.

## Overview

This repository implements a multi-RAG agent system that can:

- route user queries to domain-specific RAG agents (`finance`, `tech`, `general`)
- perform document retrieval from vector stores
- rerank and synthesize responses
- expose REST API endpoints via FastAPI
- support frontend usage and notebook-based testing

## Prerequisites

- Python 3.11+ (recommended)
- Node.js 18+ / npm
- Google API key with Gemini access

## 1. Setup Environment

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Get your Google API Key:
   - Go to https://aistudio.google.com/apikey
   - Create a new API key

4. Create `.env` file from `.env.example`:

   ```bash
   cp .env.example .env
   ```

5. Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 2. Prepare Data

1. Place your documents in the appropriate directories:
   - Finance documents: `data/raw/finance/`
   - Tech documents: `data/raw/tech/`
   - General documents: `data/raw/general/`

2. Supported formats: `.pdf`, `.txt`, `.md`, `.docx`

## 3. Ingest Data into Vector Stores

```python
from src.ingestion import DocumentLoader, DocumentChunker
from src.retrieval import vector_store_manager
from pathlib import Path

# Load documents
loader = DocumentLoader()
docs = loader.load_from_directory("data/raw/finance")

# Chunk documents
chunker = DocumentChunker()
chunks = chunker.chunk(docs)

# Add to vector store
vector_store_manager.add_documents("finance", chunks)
```

## 4. Use the System

### Option A: Use FastAPI Server

```bash
uvicorn api.main:app --reload --port 8000
```

Then visit: http://localhost:8000/docs

### Option A.1: Use React Chat UI

Start the FastAPI backend first:

```bash
uvicorn api.main:app --reload --port 8000
```

In a second terminal, start the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open: http://localhost:5173

The frontend calls the backend at `http://localhost:8000` by default. To use a different backend URL, create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

For production, build the frontend:

```bash
cd frontend
npm run build
```

If `frontend/dist` exists, FastAPI also serves it at: http://localhost:8000/app

### Option B: Use Directly in Python

```python
from src.graph import run_workflow

query = "Tell me about Apple's latest earnings"
result = run_workflow(query)

print(result["final_answer"])
print(result["sources"])
```

### Option C: Use Individual Agents

```python
from src.agents import RouterAgent, RAGAgent

# Router decides which domain
router = RouterAgent()
route = router.run({"query": "What is Python?"})

# RAG Agent answers
rag_agent = RAGAgent("tech")
answer = rag_agent.run({"query": "What is Python?"})
```

## 5. API Examples

### Single Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### Batch Queries

```bash
curl -X POST "http://localhost:8000/api/query/batch" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Query 1", "Query 2", "Query 3"]}'
```

### Get Routing Decision

```bash
curl -X POST "http://localhost:8000/api/route" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about stocks"}'
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

### System Stats

```bash
curl http://localhost:8000/api/stats
```

## 6. Project Structure

```
multi-rag-agent/
├── config/              # Configuration files
├── data/               # Data storage
│   ├── raw/           # Input documents
│   ├── processed/     # Processed data
│   └── vectorstores/  # Vector DBs
├── src/
│   ├── agents/        # Agent implementations
│   ├── graph/         # LangGraph workflow
│   ├── ingestion/     # Data loading & processing
│   ├── retrieval/     # RAG components
│   ├── tools/         # Agent tools
│   └── utils/         # Utilities
├── api/               # FastAPI application
├── logs/              # Application logs
└── notebooks/         # Jupyter experiments
```

## 7. Key Components

- **Router Agent**: Classifies queries into domains
- **RAG Agents**: Domain-specific retrieval and generation
- **Supervisor Agent**: Coordinates multi-domain queries
- **Synthesizer Agent**: Combines agent outputs
- **Vector Store Manager**: Manages Chroma vector databases
- **Reranker**: Re-ranks documents by relevance
- **Query Expander**: Expands queries for better retrieval

## 8. Configuration

All settings are in `config/settings.py`. Key configurations:

- `google_api_key`: Your Google API key
- `default_llm_model`: Default model (gemini-2.0-flash)
- `embedding_model`: Embedding model (gemini-embedding-001)
- `top_k_retrieval`: Number of documents to retrieve (10)
- `top_k_rerank`: Number of documents after reranking (4)

## 9. Troubleshooting

### "No documents found"

- Make sure documents are in correct directories
- Run ingestion script to populate vector stores

### "API key error"

- Check that GOOGLE_API_KEY is set in .env
- Verify key is valid on https://aistudio.google.com/apikey

### "Module not found"

- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

## 10. Next Steps

1. Customize system prompts in `config/agents.yaml`
2. Adjust retrieval parameters in `config/settings.py`
3. Add domain-specific document loaders
4. Implement custom tools in `src/tools/`
5. Deploy using Docker or cloud platform

## Support

For questions or issues:

1. Check the guide in `multi_rag_agent_guide.md`
2. Review logs in `logs/` directory
3. Test individual components in Jupyter notebooks
