# Logseq RAG (WIP)

Personal RAG system for querying daily journal/notes in Logseq markdown format.

## Things

- **UV** - Package/env management
- **ChromaDB** - Local vector database with automatic embeddings
- **LangChain** - Text splitting utilities
- **LiteLLM**
  - **OpenAI**

## Quick Start

```bash
# Install dependencies with UV
uv sync

# Set environment
create .env similar to .env.example

# Chunk and embed documents
uv run rag.py

# Query your notes
uv run user_query.py
```