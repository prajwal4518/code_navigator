# Codebase Navigator 🧭

A RAG-based code exploration tool that helps you understand and navigate codebases using natural language.

## 🎯 What is this?

Codebase Navigator uses **Retrieval-Augmented Generation (RAG)** to let you ask questions about any codebase in plain English. It:

1. **Ingests** code using AST-aware parsing (not naive text splitting)
2. **Stores** semantic embeddings in a vector database
3. **Retrieves** relevant code using hybrid search (semantic + keyword)
4. **Answers** your questions using an LLM with retrieved context

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized development)
- Gemini API key (optional, for cloud LLM)

### Setup

```bash
# Clone and enter the repository
cd code_navigator

# Create environment file
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import code_navigator; print(f'✅ v{code_navigator.__version__}')"
```

### Using Docker

```bash
cd docker
docker compose up
```

## 📁 Project Structure

```
code_navigator/
├── src/code_navigator/     # Main application package
│   ├── ingestion/          # Code parsing & chunking
│   ├── vectorstore/        # ChromaDB integration
│   ├── retrieval/          # Hybrid search logic
│   ├── agents/             # LLM orchestration
│   └── api/                # FastAPI endpoints
├── tests/                  # Test suite
├── configs/                # YAML configurations
├── scripts/                # Utility scripts
├── data/                   # Local data (gitignored)
└── docker/                 # Containerization
```

## 🛠️ Development

```bash
# Run linting
ruff check src/

# Run formatting
ruff format src/

# Run tests
pytest

# Run all pre-commit hooks
pre-commit run --all-files
```

## 📖 Architecture

See the [implementation plan](./docs/implementation_plan.md) for detailed architecture documentation.

## 📝 License

MIT
