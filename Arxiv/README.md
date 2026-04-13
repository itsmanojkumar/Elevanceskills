# ArXiv Expert Chatbot

An AI-powered scientific paper chatbot built with RAG (Retrieval-Augmented Generation), Streamlit, and open-source LLMs. Specialized in Computer Science research from the arXiv dataset.

## Features

- **Expert Chatbot** — Ask complex questions about CS topics with citations to real arXiv papers
- **RAG Pipeline** — Retrieves relevant papers and uses them as context for accurate answers
- **Multi-turn Conversations** — Handles follow-up questions with full conversation history
- **Paper Search** — Semantic search across arXiv papers with filtering by category and date
- **Paper Summarization** — Paste any arXiv URL or ID to get an AI-generated summary
- **Concept Visualization** — Interactive knowledge graphs, word clouds, topic clusters
- **Analytics Dashboard** — Publication trends, category distributions, author networks
- **Multiple LLM Backends** — Supports Ollama (local), Groq (free API), HuggingFace

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo>
cd Arxiv
pip install -r requirements.txt
```

### 2. Configure LLM backend

Copy `.env.example` to `.env` and fill in your preferred settings:

```bash
cp .env.example .env
```

#### Option A: Ollama (Local, Recommended)

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3.2` or `ollama pull mistral`
3. Set in `.env`: `LLM_BACKEND=ollama`

#### Option B: Groq (Free Cloud API, Fastest)

1. Get free API key at [console.groq.com](https://console.groq.com)
2. Set in `.env`:
   ```
   LLM_BACKEND=groq
   GROQ_API_KEY=your_key_here
   ```

#### Option C: HuggingFace (Free Cloud API)

1. Get free API key at [huggingface.co](https://huggingface.co/settings/tokens)
2. Set in `.env`:
   ```
   LLM_BACKEND=huggingface
   HF_API_KEY=your_key_here
   ```

### 3. (Optional) Load the Kaggle ArXiv Dataset

For a richer knowledge base, download the [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle and place the JSON file at:

```
data/arxiv-metadata-oai-snapshot.json
```

The app will automatically detect and index it on startup.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Architecture

```
User Query
    │
    ▼
Query Embedding (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Search ──→ Top-K Relevant Papers
    │                         │
    │                   ArXiv API (fresh results)
    │                         │
    ▼                         ▼
RAG Prompt Builder (context + conversation history)
    │
    ▼
LLM (Ollama / Groq / HuggingFace)
    │
    ▼
Response with Citations + Visualizations
```

## Project Structure

```
Arxiv/
├── app.py                 # Main Streamlit application
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── config.py          # Configuration & settings
│   ├── data_loader.py     # ArXiv API + Kaggle dataset loader
│   ├── embeddings.py      # Sentence transformers + FAISS index
│   ├── llm_handler.py     # Multi-backend LLM (Ollama/Groq/HF)
│   ├── rag_pipeline.py    # RAG chain + conversation management
│   └── visualizations.py # Plotly charts, word clouds, graphs
└── data/
    ├── papers_cache.json  # Cached paper metadata
    └── faiss_index/       # FAISS vector store
```

## LLM Models Tested

| Backend    | Model                        | Speed  | Quality |
|------------|------------------------------|--------|---------|
| Ollama     | llama3.2:3b                  | Fast   | Good    |
| Ollama     | mistral:7b                   | Medium | Great   |
| Ollama     | deepseek-r1:7b               | Medium | Great   |
| Groq       | llama-3.1-8b-instant         | Very Fast | Good |
| Groq       | mixtral-8x7b-32768           | Fast   | Great   |
| HuggingFace| mistralai/Mistral-7B-Instruct| Slow   | Good    |
