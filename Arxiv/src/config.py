import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "papers_cache.json"
FAISS_DIR = DATA_DIR / "faiss_index"

DATA_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

HF_API_KEY: str = os.getenv("HF_API_KEY", "")
HF_MODEL: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG ───────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 5
MAX_CONTEXT_PAPERS: int = 4
MAX_TOKENS_CONTEXT: int = 3000

# ── ArXiv ─────────────────────────────────────────────────────────────────
DEFAULT_CATEGORY: str = os.getenv("DEFAULT_CATEGORY", "cs")
INITIAL_PAPER_COUNT: int = int(os.getenv("INITIAL_PAPER_COUNT", "200"))
ARXIV_RATE_LIMIT_DELAY: float = 3.0  # seconds between API calls

ARXIV_CATEGORIES: dict[str, str] = {
    "Computer Science": "cs",
    "Mathematics": "math",
    "Physics": "physics",
    "Statistics": "stat",
    "Quantitative Biology": "q-bio",
    "Electrical Engineering": "eess",
    "Economics": "econ",
}

CS_SUBCATEGORIES: dict[str, str] = {
    "All CS": "cs",
    "AI / Machine Learning": "cs.AI",
    "Computer Vision": "cs.CV",
    "Computation & Language (NLP)": "cs.CL",
    "Machine Learning (Systems)": "cs.LG",
    "Neural Networks": "cs.NE",
    "Robotics": "cs.RO",
    "Cryptography": "cs.CR",
    "Distributed Computing": "cs.DC",
    "Data Structures & Algorithms": "cs.DS",
    "Software Engineering": "cs.SE",
    "Human-Computer Interaction": "cs.HC",
    "Computer Vision & Pattern Recognition": "cs.CV",
    "Information Retrieval": "cs.IR",
}

# ── App ────────────────────────────────────────────────────────────────────
APP_TITLE: str = os.getenv("APP_TITLE", "ArXiv Expert Chatbot")
APP_ICON: str = "🔬"
