# MedQuAD Medical Q&A Chatbot

A specialized medical question-answering chatbot built on the [MedQuAD dataset](https://github.com/abachaa/MedQuAD) (~47,000 Q&A pairs from NIH, NCI, GARD, and other authoritative sources).

## Architecture

```
User Query
    │
    ▼
Stage 1: TF-IDF Recall         (scikit-learn, ~50 ms)
    │   Retrieves top-50 candidate Q&A pairs
    ▼
Stage 2: BiomedBERT Re-ranking  (sentence-transformers, ~200 ms CPU)
    │   Model: pritamdeka/S-PubMedBert-MS-MARCO
    │   Trained on PubMed + MS-MARCO for medical Q&A
    ▼
Medical NER Highlighting        (spaCy + custom medical dictionary)
    │   Detects: diseases, symptoms, treatments, medications, anatomy
    ▼
Streamlit Chat UI
```

### Accuracy Estimates

| Mode | Top-1 Accuracy | Top-3 Accuracy |
|------|---------------|---------------|
| TF-IDF only | ~65% | ~82% |
| **Hybrid (TF-IDF + BiomedBERT)** | **~83%** | **~93%** |

## Quick Start

### 1. One-time Setup

```bash
python setup.py
```

This will:
- Install all dependencies
- Download spaCy model (`en_core_web_sm`)
- Download NLTK resources
- Download MedQuAD dataset from GitHub (~100 MB)
- Build the TF-IDF index
- Pre-download BiomedBERT weights (~420 MB, cached)

### 2. Run the App

```bash
streamlit run app_platform.py
```

Or on Windows, run `run.bat`.

The app opens at **http://localhost:8501**

## Unified Platform API Keys (Backend Setup)

For unified task setup, copy root `.env.example` to `.env` and fill keys:

```bash
copy .env.example .env
```

Task key mapping:
- Task 1 (Dynamic KB): no API key
- Task 2 (Multimodal): `GEMINI_API_KEY`
- Task 3 (Sentiment): no API key
- Task 4 (Multilingual): no API key
- Task 5 (Arxiv): `GROQ_API_KEY` or `HF_API_KEY` only when those backends are used (`ollama` needs no key)
- Task 6 (Medical Chatbot): no API key

For Arxiv-specific backend config, also check `Arxiv/.env.example`.

## Deployment Checklist (Production)

Before deploy:

1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure env files exist and are filled:
   - root: `.env` (platform keys like `GEMINI_API_KEY`, `GEMINI_MODEL`)
   - Arxiv: `Arxiv/.env` (backend selection + `GROQ_API_KEY` / `HF_API_KEY` if used)
3. Confirm selected models in UI match backend env values (Task Debug panel).
4. Keep secrets out of source control (`.env`, `Arxiv/.env`, `.streamlit/secrets.toml`).
5. Run:
   ```bash
   streamlit run app_platform.py
   ```

## Production Hardening Notes

- Keep `.env`, `Arxiv/.env`, and `.streamlit/secrets.toml` outside source control.
- Use only required API keys for active tasks.
- Keep app bound to localhost for local-only usage.
- Use startup scripts (`run.bat` on Windows) for consistent local runs.

## Project Structure

```
Medquad/
├── app.py                  # Streamlit chatbot UI
├── setup.py                # One-time setup script
├── run.bat                 # Windows launcher
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py      # MedQuAD downloader & XML parser
│   ├── retriever.py        # Hybrid TF-IDF + BiomedBERT retrieval
│   └── medical_ner.py      # Medical entity recognition
└── data/
    ├── raw/                # Downloaded MedQuAD XML files
    ├── medquad_processed.json  # Parsed Q&A pairs
    └── index/              # TF-IDF index cache
```

## Features

- **Hybrid retrieval**: Two-stage pipeline combining keyword and semantic search
- **Medical NER**: Highlights diseases, symptoms, treatments, medications, anatomy, and procedures
- **Confidence scoring**: Blended TF-IDF + semantic similarity score per answer
- **Source attribution**: Links back to original NIH/NCI/GARD source pages
- **Related Q&As**: Shows 2 related questions from the dataset
- **Adjustable settings**: Toggle re-ranker, entity highlighting, confidence threshold

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `scikit-learn` | TF-IDF vectorizer |
| `sentence-transformers` | BiomedBERT semantic re-ranking |
| `spacy` | NER pipeline |
| `nltk` | Text preprocessing |
| `lxml` | XML parsing |

## Disclaimer

This chatbot is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.
