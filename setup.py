"""
One-time setup script — run before launching the app.

    python setup.py

Steps:
  1. Install Python dependencies
  2. Download spaCy model (en_core_web_sm, ~12 MB)
  3. Download NLTK resources
  4. Download MedQuAD dataset from GitHub (~100 MB) and parse XML
  5. Download HuggingFace embedding model    (PubMedBERT, ~420 MB, cached locally)
  6. Download cross-encoder re-ranker model  (~80 MB, cached locally)
  7. Build FAISS vector index      (encode 47k docs, ~5-15 min CPU)

Everything runs locally — no API keys, no ongoing costs.
Model is cached in ~/.cache/huggingface/ and reused on every run.
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*62}\n  {desc}\n{'='*62}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: exited with code {result.returncode}")


def main():
    python = sys.executable
    root   = Path(__file__).parent
    sys.path.insert(0, str(root))

    # 1. pip install
    run(
        [python, "-m", "pip", "install", "-r", "requirements.txt"],
        "Step 1/7 — Installing dependencies",
    )

    # 2. spaCy model
    #
    # NOTE: Some environments have issues where `spacy download en_core_web_sm`
    # resolves to a broken GitHub URL (missing version), resulting in 404.
    # Installing the wheel directly is more reliable for spaCy 3.7.x.
    #
    # If this wheel URL ever changes, you can fall back to:
    #   python -m spacy download en_core_web_sm
    #
    spacy_model_whl = (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl"
    )
    run(
        [python, "-m", "pip", "install", spacy_model_whl],
        "Step 2/7 — Installing spaCy model (en_core_web_sm, wheel)",
    )

    # 3. NLTK
    print(f"\n{'='*62}\n  Step 3/7 — Downloading NLTK resources\n{'='*62}")
    import nltk
    for r in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(r, quiet=False)

    # 4. MedQuAD dataset
    print(f"\n{'='*62}\n  Step 4/7 — Downloading MedQuAD dataset (~100 MB)\n{'='*62}")
    from src.data_loader import load_dataset
    records = load_dataset()
    print(f"  Dataset ready: {len(records):,} Q&A pairs")

    # 5. Pre-download HuggingFace embedding model
    print(f"\n{'='*62}")
    print( "  Step 5/7 — Downloading HuggingFace embedding model")
    print( "  Model : pritamdeka/S-PubMedBert-MS-MARCO")
    print( "  Size  : ~420 MB  |  Cached to ~/.cache/huggingface/")
    print( "  Cost  : FREE — runs 100% locally after this download")
    print(f"{'='*62}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        # Warm-up encode
        _ = model.encode(["test medical query"])
        print("  Model downloaded and ready.")
    except Exception as e:
        print(f"  WARNING: model download failed — {e}")
        print("  The FAISS build step may fail until the model is available.")
        print("  If you see a torch>=2.6 warning, run: pip install -U \"torch>=2.6\"")

    # 6. Pre-download cross-encoder re-ranker
    print(f"\n{'='*62}")
    print( "  Step 6/7 — Downloading cross-encoder re-ranker")
    print( "  Model : cross-encoder/ms-marco-MiniLM-L-6-v2")
    print( "  Size  : ~80 MB  |  Cached to ~/.cache/huggingface/")
    print(f"{'='*62}")
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        _ = reranker.predict([("test query", "test candidate")])
        print("  Re-ranker downloaded and ready.")
    except Exception as e:
        print(f"  WARNING: re-ranker download failed — {e}")
        print("  Re-ranking may fail until the model is available.")

    # 7. Build FAISS index
    print(f"\n{'='*62}")
    print( "  Step 7/7 — Building FAISS vector index")
    print(f"  Encoding {len(records):,} documents locally (CPU)")
    print( "  Estimated time: 5–15 min (larger model + answer indexing)")
    print(f"{'='*62}")

    from src.retriever import FAISS_DIR
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        print(f"  FAISS index already exists at {FAISS_DIR} — skipping.")
        print("  (Delete the folder and re-run to rebuild.)")
    else:
        try:
            from src.retriever import MedicalRetriever
            r2 = MedicalRetriever()
            r2.build_faiss(records)
            print("\n  FAISS index built successfully!")
        except Exception as e:
            print(f"  WARNING: FAISS build failed — {e}")
            print("  The app requires FAISS. Fix the error and re-run setup.py.")

    # Summary
    from src.retriever import FAISS_DIR
    faiss_ok = FAISS_DIR.exists() and any(FAISS_DIR.iterdir())

    print(f"""
{'='*62}
  Setup complete!

  Indices:
    {'✅' if faiss_ok else '❌'} FAISS   (local HuggingFace embeddings, higher accuracy)

  Launch:
    streamlit run app.py
{'='*62}
""")


if __name__ == "__main__":
    main()
