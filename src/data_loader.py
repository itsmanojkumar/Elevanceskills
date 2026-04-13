"""
MedQuAD Dataset Downloader and Parser.

Downloads the MedQuAD dataset from GitHub, parses XML Q&A files,
and saves a clean processed dataset to disk.
"""

import os
import io
import zipfile
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_FILE = DATA_DIR / "medquad_processed.json"
MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"


def download_and_extract(dest_dir: Path) -> Path:
    """Download the MedQuAD ZIP from GitHub and extract it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "medquad_master.zip"

    if not zip_path.exists():
        print("Downloading MedQuAD dataset from GitHub (this may take a while)...")
        response = requests.get(MEDQUAD_ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        print("Download complete.")
    else:
        print("ZIP already exists, skipping download.")

    extract_dir = dest_dir / "MedQuAD-master"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print("Extraction complete.")

    return extract_dir


def parse_xml_file(xml_path: str) -> list[dict]:
    """Parse a single MedQuAD XML file and return list of Q&A records."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Determine document-level metadata
        doc_id = root.get("id", "")
        source = root.get("source", "")
        url = root.get("url", "")

        # Focus element (the medical topic)
        focus_elem = root.find("Focus")
        focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else ""

        # FocusAnnotations → semantic type
        sem_types = []
        for sem in root.findall(".//SemanticType"):
            if sem.text:
                sem_types.append(sem.text.strip())

        # QAPairs
        for qa_pair in root.findall(".//QAPair"):
            pid = qa_pair.get("pid", "")
            q_elem = qa_pair.find("Question")
            a_elem = qa_pair.find("Answer")

            if q_elem is None or a_elem is None:
                continue

            question = (q_elem.text or "").strip()
            answer = (a_elem.text or "").strip()
            qtype = q_elem.get("qtype", "")

            if not question or not answer:
                continue

            records.append(
                {
                    "id": f"{doc_id}_{pid}",
                    "source": source,
                    "url": url,
                    "focus": focus,
                    "semantic_types": sem_types,
                    "question_type": qtype,
                    "question": question,
                    "answer": answer,
                }
            )
    except Exception as e:
        print(f"  Warning: could not parse {xml_path}: {e}")
    return records


def build_dataset(extract_dir: Path) -> list[dict]:
    """Walk all XML files under extract_dir and compile Q&A records."""
    all_records: list[dict] = []
    xml_files = list(extract_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files. Parsing...")

    for xml_path in tqdm(xml_files, desc="Parsing XML"):
        records = parse_xml_file(str(xml_path))
        all_records.extend(records)

    print(f"Total Q&A pairs extracted: {len(all_records)}")
    return all_records


def load_dataset(force_rebuild: bool = False) -> list[dict]:
    """
    Load the processed MedQuAD dataset.
    Downloads and builds it on first run; returns cached JSON thereafter.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if PROCESSED_FILE.exists() and not force_rebuild:
        print("Loading cached dataset...")
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} Q&A pairs from cache.")
        return data

    # Download & parse
    raw_dir = DATA_DIR / "raw"
    extract_dir = download_and_extract(raw_dir)
    records = build_dataset(extract_dir)

    # Save processed file
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved processed dataset to {PROCESSED_FILE}")
    return records


if __name__ == "__main__":
    data = load_dataset()
    df = pd.DataFrame(data)
    print("\nDataset summary:")
    print(df.describe(include="all").to_string())
    print("\nSources:", df["source"].value_counts().to_dict())
    print("\nQuestion types:", df["question_type"].value_counts().to_dict())
