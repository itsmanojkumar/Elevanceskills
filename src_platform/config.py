from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent.parent
DATA_PLATFORM_DIR = ROOT / "data_platform"
DYNKB_INDEX_DIR = DATA_PLATFORM_DIR / "dynamic_kb_index"
DYNKB_STATE_FILE = DATA_PLATFORM_DIR / "ingestion_state.json"
LOCAL_SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"
USER_SECRETS_FILE = Path.home() / ".streamlit" / "secrets.toml"
ARXIV_ENV_FILE = ROOT / "Arxiv" / ".env"

if load_dotenv is not None:
    # Load root env first, then Arxiv env for task-specific keys.
    # Do not override already set environment variables.
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ARXIV_ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    # Task 2 (Gemini)
    gemini_api_key: str | None
    gemini_model: str

    # Storage
    data_platform_dir: Path
    dynkb_index_dir: Path
    dynkb_state_file: Path

    def key_status(self) -> dict[str, bool]:
        return {
            "GEMINI_API_KEY": bool(self.gemini_api_key),
            "GEMINI_MODEL": bool(self.gemini_model),
        }

    def dependency_status(self) -> dict[str, bool]:
        """
        Fast import checks shown in the UI for quicker debugging.
        """
        def _can_import(mod: str) -> bool:
            try:
                __import__(mod)
                return True
            except Exception:
                return False

        return {
            "google.genai (google-genai)": _can_import("google.genai"),
            "vaderSentiment": _can_import("vaderSentiment"),
            "langdetect": _can_import("langdetect"),
            "transformers": _can_import("transformers"),
        }

    def to_safe_dict(self) -> dict:
        # Never print raw keys
        return {
            "gemini_api_key": "***set***" if self.gemini_api_key else None,
            "gemini_model": self.gemini_model,
            "data_platform_dir": str(self.data_platform_dir),
            "dynkb_index_dir": str(self.dynkb_index_dir),
            "dynkb_state_file": str(self.dynkb_state_file),
            "dependency_status": self.dependency_status(),
        }


def _get_env(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def load_settings() -> Settings:
    # Support Streamlit secrets if present (without requiring it).
    gemini_key = _get_env("GEMINI_API_KEY")
    gemini_model = _get_env("GEMINI_MODEL") or "gemini-2.0-flash"
    try:
        import streamlit as st
        has_secrets_file = LOCAL_SECRETS_FILE.exists() or USER_SECRETS_FILE.exists()
        if not gemini_key and has_secrets_file:
            gemini_key = (st.secrets.get("GEMINI_API_KEY") or "").strip() or None
    except Exception:
        pass

    return Settings(
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        data_platform_dir=DATA_PLATFORM_DIR,
        dynkb_index_dir=DYNKB_INDEX_DIR,
        dynkb_state_file=DYNKB_STATE_FILE,
    )

