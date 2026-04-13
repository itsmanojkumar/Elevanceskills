"""
Unified Chatbot Platform (Router App).

Important:
- Does NOT modify or depend on MedQuAD's existing Streamlit app (`app.py`) behavior.
- Adds new tasks as separate modules under `apps/` and `src_platform/`.

Run:
  streamlit run app_platform.py
"""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

ROOT = Path(__file__).parent

# Make new platform packages importable
import sys
sys.path.insert(0, str(ROOT))

from src_platform.config import LOCAL_SECRETS_FILE, USER_SECRETS_FILE, Settings, load_settings


def _can_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _status_line(label: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    msg = f"{icon} {label}"
    if detail:
        msg += f" — {detail}"
    st.write(msg)


def _key_loaded(key_name: str) -> bool:
    env_val = os.getenv(key_name, "").strip()
    if env_val:
        return True
    # Fallback: check both env files explicitly for better diagnostics.
    for env_file in (ROOT / ".env", ROOT / "Arxiv" / ".env"):
        if env_file.exists():
            v = (dotenv_values(env_file).get(key_name) or "").strip()
            if v:
                return True
    try:
        has_secrets = LOCAL_SECRETS_FILE.exists() or USER_SECRETS_FILE.exists()
        if has_secrets:
            v = (st.secrets.get(key_name) or "").strip()
            return bool(v)
    except Exception:
        return False
    return False


def _render_task_requirements(title: str, api_keys: list[str], dependencies: list[str]) -> None:
    st.markdown(f"**{title}**")
    if api_keys:
        st.markdown("`API keys`")
        for key in api_keys:
            _status_line(f"{key}", _key_loaded(key), "loaded" if _key_loaded(key) else "missing")
    else:
        st.markdown("`API keys`")
        st.write("✅ No API key required")

    st.markdown("`Dependencies`")
    for dep in dependencies:
        _status_line(dep, _can_import(dep))


def _render_setup_help(task_name: str, model_info: list[str], api_keys: list[str], env_file: Path | None = None) -> None:
    st.markdown(f"**Model & API Setup ({task_name})**")
    st.markdown("`Current model/config`")
    for line in model_info:
        st.write(f"- {line}")

    st.markdown("`How to add API keys`")
    if not api_keys:
        st.write("- No API key is required for this task.")
        return

    for key in api_keys:
        st.write(f"- Required key: `{key}`")

    if env_file is not None:
        st.code(
            "\n".join(
                [f"# File: {env_file}", *[f"{k}=your_key_here" for k in api_keys]]
            )
        )

    st.code(
        "\n".join(
            [
                "# Option A: Environment variable (Windows PowerShell)",
                *[f'$env:{k} = "your_key_here"' for k in api_keys],
                "",
                "# Option B: Windows persistent environment variable",
                *[f'setx {k} "your_key_here"' for k in api_keys],
            ]
        )
    )


def _render_all_task_api_setup(root: Path) -> None:
    st.markdown("**All Task API Keys (one place setup)**")
    st.caption("Use this section to configure keys once for new users.")

    st.markdown("`Task -> required API keys`")
    st.write("- Task 1 — Dynamic Knowledge Base: No API key required")
    st.write("- Task 2 — Multimodal (Gemini): `GEMINI_API_KEY`")
    st.write("- Task 3 — Sentiment: No API key required")
    st.write("- Task 4 — Multilingual: No API key required")
    st.write("- Task 5 — Arxiv: depends on backend")
    st.write("  - `ollama`: no API key")
    st.write("  - `groq`: `GROQ_API_KEY`")
    st.write("  - `huggingface`: `HF_API_KEY`")
    st.write("- Task 6 — Medical Chatbot: No API key required")

    st.markdown("`Root .env (for platform tasks)`")
    st.code(
        "\n".join(
            [
                f"# File: {root / '.env'}",
                "GEMINI_API_KEY=your_gemini_key_here",
                "GEMINI_MODEL=gemini-2.0-flash",
                "",
                "# Optional: only if you want to share Arxiv backend keys globally",
                "GROQ_API_KEY=your_groq_key_here",
                "HF_API_KEY=your_hf_key_here",
            ]
        )
    )

    st.markdown("`Arxiv .env (for Task 5)`")
    st.code(
        "\n".join(
            [
                f"# File: {root / 'Arxiv' / '.env'}",
                "LLM_BACKEND=ollama  # or groq / huggingface",
                "OLLAMA_MODEL=llama3.2",
                "GROQ_API_KEY=your_groq_key_here",
                "HF_API_KEY=your_hf_key_here",
                "HF_MODEL=Qwen/Qwen2.5-72B-Instruct",
            ]
        )
    )


def _render_task_debug_panel(page: str, settings: Settings) -> None:
    root = ROOT
    with st.expander("Task Debug (selected task)", expanded=True):
        st.caption("Checks are safe and non-destructive.")

        if page.startswith("Task 1"):
            _render_task_requirements(
                "Task requirements",
                api_keys=[],
                dependencies=["faiss", "langchain", "sentence_transformers"],
            )
            idx_ok = settings.dynkb_index_dir.exists() and any(settings.dynkb_index_dir.iterdir())
            _status_line("Dynamic KB index", idx_ok, str(settings.dynkb_index_dir))
            _status_line("Ingestion state file", settings.dynkb_state_file.exists(), str(settings.dynkb_state_file))
            _render_all_task_api_setup(root)
            return

        if page.startswith("Task 2"):
            _render_task_requirements(
                "Task requirements",
                api_keys=["GEMINI_API_KEY"],
                dependencies=["google.genai"],
            )
            _status_line("Active key used by task", bool(settings.gemini_api_key), "GEMINI_API_KEY")
            _status_line("Configured backend model", True, settings.gemini_model)
            _render_setup_help(
                task_name="Task 2 — Multimodal",
                model_info=[
                    "Backend: Gemini",
                    f"Configured model from env: {settings.gemini_model}",
                    "Frontend dropdown defaults to configured backend model",
                ],
                api_keys=["GEMINI_API_KEY"],
                env_file=root / ".env",
            )
            return

        if page.startswith("Task 3"):
            _render_task_requirements(
                "Task requirements",
                api_keys=[],
                dependencies=["vaderSentiment"],
            )
            return

        if page.startswith("Task 4"):
            _render_task_requirements(
                "Task requirements",
                api_keys=[],
                dependencies=["langdetect", "transformers"],
            )
            return

        if page.startswith("Task 5"):
            arxiv_root = root / "Arxiv"
            arxiv_app = arxiv_root / "app.py"
            llm_backend = os.getenv("LLM_BACKEND", "ollama")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
            if llm_backend == "ollama":
                backend_key = []
                model_line = f"Ollama model: {ollama_model}"
            elif llm_backend == "groq":
                backend_key = ["GROQ_API_KEY"]
                model_line = f"Groq model: {groq_model}"
            else:
                backend_key = ["HF_API_KEY"]
                model_line = f"Hugging Face model: {hf_model}"

            _render_task_requirements(
                "Task requirements",
                api_keys=backend_key,
                dependencies=["arxiv", "faiss", "sentence_transformers"],
            )
            _status_line("Arxiv app file", arxiv_app.exists(), str(arxiv_app))
            _status_line("Embedded server (8505)", _is_port_open(8505), "http://localhost:8505")
            _status_line("Arxiv backend", True, llm_backend)
            if llm_backend == "ollama":
                _status_line("Ollama reachable", _is_port_open(11434), "http://localhost:11434")
            _render_setup_help(
                task_name="Task 5 — Arxiv",
                model_info=[f"Backend: {llm_backend}", model_line],
                api_keys=backend_key,
                env_file=arxiv_root / ".env",
            )
            return

        if page.startswith("Task 6"):
            _render_task_requirements(
                "Task requirements",
                api_keys=[],
                dependencies=["faiss", "spacy", "sentence_transformers"],
            )
            med_app = root / "app.py"
            data_file = root / "data" / "medquad_processed.json"
            faiss_dir = root / "data" / "faiss_index"
            _status_line("Medical app file", med_app.exists(), str(med_app))
            _status_line("Processed data", data_file.exists(), str(data_file))
            _status_line("FAISS index dir", faiss_dir.exists() and any(faiss_dir.iterdir()), str(faiss_dir))
            _status_line("Embedded server (8504)", _is_port_open(8504), "http://localhost:8504")
            _render_setup_help(
                task_name="Task 6 — Medical Chatbot",
                model_info=["Backend: local embeddings + FAISS", "No external LLM/API key required"],
                api_keys=[],
                env_file=root / ".env",
            )
            return


def _render_settings_panel(settings: Settings) -> None:
    st.markdown("### Settings / Keys")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Detected environment**")
        st.code(
            "\n".join(
                [
                    f"PYTHON: {os.environ.get('PYTHONIOENCODING', '(default)')}",
                    f"WORKDIR: {ROOT}",
                ]
            )
        )

    with cols[1]:
        st.markdown("**Key status (fast checks)**")
        checks = settings.key_status()
        for k, ok in checks.items():
            st.write(f"{'✅' if ok else '❌'} `{k}`")
        if not all(checks.values()):
            st.caption("Add missing keys in `.env` or environment variables.")

    st.markdown("**Dependency status (fast checks)**")
    deps = settings.dependency_status()
    for k, ok in deps.items():
        st.write(f"{'✅' if ok else '❌'} {k}")

    with st.expander("Show effective settings (safe)", expanded=False):
        st.json(settings.to_safe_dict())


def main() -> None:
    st.set_page_config(page_title="Chatbot Platform", layout="wide")

    settings = load_settings()
    nav_options = [
        "Task 1 — Dynamic Knowledge Base",
        "Task 2 — Multimodal (Gemini)",
        "Task 3 — Sentiment",
        "Task 4 — Multilingual",
        "Task 5 — Arxiv",
        "Task 6 — Medical Chatbot",
    ]

    if "nav_page" not in st.session_state or st.session_state.get("nav_page") not in nav_options:
        st.session_state["nav_page"] = nav_options[0]

    st.title("Unified Chatbot Platform")
    st.caption("Modular tasks in separate files. MedQuAD remains unchanged.")

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            nav_options,
            key="nav_page",
        )
        st.markdown("---")
        _render_settings_panel(settings)

    _render_task_debug_panel(page, settings)

    if page.startswith("Task 1"):
        from apps.dynkb_app import render_dynkb_app
        render_dynkb_app(settings=settings)
        return

    if page.startswith("Task 2"):
        from apps.multimodal_app import render_multimodal_app
        render_multimodal_app(settings=settings)
        return

    if page.startswith("Task 3"):
        from apps.sentiment_app import render_sentiment_app
        render_sentiment_app(settings=settings)
        return

    if page.startswith("Task 4"):
        from apps.multilingual_app import render_multilingual_app
        render_multilingual_app(settings=settings)
        return

    if page.startswith("Task 5"):
        from apps.arxiv_app import render_arxiv_app
        render_arxiv_app(settings=settings)
        return

    if page.startswith("Task 6"):
        from apps.medical_chatbot_app import render_medical_chatbot_app
        render_medical_chatbot_app(settings=settings)
        return


if __name__ == "__main__":
    main()

