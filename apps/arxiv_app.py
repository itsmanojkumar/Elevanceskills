from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import dotenv_values

from src_platform.config import Settings


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_arxiv_app(arxiv_root: Path, port: int) -> None:
    child_env = os.environ.copy()
    arxiv_env_path = arxiv_root / ".env"
    if arxiv_env_path.exists():
        for k, v in dotenv_values(arxiv_env_path).items():
            if v is not None and str(v).strip():
                child_env[k] = str(v).strip()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    subprocess.Popen(cmd, cwd=str(arxiv_root), env=child_env)


def render_arxiv_app(settings: Settings) -> None:
    _ = settings
    st.header("Task 5 — Arxiv")

    st.caption("Full-screen interactive Arxiv workspace")

    root = Path(__file__).resolve().parents[1]
    arxiv_root = root / "Arxiv"
    arxiv_app = arxiv_root / "app.py"

    if not arxiv_app.exists():
        st.error("Arxiv app not found at `Arxiv/app.py`.")
        return

    port = 8505
    target_url = f"http://localhost:{port}"

    if not _is_port_open(port):
        _start_arxiv_app(arxiv_root, port)
        st.info("Starting Arxiv... reload this page in a few seconds.")
        return

    st.markdown(
        """
        <style>
        .embed-shell {
            border: 1px solid #243047;
            border-radius: 14px;
            padding: 10px;
            background: linear-gradient(180deg, #0b1220 0%, #0b1324 100%);
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="embed-shell">', unsafe_allow_html=True)
    components.iframe(target_url, height=1200, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)

