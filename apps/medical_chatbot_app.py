from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src_platform.config import Settings


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_medical_app(root: Path, port: int) -> None:
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
    subprocess.Popen(cmd, cwd=str(root))


def render_medical_chatbot_app(settings: Settings) -> None:
    _ = settings
    st.header("Task 6 — Medical Chatbot")

    st.caption("Full-screen interactive MedQuAD workspace")

    root = Path(__file__).resolve().parents[1]
    med_app = root / "app.py"
    if not med_app.exists():
        st.error("Medical chatbot app not found at `app.py`.")
        return

    # Use a dedicated port to avoid stale/old embedded process conflicts.
    port = 8514
    target_url = f"http://localhost:{port}"

    if not _is_port_open(port):
        _start_medical_app(root, port)
        st.info("Starting Medical Chatbot... reload this page in a few seconds.")
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

