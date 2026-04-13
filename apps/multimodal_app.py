from __future__ import annotations

import streamlit as st

from src_platform.config import Settings
from src_platform.llm.gemini_client import GeminiClient


def render_multimodal_app(settings: Settings) -> None:
    st.header("Task 2 — Multimodal (Gemini)")

    st.markdown(
        "Image input + text response using Gemini (Google AI Studio key).\n\n"
        "This page also does **fast configuration checks** so missing keys fail early."
    )

    if settings.gemini_api_key:
        st.success("GEMINI_API_KEY is set.")
    else:
        st.error("GEMINI_API_KEY is missing.")
        st.code("setx GEMINI_API_KEY \"your_key_here\"")
        st.caption("Or add it to `.env` as `GEMINI_API_KEY=...`.")
        return

    st.subheader("Multimodal chat")
    model_options = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    configured_model = settings.gemini_model
    if configured_model not in model_options:
        model_options = [configured_model] + model_options
    default_idx = model_options.index(configured_model) if configured_model in model_options else 0
    model = st.selectbox("Model", options=model_options, index=default_idx, key="task2_model")
    if model != configured_model:
        st.warning(
            f"Model override in UI: configured `{configured_model}`, current `{model}`. "
            "Use configured model to avoid compatibility drift."
        )
    else:
        st.caption(f"Using configured model from backend env: `{configured_model}`")
    tab_understand, tab_generate = st.tabs(["Understand Image + Text", "Generate Image from Text"])

    with tab_understand:
        prompt = st.text_area("Your question", placeholder="Describe what you see and ask a question…", height=120)
        image = st.file_uploader("Upload an image (png/jpg/webp)", type=["png", "jpg", "jpeg", "webp"])

        if st.button("Run Gemini", use_container_width=True):
            if not prompt.strip():
                st.warning("Enter a question.")
                return

            client = GeminiClient(api_key=settings.gemini_api_key, model=model)
            with st.spinner("Calling Gemini..."):
                if image is not None:
                    resp = client.generate_with_image(
                        prompt.strip(),
                        image_bytes=image.getvalue(),
                        mime_type=image.type or "image/png",
                    )
                else:
                    resp = client.generate_text(prompt.strip())

            st.markdown("### Response")
            st.write(resp.text or "(empty)")

            with st.expander("Raw response (debug)", expanded=False):
                st.json(resp.raw)

    with tab_generate:
        st.caption("Generate an image from text prompt using Gemini image generation.")
        img_prompt = st.text_area(
            "Image generation prompt",
            placeholder="e.g. A clean medical infographic about asthma triggers, flat design",
            height=110,
            key="task2_img_prompt",
        )
        image_model = st.text_input(
            "Image model",
            value="imagen-3.0-generate-002",
            help="Use a Gemini/Imagen image model available to your API key.",
        )

        if st.button("Generate image", use_container_width=True):
            if not img_prompt.strip():
                st.warning("Enter an image prompt.")
                return

            client = GeminiClient(api_key=settings.gemini_api_key, model=model)
            try:
                with st.spinner("Generating image..."):
                    img_resp = client.generate_image(img_prompt.strip(), model=image_model.strip() or "imagen-3.0-generate-002")
                st.image(img_resp.image_bytes, caption="Generated image", use_container_width=True)
                with st.expander("Raw image response (debug)", expanded=False):
                    st.json(img_resp.raw)
            except Exception as exc:
                st.error(
                    "Image generation failed. Ensure your key has access to the selected image model.\n\n"
                    f"Details: {exc}"
                )

