from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    raw: dict


@dataclass(frozen=True)
class GeminiImageResponse:
    image_bytes: bytes
    mime_type: str
    raw: dict


class GeminiClient:
    """
    Gemini API client (Google AI Studio key).

    Uses the official `google-genai` package if installed.
    This keeps implementation small and avoids hand-rolling REST.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

        try:
            from google import genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: google-genai. Install with `pip install google-genai`."
            ) from e

        self._genai = genai
        # Avoid ambiguous key selection when both GOOGLE_API_KEY and GEMINI_API_KEY exist.
        os.environ["GOOGLE_API_KEY"] = api_key
        self._client = genai.Client(api_key=api_key)

    def generate_text(self, prompt: str) -> GeminiResponse:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        txt = getattr(resp, "text", "") or ""
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {"text": txt}
        return GeminiResponse(text=txt, raw=raw)

    def generate_with_image(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> GeminiResponse:
        # google-genai supports passing bytes as "Part" via genai.types
        types = self._genai.types
        part_img = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[prompt, part_img],
        )
        txt = getattr(resp, "text", "") or ""
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {"text": txt}
        return GeminiResponse(text=txt, raw=raw)

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = "imagen-3.0-generate-002",
    ) -> GeminiImageResponse:
        """
        Generate an image from text prompt using Gemini/Imagen.
        """
        types = self._genai.types
        chosen_model = model
        try:
            resp = self._client.models.generate_images(
                model=chosen_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
        except Exception:
            # Fallback: discover an available image model for this API key.
            discovered = self.list_image_models()
            if not discovered:
                raise RuntimeError(
                    "No image-generation model is available for this API key. "
                    "Use list models or enable an image model in Google AI Studio."
                )
            chosen_model = discovered[0]
            resp = self._client.models.generate_images(
                model=chosen_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        raw["resolved_model"] = chosen_model

        generated = getattr(resp, "generated_images", None) or []
        if not generated:
            raise RuntimeError("No image returned from Gemini image generation.")

        first = generated[0]
        image_obj = getattr(first, "image", None)
        if image_obj is None:
            raise RuntimeError("Gemini image generation response did not include image payload.")

        image_bytes = getattr(image_obj, "image_bytes", None)
        if not image_bytes:
            raise RuntimeError("Gemini image generation returned empty image bytes.")

        return GeminiImageResponse(
            image_bytes=image_bytes,
            mime_type="image/png",
            raw=raw,
        )

    def list_image_models(self) -> list[str]:
        """
        Return image-generation model names visible to this API key.
        """
        names: list[str] = []
        try:
            models = self._client.models.list()
        except Exception:
            return names

        for m in models:
            name = getattr(m, "name", "") or ""
            methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            method_text = " ".join(methods).lower() if methods else ""
            name_l = name.lower()
            if "image" in name_l or "imagen" in name_l or "generateimages" in method_text:
                if name:
                    names.append(name)
        # Stable unique order
        seen = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

