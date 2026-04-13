"""
LLM handler – returns LangChain BaseChatModel instances.

HuggingFace strategy
────────────────────
• Uses huggingface_hub.InferenceClient.chat_completion() 
• Automatically uses the new https://router.huggingface.co
• Supports 'conversational' task (required for Llama 3+ models on free providers)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from src.config import (
    LLM_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    GROQ_API_KEY,
    GROQ_MODEL,
    HF_API_KEY,
    HF_MODEL,
)

logger = logging.getLogger(__name__)

# ── Custom HF Chat Model using the new Router + Chat Completion ──────────

class HuggingFaceRouterChatLLM(BaseChatModel):
    """
    LangChain BaseChatModel that uses the HF Router Chat Completion API.
    This works with the new provider system (Groq, Together, etc.).
    """

    model: str = Field(default=HF_MODEL)
    api_key: str = Field(default="")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "huggingface-router-chat"

    def _to_hf_messages(self, messages: List[BaseMessage]) -> list[dict]:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        return [
            {"role": role_map.get(m.type, "user"), "content": m.content}
            for m in messages
        ]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=self.api_key)
        
        # chat_completion maps to the 'conversational' task on the router
        response = client.chat_completion(
            messages=self._to_hf_messages(messages),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=self.api_key)
        
        for chunk in client.chat_completion(
            messages=self._to_hf_messages(messages),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


# ── Backend builders ──────────────────────────────────────────────────────

def _build_ollama(model: str) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.7)
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.7)


def _build_groq(model: str) -> BaseChatModel:
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model,
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
    )


def _build_huggingface(model: str) -> BaseChatModel:
    """Uses the new HF Router Chat API."""
    return HuggingFaceRouterChatLLM(
        model=model,
        api_key=HF_API_KEY,
        max_tokens=1024,
        temperature=0.7,
    )


# ── Availability checks ───────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _groq_available() -> bool:
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def _hf_available() -> bool:
    return bool(HF_API_KEY and HF_API_KEY.strip())


# ── Public factory ────────────────────────────────────────────────────────

def get_llm(
    backend: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """Return a LangChain BaseChatModel for the requested backend."""
    backend = (backend or LLM_BACKEND).lower()

    if backend == "huggingface":
        if not _hf_available():
            raise RuntimeError("HF_API_KEY is missing in your .env file.")
        resolved_model = model or HF_MODEL
        llm = _build_huggingface(resolved_model)
        logger.info("LLM backend: huggingface (router) / %s", resolved_model)
        return llm

    if backend == "groq":
        if not _groq_available():
            raise RuntimeError("GROQ_API_KEY is missing in your .env file.")
        resolved_model = model or GROQ_MODEL
        llm = _build_groq(resolved_model)
        logger.info("LLM backend: groq / %s", resolved_model)
        return llm

    if backend == "ollama":
        if not _ollama_available():
            raise RuntimeError(f"Ollama not found at {OLLAMA_BASE_URL}.")
        resolved_model = model or OLLAMA_MODEL
        llm = _build_ollama(resolved_model)
        logger.info("LLM backend: ollama / %s", resolved_model)
        return llm

    raise RuntimeError(f"Unknown LLM_BACKEND '{backend}'")


def list_ollama_models() -> list[str]:
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []
