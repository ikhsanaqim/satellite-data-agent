from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv


class SimpleLLM(Protocol):
    def invoke(self, prompt: str) -> str:
        """Return a model response for a prompt."""


class MockLLM:
    """Deterministic fallback for offline demos and tests."""

    def invoke(self, prompt: str) -> str:
        if "executive summary" in prompt.lower():
            return (
                "Telemetry shows mostly healthy satellite/IoT operations, with "
                "localized degradation where latency and packet loss rise together."
            )
        return "No external LLM configured; deterministic analysis was used."


def build_llm() -> SimpleLLM:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(model=model, temperature=0.2)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        api_key = os.getenv("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=0.2,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    return MockLLM()
