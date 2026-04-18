"""
agent/llm.py
────────────
Factory for LangChain chat models.
Supports Groq, OpenAI, Anthropic, and Local (Ollama).
"""

from __future__ import annotations

import os
from typing import Optional
from langchain_core.language_models import BaseChatModel

def get_llm(
    provider: str,
    model: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.
    One-line swap: change provider in config.yaml.
    """
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )

    elif provider in ["ollama", "local"]:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Available: groq | openai | anthropic | ollama"
        )
