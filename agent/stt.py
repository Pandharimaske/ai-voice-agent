"""
agent/stt.py
────────────
Speech-to-Text abstraction layer.
Provider is set in config.yaml → stt.provider.

Hardware note (documented in README):
  ARIA uses API-based STT because local Whisper requires significant
  GPU/RAM resources. Groq's Whisper API is free-tier, extremely fast
  (~300x realtime), and the transcription quality is identical to
  running the model locally.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class BaseSTT(ABC):
    """Abstract STT interface — swap providers without touching the pipeline."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file at `audio_path` → plain text string."""
        ...


# ── Groq (Whisper via API) ─────────────────────────────────────────────────────

class GroqSTT(BaseSTT):
    """
    Uses Groq's hosted Whisper endpoint.
    Model options: whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en
    """

    SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

    def __init__(self, model: str = "whisper-large-v3", api_key: Optional[str] = None):
        from groq import Groq
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = model

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=(path.name, f.read()),
                model=self.model,
                response_format="text",
                language="en",
            )
        # Groq returns a string directly when response_format="text"
        return str(transcription).strip()


# ── OpenAI (Whisper via API) ───────────────────────────────────────────────────

class OpenAISTT(BaseSTT):
    """
    Uses OpenAI's hosted Whisper endpoint.
    Model options: whisper-1
    """

    def __init__(self, model: str = "whisper-1", api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
            )
        return transcription.text.strip()


# ── Factory ────────────────────────────────────────────────────────────────────

from typing import Optional


def get_stt(provider: str, model: str, api_key: Optional[str] = None) -> BaseSTT:
    """
    Factory function — returns the correct STT implementation.
    Called by the STT node; provider/model come from config.yaml.
    """
    registry = {
        "groq": GroqSTT,
        "openai": OpenAISTT,
    }
    cls = registry.get(provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown STT provider: '{provider}'. "
            f"Available: {list(registry.keys())}"
        )
    return cls(model=model, api_key=api_key)
