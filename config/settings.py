"""
config/settings.py
──────────────────
Loads config.yaml into validated Pydantic models.
Import `settings` anywhere in the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Sub-config models ──────────────────────────────────────────────────────────

class STTConfig(BaseModel):
    provider: str = "groq"
    model: str = "whisper-large-v3"


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1


class OutputConfig(BaseModel):
    folder: str = "./output"


class HITLConfig(BaseModel):
    enabled: bool = True
    require_confirmation_for: List[str] = ["create_file", "write_code"]


class MemoryConfig(BaseModel):
    db_path: str = "./data/checkpoints.db"
    session_history_limit: int = 20


# ── Root settings model ────────────────────────────────────────────────────────

class Settings(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @property
    def output_path(self) -> Path:
        p = Path(self.output.folder)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        p = Path(self.memory.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def api_key_for(self, provider: str) -> str:
        """Return the correct API key env var for the given provider."""
        mapping = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = mapping.get(provider)
        if not env_var:
            raise ValueError(f"Unknown provider: {provider}")
        key = os.getenv(env_var)
        if not key:
            raise EnvironmentError(
                f"Missing API key for provider '{provider}'. "
                f"Set {env_var} in your .env file."
            )
        return key


# ── Singleton loader ───────────────────────────────────────────────────────────

def _load() -> Settings:
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()


settings: Settings = _load()
