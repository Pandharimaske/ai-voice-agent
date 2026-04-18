"""
agent/state.py
──────────────
Pydantic models for structured LLM output + LangGraph AgentState TypedDict.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Pydantic models (structured LLM output) ───────────────────────────────────

class IntentAction(BaseModel):
    """A single atomic action the agent should take."""

    intent: Literal["create_file", "write_code", "summarize", "read_file", "run_terminal", "general_chat"] = Field(
        ...,
        description=(
            "create_file: create an empty or text file. "
            "write_code: generate and save code to a file. "
            "summarize: summarize provided text. "
            "read_file: read the content of an existing file. "
            "run_terminal: execute a shell command. "
            "general_chat: answer a question or have a conversation."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description="Target filename including extension (e.g. retry.py, notes.txt). "
                    "Required for create_file, write_code, and read_file. Null otherwise.",
    )
    command: Optional[str] = Field(
        None,
        description="The exact shell command to run for run_terminal. Null otherwise.",
    )
    description: str = Field(
        ...,
        description="Clear description of what to do. For write_code: describe the code. "
                    "For summarize: the text to summarize. For chat: the question/message.",
    )
    content: Optional[str] = Field(
        None,
        description="Literal content to write for create_file, or text to summarize for summarize. "
                    "Leave null for write_code, read_file, run_terminal and general_chat.",
    )


class IntentResult(BaseModel):
    """Classified intent(s) extracted from transcribed speech."""

    actions: List[IntentAction] = Field(
        ...,
        description="One or more actions to perform. Support compound commands by listing multiple.",
    )
    requires_confirmation: bool = Field(
        ...,
        description="True if ANY action writes to the filesystem (create_file, write_code) or runs a command (run_terminal). "
                    "False for read_file, summarize and general_chat.",
    )


# ── LangGraph AgentState ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Full pipeline state passed between LangGraph nodes."""

    audio_path: str                        # Path to the audio file
    transcript: str                        # STT output
    intent_result: Optional[dict]          # Serialized IntentResult
    confirmed: Optional[bool]             # HITL confirmation result
    tool_results: List[dict]               # [{action: str, result: str}, ...]
    messages: List[dict]                   # Session chat history [{role, content}, ...]
    output_path: Optional[str]            # Custom path for file operations
    error: Optional[str]                   # Error message if any node fails
