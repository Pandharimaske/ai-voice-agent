"""
agent/state.py
──────────────
AgentState definition for the ARIA Voice Agent.

Uses LangGraph's built-in `add_messages` reducer — this handles:
  - Appending new messages to history
  - ID-based deduplication (no duplicate messages on checkpoint resume)
  - Proper serialisation for the SQLite checkpointer
"""

from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Full pipeline state for the ARIA Tool-Calling Agent.

    messages        Conversation history. `add_messages` ensures new messages
                    are appended and deduped by ID on checkpoint restore.
    audio_path      Temp path of the uploaded audio file (voice input only).
    transcript      Plain-text output from the STT step.
    detected_intent Human-readable intent label for the UI.
    action_taken    Concrete action description for the UI.
    output_path     User-selected output folder for file operations.
    error           Any pipeline error message surfaced to the UI.
    """

    messages: Annotated[List, add_messages]   # required; reducer handles merging
    audio_path: Optional[str]
    transcript: Optional[str]
    detected_intent: Optional[str]
    action_taken: Optional[str]
    output_path: Optional[str]
    error: Optional[str]
