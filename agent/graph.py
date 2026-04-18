"""
agent/graph.py
──────────────
Assembles the LangGraph pipeline and wires up the SQLite checkpointer.

Pipeline topology:
  START → stt → intent → [should_confirm?]
                              ├── YES → hitl (interrupt) → tool → END
                              └── NO  → tool → END

The SQLite checkpointer persists state between the initial run and the
HITL resume call, keyed by thread_id (one per pipeline invocation).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.nodes import hitl_node, intent_node, stt_node, tool_node
from agent.state import AgentState
from config.settings import settings


# ── Routing logic ─────────────────────────────────────────────────────────────

def _should_confirm(state: AgentState) -> str:
    """
    Conditional edge: after intent classification, decide whether HITL is needed.
    Routes to 'hitl' if confirmation is required and HITL is enabled in config.
    """
    intent_result = state.get("intent_result") or {}
    requires = intent_result.get("requires_confirmation", False)

    # Check that the intent type is actually in the require list
    actions = intent_result.get("actions", [])
    any_needs_confirm = any(
        a.get("intent") in settings.hitl.require_confirmation_for
        for a in actions
    )

    if settings.hitl.enabled and requires and any_needs_confirm:
        return "hitl"
    return "tools"


def _after_hitl(state: AgentState) -> str:
    """
    Conditional edge: after HITL node, proceed to tools or END based on user choice.
    """
    if state.get("confirmed", False):
        return "tools"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────

# Module-level graph singleton (built once, reused across Gradio calls)
_graph = None
_db_conn: Optional[sqlite3.Connection] = None


def get_graph():
    """
    Return the compiled LangGraph (singleton).
    Safe to call multiple times — builds only on first call.
    """
    global _graph, _db_conn

    if _graph is not None:
        return _graph

    # SQLite checkpointer
    db_path = str(settings.db_path)
    _db_conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(_db_conn)

    # Graph definition
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("stt", stt_node)
    builder.add_node("intent", intent_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("tools", tool_node)

    # Edges
    builder.add_edge(START, "stt")
    builder.add_edge("stt", "intent")
    builder.add_conditional_edges(
        "intent",
        _should_confirm,
        {"hitl": "hitl", "tools": "tools"},
    )
    builder.add_conditional_edges(
        "hitl",
        _after_hitl,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", END)

    _graph = builder.compile(checkpointer=checkpointer)
    return _graph


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_pipeline(audio_path: str, thread_id: str, chat_history: list, output_path: str = None) -> dict:
    """
    Start a fresh pipeline run for the given audio file.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "audio_path": audio_path,
        "transcript": "",
        "intent_result": None,
        "confirmed": None,
        "tool_results": [],
        "messages": chat_history,
        "output_path": output_path,
        "error": None,
    }

    graph.invoke(initial_state, config)
    return _read_state(graph, config)


def run_pipeline_text(text: str, thread_id: str, chat_history: list, output_path: str = None) -> dict:
    """
    Start a fresh pipeline run using direct text input (skips STT).
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "audio_path": "",
        "transcript": text,
        "intent_result": None,
        "confirmed": None,
        "tool_results": [],
        "messages": chat_history,
        "output_path": output_path,
        "error": None,
    }

    graph.invoke(initial_state, config)
    return _read_state(graph, config)


def resume_pipeline(thread_id: str, confirmed: bool) -> dict:
    """
    Resume a paused (HITL-interrupted) pipeline.
    Pass confirmed=True to proceed, False to cancel.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(Command(resume={"confirmed": confirmed}), config)
    return _read_state(graph, config)


def get_thread_history(thread_id: str) -> dict:
    """Retrieve the current state of a thread without executing anything."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return _read_state(graph, config)


def _read_state(graph, config: dict) -> dict:
    """Read current state snapshot and check for pending interrupts."""
    snapshot = graph.get_state(config)
    values = snapshot.values

    # Detect interrupt
    is_interrupted = bool(snapshot.next)
    interrupt_data = None
    if is_interrupted and snapshot.tasks:
        task = snapshot.tasks[0]
        if hasattr(task, "interrupts") and task.interrupts:
            interrupt_data = task.interrupts[0].value

    return {
        "transcript": values.get("transcript", ""),
        "intent_result": values.get("intent_result"),
        "tool_results": values.get("tool_results", []),
        "messages": values.get("messages", []),
        "output_path": values.get("output_path"),
        "error": values.get("error"),
        "confirmed": values.get("confirmed"),
        "is_interrupted": is_interrupted,
        "interrupt_data": interrupt_data,
    }
