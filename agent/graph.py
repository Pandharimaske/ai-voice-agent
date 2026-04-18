"""
agent/graph.py
──────────────
Assembles the LangGraph pipeline and wires up the SQLite checkpointer.

Pipeline topology:
  START → stt → intent → [should_confirm?]
                              ├── YES → hitl (interrupt) → [confirmed?]
                              │                                ├── YES → tools → END
                              │                                └── NO  → END
                              └── NO  → tools → END

Two checkpointers are maintained:
  - _sync_conn  / SqliteSaver      — for sync invoke() calls (used by /api/process)
  - _async_conn / AsyncSqliteSaver — for astream() calls   (used by /api/process_stream)
Both point to the same SQLite file so thread state is shared.
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


# ── Routing helpers ───────────────────────────────────────────────────────────

def _should_confirm(state: AgentState) -> str:
    intent_result = state.get("intent_result") or {}
    actions = intent_result.get("actions", [])
    any_needs_confirm = any(
        a.get("intent") in settings.hitl.require_confirmation_for
        for a in actions
    )
    if settings.hitl.enabled and intent_result.get("requires_confirmation") and any_needs_confirm:
        return "hitl"
    return "tools"


def _after_hitl(state: AgentState) -> str:
    return "tools" if state.get("confirmed", False) else END


# ── Singletons ────────────────────────────────────────────────────────────────

_sync_graph = None
_async_graph = None
_sync_conn: Optional[sqlite3.Connection] = None


def _build_graph(checkpointer):
    """Build and compile the StateGraph with the given checkpointer."""
    builder = StateGraph(AgentState)
    builder.add_node("stt",    stt_node)
    builder.add_node("intent", intent_node)
    builder.add_node("hitl",   hitl_node)
    builder.add_node("tools",  tool_node)

    builder.add_edge(START, "stt")
    builder.add_edge("stt",   "intent")
    builder.add_conditional_edges("intent", _should_confirm, {"hitl": "hitl", "tools": "tools"})
    builder.add_conditional_edges("hitl",   _after_hitl,    {"tools": "tools", END: END})
    builder.add_edge("tools", END)

    return builder.compile(checkpointer=checkpointer)


def get_graph():
    """Sync graph singleton — used for invoke() calls."""
    global _sync_graph, _sync_conn
    if _sync_graph is None:
        db_path = str(settings.db_path)
        _sync_conn = sqlite3.connect(db_path, check_same_thread=False)
        _sync_graph = _build_graph(SqliteSaver(_sync_conn))
    return _sync_graph


async def get_async_graph():
    """
    Async graph singleton — used for astream() calls.
    Falls back to sync SqliteSaver if aiosqlite is not installed.
    """
    global _async_graph
    if _async_graph is None:
        db_path = str(settings.db_path)
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            import aiosqlite
            conn = await aiosqlite.connect(db_path)
            checkpointer = AsyncSqliteSaver(conn)
        except (ImportError, Exception):
            # Graceful fallback: use sync checkpointer (safe for low concurrency)
            _sync_conn2 = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(_sync_conn2)
        _async_graph = _build_graph(checkpointer)
    return _async_graph


# ── State reader ──────────────────────────────────────────────────────────────

def _read_state(graph, config: dict) -> dict:
    snapshot = graph.get_state(config)
    values = snapshot.values

    is_interrupted = bool(snapshot.next)
    interrupt_data = None
    if is_interrupted and snapshot.tasks:
        task = snapshot.tasks[0]
        if hasattr(task, "interrupts") and task.interrupts:
            interrupt_data = task.interrupts[0].value

    return {
        "transcript":    values.get("transcript", ""),
        "intent_result": values.get("intent_result"),
        "tool_results":  values.get("tool_results", []),
        "messages":      values.get("messages", []),
        "output_path":   values.get("output_path"),
        "error":         values.get("error"),
        "confirmed":     values.get("confirmed"),
        "is_interrupted": is_interrupted,
        "interrupt_data": interrupt_data,
    }


# ── Sync pipeline (used by /api/process and /api/confirm) ────────────────────

def run_pipeline(audio_path: str, thread_id: str, chat_history: list, output_path: str = None) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "audio_path":   audio_path,
        "transcript":   "",
        "intent_result": None,
        "confirmed":    None,
        "tool_results": [],
        "messages":     chat_history,
        "output_path":  output_path,
        "error":        None,
    }
    graph.invoke(initial, config)
    return _read_state(graph, config)


def run_pipeline_text(text: str, thread_id: str, chat_history: list, output_path: str = None) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "audio_path":   "",
        "transcript":   text,
        "intent_result": None,
        "confirmed":    None,
        "tool_results": [],
        "messages":     chat_history,
        "output_path":  output_path,
        "error":        None,
    }
    graph.invoke(initial, config)
    return _read_state(graph, config)


def resume_pipeline(thread_id: str, confirmed: bool) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume={"confirmed": confirmed}), config)
    return _read_state(graph, config)


def get_thread_history(thread_id: str) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return _read_state(graph, config)


# ── Async streaming pipeline (used by /api/process_stream) ───────────────────

async def astream_pipeline(audio_path: str, thread_id: str, chat_history: list, output_path: str = None):
    graph = await get_async_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "audio_path":   audio_path,
        "transcript":   "",
        "intent_result": None,
        "confirmed":    None,
        "tool_results": [],
        "messages":     chat_history,
        "output_path":  output_path,
        "error":        None,
    }
    async for event in graph.astream(initial, config, stream_mode="values"):
        yield event


async def astream_pipeline_text(text: str, thread_id: str, chat_history: list, output_path: str = None):
    graph = await get_async_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "audio_path":   "",
        "transcript":   text,
        "intent_result": None,
        "confirmed":    None,
        "tool_results": [],
        "messages":     chat_history,
        "output_path":  output_path,
        "error":        None,
    }
    async for event in graph.astream(initial, config, stream_mode="values"):
        yield event


async def astream_resume_pipeline(thread_id: str, confirmed: bool):
    graph = await get_async_graph()
    config = {"configurable": {"thread_id": thread_id}}
    async for event in graph.astream(Command(resume={"confirmed": confirmed}), config, stream_mode="values"):
        yield event
