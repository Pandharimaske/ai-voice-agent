"""
agent/graph.py
──────────────
LangGraph pipeline for ARIA — modernized for LangGraph 0.4.x.

Graph topology (ReAct loop):
    START -> stt -> agent <-> tools -> END
                   ^ interrupt() called inside agent_node for HITL
"""

from __future__ import annotations

import re
import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.nodes import agent_node, stt_node, tool_node
from agent.state import AgentState
from config.logging_config import logger
from config.settings import settings

# Mirror of the map in nodes.py — needed here to derive intent from
# interrupt_data without creating a circular import.
_TOOL_TO_INTENT: dict[str, str] = {
    "create_file":    "📄 Create File",
    "write_code":     "💻 Write Code",
    "read_file":      "📖 Read File",
    "summarize_text": "📝 Summarize Text",
    "run_terminal":   "⚡ Run Command",
}

# Regex for raw function-call syntax some models output as plain text
_FUNC_CALL_RE = re.compile(r"<function[/_]")

# Ensure data directory exists
settings.db_path.parent.mkdir(parents=True, exist_ok=True)


# ── Routing helper ────────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    """
    After the agent node runs, decide whether to call tools or finish.
    If the agent made tool calls -> go to 'tools'.
    Otherwise -> END.
    """
    last_msg = state.get("messages", [])[-1] if state.get("messages") else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(checkpointer):
    """
    Build and compile the StateGraph.

    The graph is a clean ReAct loop:
        stt -> agent -> [tools -> agent -> ...] -> END
    HITL is handled by interrupt() inside agent_node - no extra nodes needed.
    """
    builder = StateGraph(AgentState)

    builder.add_node("stt",   stt_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "stt")
    builder.add_edge("stt", "agent")

    builder.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


# ── Singleton (graph is built once per app lifecycle via FastAPI lifespan) ────

_graph: Any = None


def init_graph(checkpointer) -> None:
    """Called once at startup with the fully-initialized async checkpointer."""
    global _graph
    _graph = _build_graph(checkpointer)
    logger.info("LangGraph: Graph compiled and ready.")


def get_graph():
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call init_graph() at startup.")
    return _graph


# ── State serializer (single source of truth) ─────────────────────────────────

async def _build_response(graph, config: dict) -> dict:
    """
    Read the current graph state snapshot and convert it to the JSON
    structure the frontend expects.

    Returns
    -------
    {
        thread_id, transcript, detected_intent, action_taken,
        messages: [{role, content, intent?}],
        output_path, error,
        is_interrupted, interrupt_data
    }
    """
    snapshot = await graph.aget_state(config)
    values   = snapshot.values or {}

    # Reload detected_intent from state (may have been reset for this turn)
    detected_intent = values.get("detected_intent") or ""

    # ── Interrupt detection ───────────────────────────────────────────────────
    # In LG 0.4.x, a pending interrupt shows up as a special task in next steps.
    # The interrupt payload is the dict we passed to interrupt() in agent_node.
    is_interrupted = False
    interrupt_data = None

    for task in (snapshot.tasks or []):
        if task.interrupts:
            is_interrupted = True
            raw = task.interrupts[0].value  # the dict from interrupt({...})
            if isinstance(raw, dict):
                interrupt_data = raw
            break

    # When HITL pauses execution, interrupt() raises before agent_node's return
    # so detected_intent was never written to state. Derive it from the
    # interrupt_data tool_names so the badge still shows the right intent.
    if is_interrupted and not detected_intent and interrupt_data:
        tool_names = interrupt_data.get("tool_names", [])
        intents = [_TOOL_TO_INTENT.get(n, n.replace("_", " ").title()) for n in tool_names]
        if intents:
            detected_intent = " + ".join(intents)

    # ── Message serialisation ─────────────────────────────────────────────────
    all_msgs = values.get("messages", [])

    # Find index of last HumanMessage (intent label belongs to it)
    last_human_idx = -1
    for i, m in enumerate(all_msgs):
        if isinstance(m, HumanMessage):
            last_human_idx = i

    messages: list[dict] = []
    for i, m in enumerate(all_msgs):
        if isinstance(m, HumanMessage):
            messages.append({
                "role":    "user",
                "content": m.content,
                "intent":  detected_intent if i == last_human_idx else (
                    m.additional_kwargs.get("intent") or "💬 General Chat"
                ),
            })
        elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            # Skip if the LLM accidentally emitted raw function-call syntax as
            # plain text (model artifact — Llama does this occasionally).
            if re.search(r'<function[/_]|<function>', m.content):
                continue
            messages.append({"role": "assistant", "content": m.content})
        # ToolMessage (raw tool output) is intentionally hidden from the UI.
        # Users only see the agent's final synthesised answer.

    thread_id = config.get("configurable", {}).get("thread_id", "")

    return {
        "thread_id":       thread_id,
        "transcript":      values.get("transcript", ""),
        "detected_intent": detected_intent,
        "action_taken":    values.get("action_taken"),
        "messages":        messages,
        "output_path":     values.get("output_path"),
        "error":           values.get("error"),
        "is_interrupted":  is_interrupted,
        "interrupt_data":  interrupt_data,
    }


# ── Public pipeline functions ──────────────────────────────────────────────────

async def astream_pipeline(
    audio_path: str,
    thread_id: str,
    chat_history: list,
    output_path: str | None = None,
) -> None:
    """
    Voice input pipeline. STT node will transcribe the audio and add the
    resulting HumanMessage. Yields serialised state dicts as they update.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "audio_path":    audio_path,
        "output_path":   output_path,
        # Reset per-turn fields so last turn's intent/action don't bleed through
        "detected_intent": None,
        "action_taken":    None,
    }

    async for _ in graph.astream(initial, config, stream_mode="values"):
        yield await _build_response(graph, config)


async def astream_pipeline_text(
    text: str,
    thread_id: str,
    chat_history: list,
    output_path: str | None = None,
) -> None:
    """
    Text input pipeline. Injects the HumanMessage directly (skips STT node).
    Yields serialised state dicts as they update.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "messages":        [HumanMessage(content=text)],
        "transcript":      text,
        "output_path":     output_path,
        # Reset per-turn fields so last turn's intent/action don't bleed through
        "detected_intent": None,
        "action_taken":    None,
    }

    async for _ in graph.astream(initial, config, stream_mode="values"):
        yield await _build_response(graph, config)


async def astream_resume_pipeline(thread_id: str, confirmed: bool) -> dict:
    """
    Resume a graph that was paused by interrupt().

    LangGraph 0.4 resume pattern:
      - Command(resume=True)  -> proceed to tool execution
      - Command(resume=False) -> agent_node receives False, logs cancellation, ends
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"HITL Resume: thread={thread_id} confirmed={confirmed}")
    await graph.ainvoke(Command(resume=confirmed), config)
    return await _build_response(graph, config)


async def get_thread_history(thread_id: str) -> dict:
    """Return the full serialised state for a specific thread."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return await _build_response(graph, config)
