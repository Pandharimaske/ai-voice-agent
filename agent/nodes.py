"""
agent/nodes.py
──────────────
Graph nodes for the ARIA Voice Agent.

Key design decisions (LangGraph 0.4.x):
  - `interrupt()` from langgraph.types is called DIRECTLY inside agent_node
    when a destructive tool is about to be called. This replaces the old
    fake `hitl_interrupt` passthrough node.
  - LLM is created once via @lru_cache — thread-safe, no global mutation.
  - Tool binding (`bind_tools`) happens once at LLM creation time.
  - Intent accumulation: every unique tool intent across ALL ReAct loop passes
    is merged into a combined badge (e.g. "📖 Read File + 💻 Write Code").
    "💬 General Chat" is only shown when zero tools were called this turn.
"""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.llm import get_llm
from agent.state import AgentState
from agent.stt import get_stt
from agent.tools import ALL_TOOLS
from config.logging_config import logger
from config.settings import settings


# ── Tool registry ─────────────────────────────────────────────────────────────

# Tools that write to disk or run commands — require HITL confirmation.
UNSAFE_TOOLS = {"create_file", "write_code", "run_terminal"}

tool_node = ToolNode(ALL_TOOLS)


# ── Intent classification map ─────────────────────────────────────────────────

_TOOL_TO_INTENT: dict[str, str] = {
    "create_file":    "📄 Create File",
    "write_code":     "💻 Write Code",
    "read_file":      "📖 Read File",
    "summarize_text": "📝 Summarize Text",
    "run_terminal":   "⚡ Run Command",
}


# ── LLM singleton (created once, cached) ─────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_llm_with_tools():
    """
    Create and cache the LLM with tools bound.
    lru_cache ensures this runs exactly once — safe for async multi-call patterns.
    """
    api_key = settings.api_key_for(settings.llm.provider)
    llm = get_llm(
        settings.llm.provider,
        settings.llm.model,
        settings.llm.temperature,
        api_key,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS, tool_choice="auto")
    logger.info(
        f"LLM initialized: {settings.llm.provider}/{settings.llm.model} "
        f"with {len(ALL_TOOLS)} tools"
    )
    return llm_with_tools


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are ARIA (Audio Reasoning Intelligence Agent) — a world-class local AI assistant.

You have access to these tools:
  • create_file    — create text/config files
  • write_code     — generate and save source code
  • read_file      — read an existing file from disk
  • summarize_text — produce a bullet-point summary of provided text
  • run_terminal   — execute a shell command

Rules:
  1. Use tools whenever the user wants to create files, write code, read files,
     summarize text, or run commands. Do NOT describe actions — just do them.
  2. For general conversation (greetings, explanations, questions), respond directly
     WITHOUT calling any tools.
  3. Compound commands: if the user asks for multiple things, call multiple tools
     in sequence. You can call tools multiple times in one turn.
  4. After a tool succeeds, always give the user a brief friendly confirmation.
  5. **ACTIVE OUTPUT FOLDER**: `{output_path}` — use this value for the `folder`
     argument in ALL file creation/reading tools.
"""


# ── Node 1: Speech-to-Text ────────────────────────────────────────────────────

def stt_node(state: AgentState) -> dict:
    """
    Transcribes the audio file (if present) and adds the transcript as a HumanMessage.
    Skipped if no audio_path is in state (text-input path).
    """
    audio_path = state.get("audio_path")
    if not audio_path:
        return {}

    logger.info(f"STT: Processing {audio_path}")
    api_key = settings.api_key_for(settings.stt.provider)
    stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
    transcript = stt.transcribe(audio_path)
    short = transcript[:80] + ("…" if len(transcript) > 80 else "")
    logger.info(f"STT: Transcript → '{short}'")

    return {
        "transcript": transcript,
        "messages": [HumanMessage(content=transcript)],
    }


# ── Node 2: Agent (Reasoning + HITL) ─────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    Core reasoning node. Invokes the LLM and decides what to do next.

    Intent accumulation:
      Every tool label from every ReAct loop pass is merged into a combined
      intent string (e.g. "📖 Read File + 💻 Write Code"). Only falls back
      to "💬 General Chat" when zero tools are called in the entire turn.

    HITL (Human-in-the-Loop):
      If HITL is enabled and the LLM wants to call a destructive tool,
      interrupt() pauses the graph and surfaces a payload to the caller.
      Resuming with True proceeds; False cancels.
    """
    logger.info("Agent: Reasoning…")

    llm_with_tools = _get_llm_with_tools()
    output_path = state.get("output_path") or str(settings.output_path)

    system_msg = SystemMessage(
        content=_SYSTEM_TEMPLATE.format(output_path=output_path)
    )
    messages = [system_msg] + list(state.get("messages", []))
    response = llm_with_tools.invoke(messages)

    # ── Tool calls this pass? ──────────────────────────────────────────────────
    if hasattr(response, "tool_calls") and response.tool_calls:

        # Collect unique intent labels from THIS batch of tool calls
        new_intents: list[str] = []
        for tc in response.tool_calls:
            label = _TOOL_TO_INTENT.get(tc["name"], tc["name"].replace("_", " ").title())
            if label not in new_intents:
                new_intents.append(label)

        # Merge with intents from PREVIOUS passes in this turn (compound commands)
        existing = state.get("detected_intent") or ""
        existing_list: list[str] = (
            [i.strip() for i in existing.split(" + ")]
            if existing and existing != "💬 General Chat"
            else []
        )
        for ni in new_intents:
            if ni not in existing_list:
                existing_list.append(ni)

        combined_intent = " + ".join(existing_list)

        # Build human-readable action summary for HITL / UI
        calls_summary = "\n".join(
            f"  • **{tc['name']}**({', '.join(f'{k}={repr(v)[:40]}' for k, v in tc.get('args', {}).items())})"
            for tc in response.tool_calls
        )

        # ── HITL check ────────────────────────────────────────────────────────
        if settings.hitl.enabled:
            unsafe = [tc for tc in response.tool_calls if tc["name"] in UNSAFE_TOOLS]
            if unsafe:
                logger.info(f"Agent: HITL triggered for {[tc['name'] for tc in unsafe]}")
                confirmed = interrupt({
                    "message": "⚠️ **Safety Check** — ARIA wants to perform these actions:",
                    "actions_summary": calls_summary,
                    "tool_names": [tc["name"] for tc in unsafe],
                })
                if not confirmed:
                    logger.info("Agent: HITL — user cancelled.")
                    # CRITICAL: return a clean AIMessage with NO tool_calls.
                    # If we return the original `response` (which has tool_calls),
                    # _should_continue will route to the tools node and execute
                    # the action anyway — defeating the purpose of cancellation.
                    cancel_msg = AIMessage(
                        content="I've cancelled the requested action as you asked. Let me know if you'd like to do something else."
                    )
                    return {
                        "messages": [cancel_msg],
                        "detected_intent": combined_intent,
                        "action_taken": "❌ Cancelled by user.",
                    }

        logger.info(f"Agent: Intent → {combined_intent}")
        return {
            "messages": [response],
            "detected_intent": combined_intent,
            "action_taken": calls_summary,
        }

    # ── No tools called this pass ──────────────────────────────────────────────
    # If tools ran in a previous pass, keep the accumulated intent.
    # Only assign "💬 General Chat" when this is a pure chat turn (no tools at all).
    result: dict = {"messages": [response]}
    if not state.get("detected_intent"):
        result["detected_intent"] = "💬 General Chat"
        logger.info("Agent: Intent → 💬 General Chat")

    return result
