"""
agent/nodes.py
──────────────
LangGraph node functions.
Each function receives AgentState, returns a partial state update dict.

Pipeline:
  stt_node → intent_node → [conditional] → hitl_node → tool_node
                                         → tool_node (if no HITL needed)
"""

from __future__ import annotations

import traceback

from langgraph.types import interrupt

from agent.intent import get_intent_chain, get_llm
from agent.state import AgentState, IntentAction, IntentResult
from agent.stt import get_stt
from agent.tools import (
    create_file, general_chat, read_file,
    run_terminal, summarize_text, write_code,
)
from config.logging_config import logger
from config.settings import settings


# ── Node 1: Speech-to-Text ────────────────────────────────────────────────────

def stt_node(state: AgentState) -> dict:
    """Transcribe the audio file at state['audio_path'] → transcript."""
    # Pre-populated transcript (text input path) — skip STT entirely
    if state.get("transcript"):
        return {"transcript": state["transcript"], "error": None}

    try:
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(state["audio_path"])

        if not transcript.strip():
            return {"error": "STT returned an empty transcript. Please speak clearly and try again."}

        logger.info(f"STT complete: '{transcript[:60]}...'")
        return {"transcript": transcript, "error": None}

    except FileNotFoundError as e:
        return {"error": f"Audio file not found: {e}"}
    except Exception as e:
        return {"error": f"STT failed: {type(e).__name__}: {e}"}


# ── Node 2: Intent Classification ─────────────────────────────────────────────

def intent_node(state: AgentState) -> dict:
    """Classify intent(s) from the transcript using structured LLM output."""
    if state.get("error"):
        return {}  # Propagate error, skip this node

    try:
        api_key = settings.api_key_for(settings.llm.provider)

        # Pass last 10 messages as context so the LLM can resolve pronouns / references
        history_msgs = state.get("messages", [])[-10:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history_msgs)

        chain = get_intent_chain(
            settings.llm.provider,
            settings.llm.model,
            settings.llm.temperature,
            api_key,
        )
        result: IntentResult = chain.invoke({
            "transcript": state["transcript"],
            "history": history_text,
        })

        logger.info(f"Intent classified: {[a.intent for a in result.actions]}")

        # Safety net: ensure at least one action exists
        if not result.actions:
            result = IntentResult(
                actions=[
                    IntentAction(
                        intent="general_chat",
                        description=state.get("transcript", ""),
                    )
                ],
                requires_confirmation=False,
            )

        return {"intent_result": result.model_dump(), "error": None}

    except Exception as e:
        logger.warning(f"Intent classification failed, falling back to chat: {e}")
        # Graceful degradation — always respond, never crash
        fallback = IntentResult(
            actions=[
                IntentAction(
                    intent="general_chat",
                    description=state.get("transcript", ""),
                )
            ],
            requires_confirmation=False,
        )
        return {
            "intent_result": fallback.model_dump(),
            "error": f"Intent classification failed (falling back to chat): {e}",
        }


# ── Node 3: Human-in-the-Loop (HITL) ─────────────────────────────────────────

def hitl_node(state: AgentState) -> dict:
    """
    Pause execution and ask for user confirmation before filesystem operations.

    Uses LangGraph's interrupt() — the graph checkpoints here and waits for
    a Command(resume={confirmed: bool}) call from the API layer.
    """
    if state.get("error") and not state.get("intent_result"):
        return {"confirmed": False}

    intent_result = IntentResult(**state["intent_result"])

    icons = {
        "create_file": "📄", "write_code": "💻", "summarize": "📋",
        "read_file": "📖", "run_terminal": "⚡", "general_chat": "💬",
    }

    lines = []
    for action in intent_result.actions:
        line = f"{icons.get(action.intent, '•')} [{action.intent}] {action.description or ''}"
        if action.filename:
            line += f"  →  `{action.filename}`"
        if action.command:
            line += f"  →  `{action.command}`"
        lines.append(line)

    summary = "\n".join(lines)
    logger.info("HITL: Pipeline paused — awaiting user approval")

    # interrupt() checkpoints state to SQLite and suspends execution here.
    # Resumes when server.py calls graph.invoke(Command(resume={...}), config)
    response = interrupt({
        "message": f"ARIA wants to perform the following actions:\n\n{summary}\n\nProceed?",
        "actions_summary": summary,
        "actions": [a.model_dump() for a in intent_result.actions],
    })

    confirmed: bool = response.get("confirmed", False)
    logger.info(f"HITL: {'APPROVED' if confirmed else 'CANCELLED'}")
    return {"confirmed": confirmed}


# ── Node 4: Tool Execution ─────────────────────────────────────────────────────

def tool_node(state: AgentState) -> dict:
    """
    Execute the classified actions.
    Handles compound commands by iterating over all IntentActions.
    """
    if state.get("error") and not state.get("intent_result"):
        return {
            "tool_results": [{"action": "error", "result": f"❌ {state['error']}"}]
        }

    intent_result = IntentResult(**state["intent_result"])
    api_key = settings.api_key_for(settings.llm.provider)
    llm = get_llm(
        settings.llm.provider, settings.llm.model,
        settings.llm.temperature, api_key,
    )

    # Use per-request output_path if set, else fall back to global config
    output_folder = state.get("output_path") or str(settings.output_path)

    results: list[dict] = []

    for action in intent_result.actions:
        try:
            if action.intent == "create_file":
                result = create_file(
                    filename=action.filename or "untitled.txt",
                    content=action.content or "",
                    folder=output_folder,
                )
                logger.info(f"Tool: create_file → {action.filename}")
                results.append({"action": f"Created file: `{action.filename}`", "result": result})

            elif action.intent == "read_file":
                result = read_file(
                    filename=action.filename or "untitled.txt",
                    folder=output_folder,
                )
                logger.info(f"Tool: read_file → {action.filename}")
                results.append({"action": f"Read file: `{action.filename}`", "result": result})

            elif action.intent == "run_terminal":
                result = run_terminal(command=action.command or "")
                logger.info(f"Tool: run_terminal → '{action.command}'")
                results.append({"action": f"Executed: `{action.command}`", "result": result})

            elif action.intent == "write_code":
                result = write_code(
                    filename=action.filename or "generated.py",
                    description=action.description,
                    llm=llm,
                    folder=output_folder,
                )
                logger.info(f"Tool: write_code → {action.filename}")
                results.append({"action": f"Generated code: `{action.filename}`", "result": result})

            elif action.intent == "summarize":
                text = action.content or action.description or state.get("transcript", "")
                result = summarize_text(text=text, llm=llm)
                results.append({"action": "Summarized text", "result": result})

            elif action.intent == "general_chat":
                result = general_chat(
                    message=action.description or state.get("transcript", ""),
                    history=state.get("messages", []),
                    llm=llm,
                    history_limit=settings.memory.session_history_limit,
                )
                results.append({"action": "Chat response", "result": result})

        except Exception as e:
            logger.error(f"Tool {action.intent} failed: {e}")
            results.append({
                "action": f"Failed: {action.intent}",
                "result": f"❌ Error during {action.intent}: {type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            })

    # Build new chat history entries
    new_messages: list[dict] = []
    transcript = state.get("transcript", "")
    if transcript:
        new_messages.append({"role": "user", "content": transcript})
    for r in results:
        new_messages.append({"role": "assistant", "content": r["result"]})

    updated_messages = (state.get("messages") or []) + new_messages

    return {
        "tool_results": results,
        "messages": updated_messages,
        "error": state.get("error"),  # pass through any soft warning
    }
