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
from typing import Any

from langgraph.types import interrupt

from agent.intent import get_intent_chain, get_llm
from agent.state import AgentState, IntentResult
from agent.stt import get_stt
from agent.tools import create_file, general_chat, summarize_text, write_code, read_file, run_terminal
from config.settings import settings
from config.logging_config import logger


# ── Node 1: Speech-to-Text ────────────────────────────────────────────────────

def stt_node(state: AgentState) -> dict:
    """Transcribe the audio file at state['audio_path'] → transcript."""
    # If transcript is already provided (e.g. from direct text input), skip STT
    if state.get("transcript"):
        return {"transcript": state["transcript"], "error": None}

    try:
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(state["audio_path"])

        if not transcript.strip():
            return {"error": "STT returned an empty transcript. Please speak clearly and try again."}

        return {"transcript": transcript, "error": None}

    except FileNotFoundError as e:
        return {"error": f"Audio file not found: {e}"}
    except Exception as e:
        return {"error": f"STT failed: {type(e).__name__}: {e}"}


# ── Node 2: Intent Classification ─────────────────────────────────────────────

def intent_node(state: AgentState) -> dict:
    """Classify intent(s) from the transcript using structured LLM output."""
    if state.get("error"):
        return {}  # Propagate error, skip

    try:
        api_key = settings.api_key_for(settings.llm.provider)
        # Format chat history for the intent classifier
        history_msgs = state.get("messages", [])[-10:]  # last 10 messages for context
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])

        chain = get_intent_chain(
            settings.llm.provider,
            settings.llm.model,
            settings.llm.temperature,
            api_key,
        )
        result: IntentResult = chain.invoke({
            "transcript": state["transcript"],
            "history": history_text
        })
        
        logger.info(f"Intent classified: {[a.intent for a in result.actions]}")
        
        # Ensure at least one action exists
        if not result.actions:
            result.actions = [{
                "intent": "general_chat",
                "filename": None,
                "description": state.get("transcript", ""),
                "content": None,
            }]
            
        return {"intent_result": result.model_dump(), "error": None}

    except Exception as e:
        # Graceful degradation: fall back to general_chat
        fallback = IntentResult(
            actions=[{
                "intent": "general_chat",
                "filename": None,
                "description": state.get("transcript", ""),
                "content": None,
            }],
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
    a Command(resume={confirmed: bool}) call from the UI.
    """
    if state.get("error") and not state.get("intent_result"):
        return {"confirmed": False}

    intent_result = IntentResult(**state["intent_result"])

    # Build a human-readable action summary
    lines = []
    for i, action in enumerate(intent_result.actions, 1):
        icon = {
            "create_file": "📄", 
            "write_code": "💻", 
            "summarize": "📋", 
            "read_file": "📖", 
            "run_terminal": "⚡", 
            "general_chat": "💬"
        }.get(action.intent, "•")
        
        line = f"{icon} [{action.intent}] {action.description or ''}"
        if action.filename:
            line += f"  →  `{action.filename}`"
        if action.command:
            line += f"  →  `{action.command}`"
        lines.append(line)

    summary = "\n".join(lines)

    # ← interrupt() pauses the graph here; resumes when UI calls Command(resume=...)
    logger.info("HITL: Pipeline paused. Waiting for user approval...")
    response = interrupt({
        "message": f"ARIA wants to perform the following actions:\n\n{summary}\n\nProceed?",
        "actions_summary": summary,
        "actions": [a.model_dump() for a in intent_result.actions],
    })

    confirmed: bool = response.get("confirmed", False)
    logger.info(f"HITL: User decision: {'APPROVED' if confirmed else 'CANCELLED'}")
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
    llm = get_llm(settings.llm.provider, settings.llm.model, settings.llm.temperature, api_key)
    
    # Use custom output_path if set in state, else fallback to global setting
    output_folder = state.get("output_path") or str(settings.output_path)

    results: list[dict] = []
    new_messages: list[dict] = []

    for action in intent_result.actions:
        try:
            if action.intent == "create_file":
                result = create_file(
                    filename=action.filename or "untitled.txt",
                    content=action.content or "",
                    folder=output_folder,
                )
                logger.info(f"Execution: create_file -> {action.filename}")
                results.append({"action": f"Created file: `{action.filename}`", "result": result})

            elif action.intent == "read_file":
                result = read_file(
                    filename=action.filename or "untitled.txt",
                    folder=output_folder,
                )
                logger.info(f"Execution: read_file -> {action.filename}")
                results.append({"action": f"Read file: `{action.filename}`", "result": result})

            elif action.intent == "run_terminal":
                result = run_terminal(
                    command=action.command or "",
                )
                logger.info(f"Execution: run_terminal -> '{action.command}'")
                results.append({"action": f"Executed Command: `{action.command}`", "result": result})

            elif action.intent == "write_code":
                result = write_code(
                    filename=action.filename or "generated.py",
                    description=action.description,
                    llm=llm,
                    folder=output_folder,
                )
                logger.info(f"Execution: write_code -> {action.filename}")
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
            results.append({
                "action": f"Failed: {action.intent}",
                "result": f"❌ Error during {action.intent}: {type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            })

    # Update chat history
    transcript = state.get("transcript", "")
    if transcript:
        new_messages.append({"role": "user", "content": transcript})
    for r in results:
        new_messages.append({"role": "assistant", "content": r["result"]})

    # Merge with existing history
    updated_messages = (state.get("messages") or []) + new_messages

    return {
        "tool_results": results,
        "messages": updated_messages,
        "error": state.get("error"),  # pass through any soft error
    }
