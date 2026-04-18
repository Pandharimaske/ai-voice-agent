"""
agent/tools.py
──────────────
Concrete tool implementations.
All filesystem writes are sandboxed to the configured output folder.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Remove dangerous characters from user-supplied filenames."""
    # Strip path traversal
    name = Path(name).name
    # Replace spaces and special chars
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "untitled.txt"


def _safe_path(filename: str, folder: str) -> Path:
    """
    Return an absolute Path inside `folder`.
    Checks if the folder exists; if not, logs that it is creating it.
    If a file with that name already exists, appends a timestamp suffix.
    """
    from config.logging_config import logger
    
    output_dir = Path(folder)
    
    if not output_dir.exists():
        logger.info(f"Folder '{folder}' does not exist. Creating it now.")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Using existing folder: '{folder}'")

    clean_name = _sanitize_filename(filename)
    target = output_dir / clean_name

    if target.exists():
        stem = target.stem
        suffix = target.suffix
        ts = int(time.time())
        target = output_dir / f"{stem}_{ts}{suffix}"

    return target


# ── Tool functions ─────────────────────────────────────────────────────────────

def create_file(filename: str, content: str = "", folder: str = "./output") -> str:
    """Create a file with optional content in the output folder."""
    path = _safe_path(filename, folder)
    path.write_text(content, encoding="utf-8")
    return f"✅ File created: `{path}`"


def write_code(
    filename: str,
    description: str,
    llm: BaseChatModel,
    folder: str = "./output",
) -> str:
    """
    Generate code via LLM based on `description`, save to `filename`.
    Returns a preview of the generated code.
    """
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".java": "Java", ".cpp": "C++", ".c": "C", ".go": "Go",
        ".rs": "Rust", ".sh": "Bash", ".sql": "SQL", ".html": "HTML",
        ".css": "CSS", ".rb": "Ruby", ".php": "PHP",
    }
    suffix = Path(filename).suffix.lower()
    lang = ext_map.get(suffix, "the appropriate language")

    messages = [
        SystemMessage(content=(
            f"You are an expert {lang} developer. "
            "Write clean, well-commented, production-quality code. "
            "Return ONLY the raw code — no markdown fences, no preamble, no explanation."
        )),
        HumanMessage(content=f"Write {lang} code for: {description}"),
    ]

    response = llm.invoke(messages)
    code: str = response.content.strip()

    # Strip accidental markdown fences
    code = re.sub(r"^```[\w]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    path = _safe_path(filename, folder)
    path.write_text(code, encoding="utf-8")

    preview = code[:600] + ("\n... (truncated)" if len(code) > 600 else "")
    return f"✅ Code saved to `{path}`\n\n```{suffix.lstrip('.')}\n{preview}\n```"


def summarize_text(text: str, llm: BaseChatModel) -> str:
    """Summarize provided text using the LLM."""
    if not text.strip():
        return "⚠️ No text provided to summarize."

    messages = [
        SystemMessage(content=(
            "You are a precise summarizer. "
            "Provide a clear, structured summary with key points. "
            "Use bullet points for main ideas. Be concise but comprehensive."
        )),
        HumanMessage(content=f"Summarize the following:\n\n{text}"),
    ]

    response = llm.invoke(messages)
    return response.content


def general_chat(
    message: str,
    history: List[dict],
    llm: BaseChatModel,
    history_limit: int = 20,
    stream: bool = False,
):
    """
    Conversational response with session memory.
    `history` is a list of {role: str, content: str} dicts.
    If stream=True, returns a generator of tokens.
    """
    messages: list = [
        SystemMessage(content=(
            "You are ARIA, a helpful voice-controlled AI assistant. "
            "You can create files, write code, summarize text, and answer questions. "
            "Be helpful, concise, and friendly."
        ))
    ]

    # Inject session history (bounded by limit)
    for entry in history[-history_limit:]:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))
    
    if stream:
        return llm.stream(messages)
    
    response = llm.invoke(messages)
    return response.content


def read_file(filename: str, folder: str = "./output") -> str:
    """Read the content of a file from the specified folder."""
    from config.logging_config import logger
    
    clean_name = Path(filename).name
    path = Path(folder) / clean_name
    
    if not path.exists():
        logger.error(f"Read failed: File not found at {path}")
        return f"❌ Error: File `{filename}` not found in `{folder}`."
    
    try:
        content = path.read_text(encoding="utf-8")
        logger.info(f"Successfully read file: {path}")
        return f"📖 **Content of `{filename}`:**\n\n```\n{content}\n```"
    except Exception as e:
        logger.error(f"Read failed for {path}: {str(e)}")
        return f"❌ Error reading file `{filename}`: {str(e)}"


def run_terminal(command: str) -> str:
    """Execute a shell command locally and return the output."""
    import subprocess
    from config.logging_config import logger
    
    logger.info(f"Execution: run_terminal -> '{command}'")
    try:
        # We use shell=True for convenience but it requires HITL safety
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # Safety timeout
        )
        
        output = result.stdout.strip()
        error = result.stderr.strip()
        
        if result.returncode == 0:
            logger.info(f"Command successful: {command}")
            return f"✅ **Command Executed:** `{command}`\n\n```\n{output}\n```"
        else:
            logger.error(f"Command failed with code {result.returncode}: {error}")
            return f"⚠️ **Command Failed (Code {result.returncode}):**\n\n```\n{error}\n{output}\n```"
            
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return f"❌ **Error:** Command timed out after 30 seconds."
    except Exception as e:
        logger.error(f"Command error: {str(e)}")
        return f"❌ **Error executing command:** {str(e)}"
