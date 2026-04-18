"""
agent/tools.py
──────────────
LangChain tools for ARIA.
Decorated with @tool so the LLM can call them via tool-calling APIs.

Safety: all file operations are sandboxed to the user-selected output folder.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from config.logging_config import logger


# ── Path helpers ──────────────────────────────────────────────────────────────

def _safe_path(filename: str, folder: str = "./output") -> Path:
    """
    Resolve `filename` inside `folder`, creating the folder if needed.
    Strips directory traversal components so the LLM can't escape the sandbox.
    """
    clean_name = Path(filename).name          # strip any ../ attempts
    target_dir = Path(folder).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / clean_name


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def create_file(filename: str, content: str = "", folder: str = "./output") -> str:
    """
    Creates a new text file with optional content.
    Use this for: saving notes, creating config files, writing plain-text data.

    Args:
        filename: Name of the file including extension (e.g., 'notes.txt').
        content: Text content to write into the file. Can be empty.
        folder: Target directory. Defaults to './output'.
    """
    try:
        path = _safe_path(filename, folder)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Tool: create_file → {path}")
        size = len(content.encode())
        return (
            f"✅ File `{filename}` created successfully in `{folder}`.\n"
            f"   Size: {size} bytes | Path: `{path}`"
        )
    except Exception as e:
        logger.error(f"Tool: create_file failed — {e}")
        return f"❌ Error creating file: {e}"


@tool
def write_code(filename: str, code: str, folder: str = "./output") -> str:
    """
    Generates and saves source code to a file.
    Use this for: Python scripts, JavaScript, HTML, CSS, shell scripts, or any programming task.
    The tool automatically strips accidental markdown fences and saves clean code.

    Args:
        filename: Target filename with extension (e.g., 'app.py', 'index.html').
        code: The source code content to save.
        folder: Target directory. Defaults to './output'.
    """
    try:
        # Strip markdown fences that LLMs sometimes include
        code = re.sub(r"^```[\w]*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE).strip()

        path = _safe_path(filename, folder)
        path.write_text(code, encoding="utf-8")
        logger.info(f"Tool: write_code → {path}")

        lines = code.count("\n") + 1
        preview = textwrap.shorten(code, width=300, placeholder="…")
        return (
            f"✅ Code saved to `{path}`.\n"
            f"   Lines: {lines} | Language: {Path(filename).suffix.lstrip('.') or 'text'}\n"
            f"   Preview:\n```\n{preview}\n```"
        )
    except Exception as e:
        logger.error(f"Tool: write_code failed — {e}")
        return f"❌ Error saving code: {e}"


@tool
def read_file(filename: str, folder: str = "./output") -> str:
    """
    Reads the full content of a file from the output folder.
    Use this FIRST when the user asks to: explain, modify, summarize, debug, or analyze an existing file.

    Args:
        filename: Name of the file to read (e.g., 'app.py').
        folder: Source directory. Defaults to './output'.
    """
    try:
        path = _safe_path(filename, folder)
        if not path.exists():
            return f"❌ File `{filename}` not found in `{folder}`."
        content = path.read_text(encoding="utf-8")
        logger.info(f"Tool: read_file → {path}")
        size_kb = round(len(content.encode()) / 1024, 1)
        return f"📖 Content of `{filename}` ({size_kb} KB):\n\n```\n{content}\n```"
    except Exception as e:
        logger.error(f"Tool: read_file failed — {e}")
        return f"❌ Error reading file: {e}"


@tool
def summarize_text(text: str) -> str:
    """
    Produces a concise bullet-point summary of the provided text.
    Use this when the user asks to summarize, condense, or give key points of any text.
    The summary is returned directly — it does NOT save to a file unless you also call create_file.

    Args:
        text: The text content to summarize.
    """
    # This is a tool that signals intent to the agent — the LLM itself
    # generates the summary via its own response after this tool is invoked.
    # We return the raw text so the agent can use it in its final response.
    logger.info(f"Tool: summarize_text — input length {len(text)} chars")
    if len(text) < 50:
        return f"📝 Text is very short. Here it is as-is:\n\n{text}"
    return f"📝 Text to summarize ({len(text)} chars):\n\n{text}\n\n---\nGenerate a concise summary with bullet points."


@tool
def run_terminal(command: str) -> str:
    """
    Executes a safe shell command on the local system.
    Use this for: running Python scripts, listing files ('ls'), checking versions,
    installing packages ('pip install'), or any other shell operation.

    Args:
        command: The exact shell command to run (e.g., 'python app.py', 'ls ./output').
    """
    try:
        logger.info(f"Tool: run_terminal → '{command}'")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        if result.returncode == 0:
            output = stdout or "(no output)"
            return f"✅ Command ran successfully.\n```\n{output}\n```"
        else:
            return (
                f"⚠️ Command exited with code {result.returncode}.\n"
                f"stderr:\n```\n{stderr}\n```\n"
                f"stdout:\n```\n{stdout}\n```"
            )
    except subprocess.TimeoutExpired:
        return "❌ Command timed out after 30 seconds."
    except Exception as e:
        logger.error(f"Tool: run_terminal failed — {e}")
        return f"❌ Error executing command: {e}"


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [create_file, write_code, read_file, summarize_text, run_terminal]
