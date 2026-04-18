"""
agent/intent.py
───────────────
Intent classification using LangChain + Pydantic structured output.
Supports compound commands (multiple intents in one utterance).
LLM provider is set in config.yaml → llm.provider.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent.state import IntentResult


# ── Prompt ─────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """\
You are the intent classifier for ARIA, a voice-controlled AI agent.

Your job is to analyze transcribed speech and extract ALL the user's intended \
actions — including compound commands (e.g. "write a sort function AND save it \
to sort.py").

## Intent Types
- **create_file**: User wants to create a file or folder (text, empty, or with content)
- **write_code**: User wants code generated and saved to a file
- **read_file**: Read the content of an existing file. Use this if the user asks to see a file, explain a file, or summarize an EXISTING file.
- **run_terminal**: Execute a shell command. Use this if the user asks to run a script, install a package, or check something in the terminal.
- **summarize**: Summarize provided text. (If they provide a filename, use read_file FIRST).
- **general_chat**: Use for greetings, questions, or if no other intent fits.

## Rules
1. A single utterance CAN contain multiple intents — list ALL of them.
2. `requires_confirmation` must be `true` if ANY action touches the filesystem \
   (create_file or write_code) or runs a terminal command (run_terminal). False only when ALL actions are read_file or general_chat.
3. For filenames: infer a sensible name + extension if not explicitly stated.
4. For `write_code`: set content to null — the coding agent will generate it.
5. For `create_file` with explicit content: put that content in the content field.
6. Extraction Rules:
    - filename: For create_file, write_code, and read_file.
    - command: For run_terminal (the exact shell command to run).
    - content: For create_file (the body of the file).
    - description: For write_code (what the code should do) or summarize (the text to summarize).
    - requires_confirmation: Set to 'true' for create_file, write_code, and run_terminal. Set to 'false' for read_file and general_chat.
7. Never invent intents that aren't in the user's speech.
"""

INTENT_HUMAN_PROMPT = """\
Conversation History:
{history}

Latest Transcript: {transcript}
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(
    provider: str,
    model: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.
    One-line swap: change provider in config.yaml.
    """
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Available: groq | openai | anthropic"
        )


# ── Intent chain ───────────────────────────────────────────────────────────────

def get_intent_chain(provider: str, model: str, temperature: float, api_key: Optional[str] = None):
    """
    Returns a runnable chain:  {transcript, history} → IntentResult
    """
    llm = get_llm(provider, model, temperature, api_key)
    structured_llm = llm.with_structured_output(IntentResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_PROMPT),
        ("human", INTENT_HUMAN_PROMPT),
    ])

    return prompt | structured_llm
