<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c6aff&height=200&section=header&text=ARIA&fontSize=90&fontColor=ffffff&fontAlignY=38&desc=Audio%20Reasoning%20Intelligence%20Agent&descAlignY=60&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.x-7c6aff?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-FF6B35?style=for-the-badge&logo=chainlink&logoColor=white)](https://smith.langchain.com)

<br/>

> **Speak a command. Watch ARIA think. See it happen.**
> A production-grade, voice-controlled local AI agent with real-time pipeline visualization,
> Human-in-the-Loop safety controls, and persistent session memory.

<br/>

</div>

---

## 📋 Assignment Requirements Checklist

> Every requirement from the spec — plus all 5 bonus items — is implemented.

### Core Requirements

| # | Requirement | Status | Implementation |
|---|---|---|---|
| 1a | Microphone input | ✅ | `MediaRecorder` API in `recorder.js` with live waveform visualizer |
| 1b | Audio file upload (.wav / .mp3) | ✅ | 📂 upload button → `POST /api/process_stream` |
| 2 | Speech-to-Text (Whisper) | ✅ | Groq `whisper-large-v3` via `agent/stt.py` *(see hardware note below)* |
| 3 | Intent understanding via LLM | ✅ | Llama 3.3 70B tool-calling via Groq *(see hardware note below)* |
| 3a | Create file intent | ✅ | `create_file` tool |
| 3b | Write code intent | ✅ | `write_code` tool |
| 3c | Summarize text intent | ✅ | `summarize_text` tool |
| 3d | General chat intent | ✅ | No-tool path in `agent_node` |
| 4 | File operations | ✅ | Sandboxed to `output/` via `_safe_path()` |
| 4a | Code generation | ✅ | `write_code` strips markdown fences, saves clean code |
| 4b | Text summarization | ✅ | `summarize_text` tool → LLM generates bullet points |
| 5 | Web UI | ✅ | FastAPI + custom HTML/CSS/JS (no Streamlit dependency) |
| 5a | Transcribed text displayed | ✅ | Shown in user message bubble + pipeline tracker |
| 5b | Detected intent displayed | ✅ | Animated badge on every user message |
| 5c | Action taken displayed | ✅ | Logged in `action_taken` state field, shown in pipeline |
| 5d | Final output / result | ✅ | Agent's final response shown in chat; output files panel auto-refreshes |

### Bonus Features (All 5 Implemented)

| Bonus | Status | Implementation |
|---|---|---|
| ✦ Compound Commands | ✅ | Chain multiple tools in one utterance; intent badge accumulates: `📖 Read File + 💻 Write Code + 📄 Create File` |
| ✦ Human-in-the-Loop | ✅ | LangGraph `interrupt()` inside `agent_node` — UI shows action summary before any file/code/command executes |
| ✦ Graceful Degradation | ✅ | All tools wrapped in `try/except`; STT failures return `❌` messages; unintelligible audio falls through to general chat |
| ✦ Memory | ✅ | `AsyncSqliteSaver` checkpointer — full session history persisted in SQLite; sessions loadable and deletable from sidebar |
| ✦ Additional tools | ✅ | `read_file` + `run_terminal` beyond the minimum spec |

---

## 🔧 Hardware Note — Why Groq instead of Ollama?

> *The assignment asks for a local/HuggingFace model but explicitly allows an API if hardware is a constraint. Here is the documented reasoning:*

**STT — Groq `whisper-large-v3` instead of local Whisper:**
- Running `whisper-large-v3` locally requires ~6 GB VRAM and significant CPU time per transcription.
- Groq's Whisper inference is **10–50× faster** than CPU-based local Whisper at the same accuracy.
- The `agent/stt.py` abstraction is **provider-agnostic** — swap `provider: "openai"` in `config.yaml` to use OpenAI's Whisper API, or change the code to load a local HuggingFace pipeline with one function.

**LLM — Groq `llama-3.3-70b-versatile` instead of Ollama:**
- Llama 3.3 70B requires ~40 GB VRAM (or ~80 GB in 4-bit) to run locally.
- Smaller local models (7B–13B) via Ollama were tested but **consistently failed structured tool-calling**, producing malformed JSON or hallucinated function names.
- The `agent/llm.py` factory supports Ollama out of the box — set `provider: "ollama"` and `model: "llama3.2"` in `config.yaml` to switch if your hardware supports it.

---

## 🎬 Deliverables

| Deliverable | Link |
|---|---|
| 📦 **Code Repository** | [github.com/Pandharimaske/ai-voice-agent](https://github.com/Pandharimaske/ai-voice-agent) |
| 🎥 **Video Demo** | *(add YouTube unlisted link here)* |
| 📝 **Technical Article** | *(add Substack / Dev.to / Medium link here)* |

---

## ✨ What is ARIA?

**ARIA** (Audio Reasoning Intelligence Agent) is a fully local, voice-controlled AI agent that:

1. 🎤 **Hears** your voice or accepts typed commands
2. 🧠 **Reasons** using a 70B LLM with 5 specialized tools
3. ⚡ **Acts** — creates files, writes code, reads files, summarizes text, runs commands
4. 🔒 **Asks** before doing anything destructive (Human-in-the-Loop)
5. 💾 **Remembers** every session in a local SQLite database

All of this runs through a sleek dark-mode UI with a real-time pipeline tracker, animated intent badges, and session management.

---

## 🏗️ Architecture

```
+---------------------------------------------------------------------+
|                         ARIA Pipeline                               |
|                                                                     |
|   Audio / Text Input                                                |
|          |                                                          |
|   +------+------+                                                   |
|   |  STT Node   |  Groq Whisper-large-v3 -> plain text transcript  |
|   +------+------+                                                   |
|          | transcript                                               |
|   +------+------+        +--------------+                          |
|   | Agent Node  |<------►|  Tools Node  |  LangGraph ReAct loop    |
|   |  (LLM +     |        |  (ToolNode)  |                          |
|   |   Tools)    |        +--------------+                          |
|   +------+------+                                                   |
|          |                                                          |
|    [HITL needed?]                                                   |
|     +- YES --> interrupt() --> UI confirmation dialog               |
|     |          Command(resume=True/False) --> continue/cancel       |
|     +- NO  --> stream final response to UI                         |
|                                                                     |
|   Persistence: AsyncSqliteSaver (LangGraph checkpointer)           |
+---------------------------------------------------------------------+
```

### Graph Topology (4 nodes)

```
START --> stt --> agent <--> tools
                   |
                  END
```

The **Agent node** is the single source of truth for:
- Tool selection & reasoning (LLM)
- Intent classification & accumulation
- HITL interrupt control via `interrupt()`

---

## 🛠️ Tools

| Tool | Intent Badge | Triggers HITL? | Description |
|---|---|---|---|
| `create_file` | 📄 Create File | ✅ Yes | Creates text/config/note files |
| `write_code` | 💻 Write Code | ✅ Yes | Generates & saves source code (any language) |
| `read_file` | 📖 Read File | ❌ No | Reads existing files for analysis/debug |
| `summarize_text` | 📝 Summarize Text | ❌ No | Bullet-point summarization |
| `run_terminal` | ⚡ Run Command | ✅ Yes | Executes shell commands with 30s timeout |

**Compound commands** are fully supported — say *"Read my script, summarize it, then save the summary"* and ARIA chains all 3 tools automatically. The intent badge accumulates: `📖 Read File + 📝 Summarize Text + 📄 Create File`.

---

## 🔒 Human-in-the-Loop (HITL)

Powered by LangGraph's **`interrupt()`** function (0.4.x pattern — not the deprecated `interrupt_before` node):

```
Agent decides to write_code
          |
   interrupt() called inside agent_node
          |
   UI shows confirmation dialog with exact action summary
          |
   User clicks [Confirm]  -->  Command(resume=True)  --> tool executes
   User clicks [Cancel]   -->  Command(resume=False) --> clean cancellation
                               (returns fresh AIMessage with no tool_calls
                                so _should_continue() routes to END, not tools)
```

> ⚠️ **Critical bug this design solves**: On cancellation, ARIA returns a clean `AIMessage` with no `tool_calls`. If it returned the original LLM response (which *does* have `tool_calls`), the graph router would still execute the cancelled tools — defeating the purpose of the confirmation.

---

## 📦 Project Structure

```
ai-voice-agent/
│
├── agent/
│   ├── __init__.py        # Package exports
│   ├── state.py           # AgentState TypedDict (add_messages reducer)
│   ├── nodes.py           # stt_node, agent_node (interrupt + lru_cache)
│   ├── tools.py           # 5 LangChain @tool definitions + ALL_TOOLS registry
│   ├── graph.py           # StateGraph compilation, streaming, HITL resume
│   ├── llm.py             # Multi-provider LLM factory (Groq/OpenAI/Anthropic/Ollama)
│   └── stt.py             # STT abstraction (Groq Whisper / OpenAI)
│
├── config/
│   ├── settings.py        # Pydantic settings (YAML + env var loading)
│   └── logging_config.py  # Structured logger
│
├── ui/
│   ├── index.html         # Single-page app
│   ├── style.css          # Dark-mode design system (CSS variables)
│   ├── app.js             # SSE streaming, pipeline tracker, HITL flow
│   └── recorder.js        # MediaRecorder audio capture + visualizer
│
├── server.py              # FastAPI app (lifespan, SSE, REST endpoints)
├── config.yaml            # Single config file — swap models in one line
├── .env.example           # API key template (copy to .env)
├── pyproject.toml         # uv/pip project definition
└── .env                   # API keys (never committed)
```

---

## 🚀 Quick Start

### 1 · Clone & install

```bash
git clone https://github.com/Pandharimaske/ai-voice-agent.git
cd ai-voice-agent

# Install uv (recommended — fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### 2 · Configure API keys

```bash
cp .env.example .env
```

```env
# Required
GROQ_API_KEY=gsk_...

# Optional — LangSmith observability
LANGCHAIN_API_KEY=lsv2_pt_...
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=AI-Voice-Agent
```

### 3 · (Optional) Swap models

Edit `config.yaml` — zero code changes needed:

```yaml
stt:
  provider: "groq"
  model: "whisper-large-v3"         # or "openai" + "whisper-1"

llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"  # or "openai"/"gpt-4o" or "ollama"/"llama3.2"
  temperature: 0.1

hitl:
  enabled: true    # set false to skip confirmations
```

### 4 · Run

```bash
uv run python server.py
```

Open **http://localhost:8000** 🎉

---

## 🖥️ UI Features

| Feature | Description |
|---|---|
| **Pipeline Steps Bar** | Live tracker: `🎤 Transcribe → 🧠 Reason → ⚡ Execute → ✅ Done` |
| **Intent Badges** | Per-turn badge accumulates all tools: `📖 Read File + 💻 Write Code` |
| **Voice Recording** | Click 🎙️ to record via mic with live waveform visualizer |
| **File Upload** | Click 📂 to upload any `.wav` / `.mp3` audio file |
| **HITL Dialog** | Confirms destructive actions with full action summary before execution |
| **Session Sidebar** | Full history with one-click load & 🗑️ delete |
| **Output Files Panel** | Auto-refreshes after every tool run |
| **Output Path Selector** | Switch / add output folders without restarting |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/process_text_stream` | Text → SSE stream of agent updates |
| `POST` | `/api/process_stream` | Audio upload → SSE stream |
| `POST` | `/api/transcribe` | Audio → transcript only (no agent) |
| `POST` | `/api/confirm` | Resume HITL (`{thread_id, confirmed: bool}`) |
| `GET` | `/api/sessions` | List all saved sessions |
| `GET` | `/api/sessions/{id}` | Load a specific session |
| `DELETE` | `/api/sessions/{id}` | Delete session from SQLite |
| `GET` | `/api/output_files` | List files in output folder |

**SSE event payload:**

```jsonc
{
  "thread_id": "uuid",
  "transcript": "Create a Python file with a retry function",
  "detected_intent": "💻 Write Code + 📄 Create File",
  "action_taken": "  • write_code(filename='retry.py', ...)",
  "messages": [
    { "role": "user",      "content": "...", "intent": "💻 Write Code + 📄 Create File" },
    { "role": "assistant", "content": "Done! Saved retry.py to ./output" }
  ],
  "is_interrupted": true,
  "interrupt_data": { "message": "Confirm?", "tool_names": ["write_code"] },
  "output_path": "./output",
  "error": null
}
```

---

## 🧠 Technical Design Decisions

### `add_messages` Reducer (LangGraph 0.4)
```python
class AgentState(TypedDict, total=False):
    messages: Annotated[List, add_messages]  # ID-based dedup on checkpoint restore
```

### `interrupt()` — Code-level HITL (not `interrupt_before`)
```python
# Runs INSIDE agent_node — granular, condition-based
confirmed = interrupt({"message": "...", "tool_names": [...]})
if not confirmed:
    return {"messages": [AIMessage(content="Cancelled.")]}  # no tool_calls!
```

### Intent Accumulation Across ReAct Passes
```
Pass 1: LLM calls read_file   → detected_intent = "📖 Read File"
Pass 2: LLM calls write_code  → detected_intent = "📖 Read File + 💻 Write Code"
New turn starts               → reset to None
```

### FastAPI Lifespan for AsyncSqliteSaver
```python
@asynccontextmanager
async def lifespan(app):
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        init_graph(saver)  # compile graph once
        yield              # serve requests
    # connection closed automatically on shutdown
```

---

## ⚙️ Supported Model Providers

| Provider | LLM | STT |
|---|---|---|
| **Groq** ⭐ recommended | `llama-3.3-70b-versatile`, `mixtral-8x7b` | `whisper-large-v3` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | `whisper-1` |
| **Anthropic** | `claude-3-5-sonnet-20241022` | — |
| **Ollama** (local) | `llama3.2`, `mistral`, `codellama` | — |

---

## 🔭 Observability with LangSmith

ARIA ships with **zero-config LangSmith tracing** — LangChain and LangGraph automatically pick up these env vars:

```env
LANGCHAIN_API_KEY=lsv2_pt_...
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=AI-Voice-Agent
```

Every pipeline run is visible at **https://smith.langchain.com** with full STT → Agent → Tools → Response traces, token usage, latency breakdowns, and HITL checkpoint events.

---

## 🗺️ Roadmap

- [ ] Text-to-Speech output (read responses aloud)
- [ ] Streaming token-by-token response in chat
- [ ] Multi-file context (RAG over output folder)
- [ ] Plugin system for custom tools
- [ ] Docker compose for one-command deploy

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c6aff&height=100&section=footer" width="100%"/>

**Built with LangGraph · LangChain · FastAPI · Groq**

⭐ Star this repo if ARIA impressed you!

</div>
