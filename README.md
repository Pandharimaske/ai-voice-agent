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

## ✨ What is ARIA?

**ARIA** (Audio Reasoning Intelligence Agent) is a fully local, voice-controlled AI agent that:

1. 🎤 **Hears** your voice or accepts typed commands
2. 🧠 **Reasons** using a 70B LLM with 5 specialized tools
3. ⚡ **Acts** — creates files, writes code, reads files, summarizes text, runs commands
4. 🔒 **Asks** before doing anything destructive (Human-in-the-Loop)
5. 💾 **Remembers** every session in a local SQLite database

All of this runs through a sleek dark-mode UI with a real-time pipeline tracker, animated intent badges, and session management.

---

## 🎬 Demo

| Feature | Screenshot |
|---|---|
| Voice input → pipeline tracker → response | *(record your own demo)* |
| HITL confirmation dialog | *(record your own demo)* |
| Output files auto-appearing in sidebar | *(record your own demo)* |

> 🎙️ **Sample voice commands** are included in `samples/` — use them to test file upload instantly.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARIA Pipeline                                │
│                                                                     │
│   Audio / Text Input                                                │
│          │                                                          │
│   ┌──────▼──────┐                                                   │
│   │  STT Node   │  Groq Whisper-large-v3 → plain text transcript   │
│   └──────┬──────┘                                                   │
│          │ transcript                                               │
│   ┌──────▼──────┐        ┌──────────────┐                          │
│   │ Agent Node  │◄──────►│  Tools Node  │  LangGraph ReAct loop    │
│   │  (LLM +     │        │  (ToolNode)  │                          │
│   │   Tools)    │        └──────────────┘                          │
│   └──────┬──────┘                                                   │
│          │                                                          │
│    [HITL needed?]                                                   │
│     ├─ YES ──► interrupt() ──► UI confirmation dialog               │
│     │          Command(resume=True/False) ──► continue/cancel       │
│     └─ NO  ──► stream final response to UI                          │
│                                                                     │
│   Persistence: AsyncSqliteSaver (LangGraph checkpointer)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Graph Topology

```
START ──► stt ──► agent ◄──► tools
                   │
                  END
```

The **Agent node** is the single source of truth for:
- Tool selection & reasoning (LLM)
- Intent classification & accumulation
- HITL interrupt control via `interrupt()`

---

## 🛠️ Tools

| Tool | Intent Badge | Description |
|---|---|---|
| `create_file` | 📄 Create File | Creates text/config/note files |
| `write_code` | 💻 Write Code | Generates & saves source code (any language) |
| `read_file` | 📖 Read File | Reads existing files for analysis/debug |
| `summarize_text` | 📝 Summarize Text | Bullet-point summarization |
| `run_terminal` | ⚡ Run Command | Executes shell commands with 30s timeout |

**Compound commands** are fully supported — say *"Read my script, summarize it, then save the summary"* and ARIA chains all 3 tools automatically. The intent badge accumulates: `📖 Read File + 📝 Summarize Text + 📄 Create File`.

---

## 🔒 Human-in-the-Loop (HITL)

Powered by LangGraph's **`interrupt()`** function (0.4.x pattern — not the old `interrupt_before` node):

```
Agent decides to write_code → interrupt() called inside agent_node
                            ↓
           UI shows confirmation dialog with action summary
                            ↓
         User clicks ✅ Confirm  →  Command(resume=True)  → tool executes
         User clicks ✗ Cancel   →  Command(resume=False) → clean cancellation
                                   (returns AIMessage with no tool_calls,
                                    so the graph routes to END correctly)
```

> ⚠️ **Critical design note**: On cancellation, ARIA returns a fresh `AIMessage` (not the original response with `tool_calls`). This ensures `_should_continue()` routes to `END` instead of the tools node — preventing the cancelled action from executing anyway.

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
├── pyproject.toml         # uv/pip project definition
└── .env                   # API keys (never committed)
```

---

## 🚀 Quick Start

### 1 · Clone & install

```bash
git clone https://github.com/Pandharimaske/ai-voice-agent.git
cd ai-voice-agent

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### 2 · Configure API keys

```bash
cp .env.example .env   # or create .env manually
```

```env
# .env
GROQ_API_KEY=gsk_...          # Required — get from console.groq.com
OPENAI_API_KEY=sk-...         # Optional — only if using OpenAI provider
ANTHROPIC_API_KEY=sk-ant-...  # Optional — only if using Anthropic provider
```

### 3 · (Optional) Swap models

Edit `config.yaml` — no code changes needed:

```yaml
stt:
  provider: "groq"
  model: "whisper-large-v3"       # or "openai" + "whisper-1"

llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile" # or "openai"/"gpt-4o" or "ollama"/"llama3.2"
  temperature: 0.1

hitl:
  enabled: true   # set false to skip confirmations
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
| **Intent Badges** | Shows all tools used per turn — accumulates: `📖 Read File + 💻 Write Code` |
| **Voice Recording** | Click 🎙️ to record via mic with live waveform visualizer |
| **File Upload** | Click 📂 to upload any `.wav` / `.mp3` audio file |
| **HITL Dialog** | Confirms destructive actions before execution |
| **Session Sidebar** | Full history with one-click load & 🗑️ delete |
| **Output Files Panel** | Auto-refreshes after every tool run — shows files created |
| **Output Path Selector** | Switch / add output folders without restarting |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/process_text_stream` | Text → SSE stream of agent updates |
| `POST` | `/api/process_stream` | Audio upload → SSE stream |
| `POST` | `/api/transcribe` | Audio → transcript only |
| `POST` | `/api/confirm` | Resume HITL (`{thread_id, confirmed: bool}`) |
| `GET` | `/api/sessions` | List all saved sessions |
| `GET` | `/api/sessions/{id}` | Load a specific session |
| `DELETE` | `/api/sessions/{id}` | Delete a session from SQLite |
| `GET` | `/api/output_files` | List files in output folder |

All streaming endpoints return **Server-Sent Events (SSE)** with this payload per event:

```jsonc
{
  "thread_id": "uuid",
  "transcript": "what the user said",
  "detected_intent": "📖 Read File + 💻 Write Code",
  "action_taken": "  • write_code(filename='app.py', ...)",
  "messages": [
    { "role": "user",      "content": "...", "intent": "💻 Write Code" },
    { "role": "assistant", "content": "Done! Here's what I created..." }
  ],
  "is_interrupted": false,
  "interrupt_data": null,
  "output_path": "./output",
  "error": null
}
```

---

## 🧠 Technical Design Decisions

### `add_messages` Reducer
```python
# agent/state.py
class AgentState(TypedDict, total=False):
    messages: Annotated[List, add_messages]  # LG 0.4 — dedupes by message ID
```
Using `add_messages` from `langgraph.graph.message` instead of `operator.add` ensures proper ID-based deduplication when checkpoints are restored — no duplicate messages on resume.

### `interrupt()` instead of `interrupt_before`
```python
# agent/nodes.py — called INSIDE agent_node, not as a graph config option
confirmed = interrupt({"message": "...", "actions_summary": "...", "tool_names": [...]})
if not confirmed:
    return {"messages": [AIMessage(content="Cancelled.")]}  # no tool_calls!
```
The LangGraph 0.4 `interrupt()` function gives **code-level control** over exactly when and why a graph pauses — no static node list needed.

### LRU-cached LLM
```python
@functools.lru_cache(maxsize=1)
def _get_llm_with_tools():
    ...  # runs exactly once, thread-safe
```

### Intent Accumulation
Each ReAct loop pass merges its tool labels into the existing `detected_intent`:
```
Turn 1 pass: read_file  → "📖 Read File"
Turn 1 pass: write_code → "📖 Read File + 💻 Write Code"
Turn 2 starts          → reset to None, starts fresh
```

### FastAPI Lifespan
```python
@asynccontextmanager
async def lifespan(app):
    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as saver:
        init_graph(saver)
        yield
    # saver.close() called automatically
```
Replaces fragile global singletons — the SQLite connection is opened and closed cleanly with the app lifecycle.

---

## ⚙️ Supported Model Providers

| Provider | LLM Examples | STT |
|---|---|---|
| **Groq** ⭐ | `llama-3.3-70b-versatile`, `mixtral-8x7b` | `whisper-large-v3` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | `whisper-1` |
| **Anthropic** | `claude-3-5-sonnet-20241022` | — |
| **Ollama** | `llama3.2`, `mistral`, `codellama` | — |

Switch providers by editing **two lines** in `config.yaml` — no code changes.

---

## 🗺️ Roadmap

- [ ] Text-to-Speech output (read responses aloud)
- [ ] Streaming token-by-token response in chat
- [ ] Multi-file context (RAG over output folder)
- [ ] Plugin system for custom tools
- [ ] Docker compose for one-command deploy

---

## 🔭 Observability with LangSmith

ARIA ships with **zero-config LangSmith tracing** — LangChain and LangGraph automatically detect the env vars and send every run, tool call, and LLM invocation to your LangSmith dashboard.

### Setup (already done if your `.env` has these)

```env
LANGCHAIN_API_KEY=lsv2_pt_...      # from smith.langchain.com
LANGSMITH_TRACING_V2=true          # enables automatic tracing
LANGSMITH_PROJECT=AI-Voice-Agent   # your project name in LangSmith
```

No code changes needed — the moment the server starts, every pipeline run is visible at **https://smith.langchain.com** with:

| What you see | Details |
|---|---|
| **Run traces** | Full STT → Agent → Tools → Response chain |
| **LLM inputs/outputs** | Every prompt, tool call schema, and completion |
| **Latency breakdown** | Time spent in each node |
| **Token usage** | Per-run cost tracking |
| **Error traces** | Full stack trace on failures |
| **HITL interrupts** | Visible as checkpoint events |

> 💡 You can tag runs, add feedback scores, and set up evaluators directly from the dashboard — no extra instrumentation needed.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c6aff&height=100&section=footer" width="100%"/>

**Built with LangGraph · LangChain · FastAPI · Groq**

⭐ Star this repo if ARIA impressed you!

</div>
