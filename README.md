# ARIA — Audio Reasoning Intelligence Agent

> A voice-controlled local AI agent: speak a command → transcribe → classify intent → execute tools → see results.

---

## Demo

[![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red?style=for-the-badge)](YOUR_YOUTUBE_LINK)
[![Article](https://img.shields.io/badge/📝_Read_Article-Dev.to-black?style=for-the-badge)](YOUR_ARTICLE_LINK)

---

## Architecture

```
Audio Input (mic / file upload)
         │
    ┌────▼────┐
    │  STT    │  Groq Whisper API / OpenAI Whisper
    └────┬────┘
         │  transcript
    ┌────▼────┐
    │  Intent │  LLM + Pydantic structured output → IntentResult
    └────┬────┘
         │
   [HITL needed?]
    ├── YES → ┌────────┐  interrupt()   ┌─────────────┐
    │         │  HITL  │ ─────────────► │  Web UI JS  │  Confirm / Cancel
    │         └────┬───┘ ◄──────────── └─────────────┘
    └── NO   ──────┘
                   │
    ┌──────────────▼──────────┐
    │       Tool Router       │  compound commands supported
    ├── create_file           │
    ├── write_code            │
    ├── summarize             │
    └── general_chat          │
    └─────────────────────────┘
         │
    ┌────▼────┐
    │ Web UI  │  Transcript · Intent · Action · Result · Chat history
    └─────────┘
```

### Key Design Decisions

| Layer | Choice | Reason |
|-------|--------|--------|
| **STT** | Groq Whisper API | Local Whisper requires 4–8 GB VRAM; Groq provides ~300× realtime speed at zero cost |
| **LLM** | Groq (default) / OpenAI / Anthropic | Configurable via `config.yaml` — one-line swap |
| **Structured output** | Pydantic `IntentResult` + `.with_structured_output()` | Type-safe, validated intent extraction |
| **Orchestration** | LangGraph | State machine with checkpointing, clean HITL interrupt |
| **Checkpointer** | SQLite | Zero-dependency persistent storage for HITL state |
| **UI** | Custom Web Frontend | Premium HTML/CSS/JS + FastAPI backend for full control over aesthetics |

---

## Features

- 🎙️ **Dual audio input** — microphone or file upload (.wav, .mp3, .m4a, ...)
- 🧠 **Intent classification** — powered by Pydantic structured output
- 📋 **Compound commands** — "write a sort function AND save it to sort.py" works natively
- ⚡ **Human-in-the-Loop** — confirmation dialog before any filesystem operation
- 💾 **SQLite checkpointing** — HITL state survives across UI interactions
- 🔁 **Session memory** — chat history maintained across turns
- 🛡️ **Sandboxed output** — ALL file writes go to `./output/` (configurable)
- 🔧 **One-config swap** — change `config.yaml` to switch provider/model

### Supported Intents

| Intent | Example command |
|--------|----------------|
| `create_file` | "Create a file called notes.txt" |
| `write_code` | "Write a Python retry decorator and save it to retry.py" |
| `summarize` | "Summarize this: [text]" |
| `general_chat` | "What's the difference between RAG and fine-tuning?" |
| **Compound** | "Write a Fibonacci function AND save it to fib.py AND create a README" |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ARIA
cd ARIA
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your API key(s)
```

```bash
# Edit config.yaml to choose your provider/model
# Default uses Groq (free tier, fastest)
```

**Minimum: set `GROQ_API_KEY` in `.env`** — everything else works out of the box.

### 3. Run

```bash
python server.py
# Opens at http://localhost:8000
```

---

## Configuration Reference

All settings live in `config.yaml`:

```yaml
stt:
  provider: "groq"           # groq | openai
  model: "whisper-large-v3"

llm:
  provider: "groq"           # groq | openai | anthropic
  model: "llama-3.3-70b-versatile"
  temperature: 0.1

output:
  folder: "./output"         # ← change to any path

hitl:
  enabled: true              # set false to auto-confirm all
  require_confirmation_for:
    - "create_file"
    - "write_code"

memory:
  db_path: "./data/checkpoints.db"
  session_history_limit: 20
```

### Switching providers

```yaml
# To use OpenAI:
stt:
  provider: "openai"
  model: "whisper-1"
llm:
  provider: "openai"
  model: "gpt-4o-mini"

# To use Anthropic for LLM:
llm:
  provider: "anthropic"
  model: "claude-3-5-haiku-20241022"
```

---

## Hardware Note

ARIA uses API-based STT (Groq Whisper) rather than running Whisper locally because:

- `whisper-large-v3` requires ~6 GB VRAM to run efficiently
- Local CPU inference on `whisper-small` takes 30–60 seconds per clip
- Groq's hosted Whisper runs at ~300× realtime and is **free** on the developer tier
- Transcription quality is identical to running the model locally

If you want to run fully offline, replace `GroqSTT` in `agent/stt.py` with `faster-whisper` — see the comments in that file.

---

## Project Structure

```
ARIA/
├── config.yaml           # ← Start here
├── .env                  # API keys (git-ignored)
├── server.py             # FastAPI backend (entry point)
├── ui/                   # Premium Web Frontend
│   ├── index.html        # Structure
│   ├── style.css         # Premium design
│   ├── app.js            # Frontend logic
│   └── recorder.js       # Mic recording logic
├── agent/
│   ├── state.py          # Pydantic models + AgentState
│   ├── stt.py            # STT abstraction (Groq / OpenAI)
│   ├── intent.py         # LLM factory + intent chain
│   ├── tools.py          # create_file, write_code, summarize, chat
│   ├── nodes.py          # LangGraph node functions
│   └── graph.py          # Graph assembly + SQLite checkpointer
├── config/
│   ├── settings.py       # Pydantic settings loader
│   └── logging_config.py # Centralized logging setup
├── output/               # ALL file writes go here (sandboxed)
└── data/
    └── checkpoints.db    # LangGraph SQLite checkpointer
```

---

## Bonus Features Implemented

- [x] **Compound commands** — multiple intents in one utterance
- [x] **Human-in-the-Loop** — confirmation before file ops
- [x] **Graceful degradation** — intent fallback to chat, error display
- [x] **Session memory** — chat history across turns
- [x] **SQLite checkpointing** — HITL state persists across UI calls

---

## License

MIT
