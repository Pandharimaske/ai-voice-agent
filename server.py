"""
server.py
─────────
ARIA — FastAPI backend.
Optimized for streaming status updates and total absence of TTS.

Run:  python server.py
UI:   http://localhost:8000
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import (
    astream_pipeline,
    astream_pipeline_text,
    astream_resume_pipeline,
    get_thread_history,
)
from config.logging_config import logger
from config.settings import settings

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConfirmRequest(BaseModel):
    thread_id: str
    confirmed: bool


class TextProcessRequest(BaseModel):
    text: str
    chat_history: list = []
    thread_id: str | None = None
    output_path: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _save_upload(audio: UploadFile) -> str:
    """Save an uploaded audio file to a temp path; caller must unlink."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await audio.read())
    tmp.flush()
    tmp.close()
    return tmp.name


async def _sse_generator(stream_iterator):
    """Wrap an async generator as SSE events."""
    try:
        async for event in stream_iterator:
            # Detect interrupt state
            is_interrupted = event.get("confirmed") is None and event.get("intent_result") is not None
            
            # Enrich the event for the UI
            event["is_interrupted"] = is_interrupted
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(str(UI_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ARIA"}


# ── Streaming endpoints (SSE) ─────────────────────────────────────────────────

@app.post("/api/process_stream")
async def process_audio_stream(
    audio: UploadFile = File(...),
    chat_history: str = Form(default="[]"),
):
    """Streaming audio processing — emits SSE events."""
    try:
        history = json.loads(chat_history)
    except Exception:
        history = []

    tmp_path = await _save_upload(audio)
    thread_id = str(uuid.uuid4())

    async def stream_and_cleanup():
        try:
            async for chunk in _sse_generator(astream_pipeline(tmp_path, thread_id, history)):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return StreamingResponse(stream_and_cleanup(), media_type="text/event-stream")


@app.post("/api/process_text_stream")
async def process_text_stream(body: TextProcessRequest):
    """Streaming text processing."""
    thread_id = body.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_generator(astream_pipeline_text(body.text, thread_id, body.chat_history, body.output_path)),
        media_type="text/event-stream",
    )


@app.post("/api/confirm_stream")
async def confirm_stream(body: ConfirmRequest):
    """Streaming HITL confirmation."""
    return StreamingResponse(
        _sse_generator(astream_resume_pipeline(body.thread_id, body.confirmed)),
        media_type="text/event-stream",
    )


# ── STT only ──────────────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """STT only — used for live voice transcription in UI."""
    tmp_path = await _save_upload(audio)
    try:
        from agent.stt import get_stt
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(tmp_path)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Session history ───────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all past thread IDs."""
    import sqlite3
    db_path = str(settings.db_path)
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
        sessions = [{"thread_id": row[0]} for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception:
        return []


@app.get("/api/sessions/{thread_id}")
async def get_session(thread_id: str):
    """Return the full state for a specific thread."""
    try:
        return get_thread_history(thread_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
