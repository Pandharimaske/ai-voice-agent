"""
server.py
─────────
ARIA — FastAPI backend.
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
    resume_pipeline,
    run_pipeline,
    run_pipeline_text,
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


class PipelineResponse(BaseModel):
    thread_id: str
    transcript: str
    intent_result: dict | None
    tool_results: list
    messages: list
    error: str | None
    is_interrupted: bool
    interrupt_data: dict | None
    confirmed: bool | None
    output_path: str | None = None


class TextProcessRequest(BaseModel):
    text: str
    chat_history: list = []
    thread_id: str | None = None
    output_path: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_to_response(result: dict, thread_id: str) -> PipelineResponse:
    """Convert a _read_state dict to a PipelineResponse."""
    return PipelineResponse(
        thread_id=thread_id,
        transcript=result.get("transcript", ""),
        intent_result=result.get("intent_result"),
        tool_results=result.get("tool_results", []),
        messages=result.get("messages", []),
        error=result.get("error"),
        is_interrupted=result.get("is_interrupted", False),
        interrupt_data=result.get("interrupt_data"),
        confirmed=result.get("confirmed"),
        output_path=result.get("output_path"),
    )


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


# ── /api/process  (used by the frontend) ─────────────────────────────────────

@app.post("/api/process", response_model=PipelineResponse)
async def process_audio(
    audio: UploadFile = File(...),
    chat_history: str = Form(default="[]"),
):
    """
    Receive audio, run the full pipeline synchronously.
    Returns either a complete result or an interrupted state awaiting HITL.
    """
    try:
        history = json.loads(chat_history)
    except Exception:
        history = []

    tmp_path = await _save_upload(audio)
    thread_id = str(uuid.uuid4())

    try:
        logger.info(f"Process audio — thread {thread_id[:8]}")
        result = run_pipeline(tmp_path, thread_id, history)
        return _result_to_response(result, thread_id)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── /api/confirm  (used by the frontend) ─────────────────────────────────────

@app.post("/api/confirm", response_model=PipelineResponse)
async def confirm_action(body: ConfirmRequest):
    """Resume a paused (HITL-interrupted) pipeline."""
    try:
        logger.info(f"HITL resume — thread {body.thread_id[:8]} confirmed={body.confirmed}")
        result = resume_pipeline(body.thread_id, body.confirmed)
        return _result_to_response(result, body.thread_id)
    except Exception as e:
        logger.error(f"Resume error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/process_text  (dev/testing convenience) ─────────────────────────────

@app.post("/api/process_text", response_model=PipelineResponse)
async def process_text(body: TextProcessRequest):
    """Run the pipeline on direct text input (skips STT)."""
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        logger.info(f"Process text — thread {thread_id[:8]}")
        result = run_pipeline_text(body.text, thread_id, body.chat_history, body.output_path)
        return _result_to_response(result, thread_id)
    except Exception as e:
        logger.error(f"Text pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming endpoints (SSE — bonus, not used by default frontend) ───────────

@app.post("/api/process_stream")
async def process_audio_stream(
    audio: UploadFile = File(...),
    chat_history: str = Form(default="[]"),
):
    """Streaming version — emits SSE events as each pipeline stage completes."""
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
    thread_id = body.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_generator(astream_pipeline_text(body.text, thread_id, body.chat_history, body.output_path)),
        media_type="text/event-stream",
    )


@app.post("/api/confirm_stream")
async def confirm_stream(body: ConfirmRequest):
    return StreamingResponse(
        _sse_generator(astream_resume_pipeline(body.thread_id, body.confirmed)),
        media_type="text/event-stream",
    )


# ── Transcribe only ───────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """STT only — useful for testing transcription quality."""
    tmp_path = await _save_upload(audio)
    try:
        from agent.stt import get_stt
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(tmp_path)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Session history ───────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all past thread IDs from the SQLite checkpointer."""
    import sqlite3
    db_path = str(settings.db_path)
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
        )
        sessions = [{"thread_id": row[0]} for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception:
        return []


@app.get("/api/sessions/{thread_id}")
async def get_session(thread_id: str):
    """Return the full state for a specific past session."""
    try:
        return get_thread_history(thread_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── TTS (optional) ────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/api/tts")
async def text_to_speech(body: TTSRequest):
    """Convert text to speech (requires: pip install gtts)."""
    try:
        from gtts import gTTS
        import io
        clean = re.sub(r"```[\s\S]*?```", "", body.text)
        clean = re.sub(r"[`*#]", "", clean).strip()
        tts = gTTS(text=clean, lang=body.lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except ImportError:
        raise HTTPException(status_code=501, detail="Install gtts: pip install gtts")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
