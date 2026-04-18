"""
server.py
─────────
ARIA — FastAPI backend.
Replaces app.py entirely. Serves the UI and exposes the pipeline API.

Run:  python server.py
UI:   http://localhost:8000
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import resume_pipeline, run_pipeline
from config.logging_config import logger

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from ui/
UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

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

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the main UI."""
    return FileResponse(str(UI_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ARIA"}


@app.post("/api/process", response_model=PipelineResponse)
async def process_audio(
    audio: UploadFile = File(...),
    chat_history: str = Form(default="[]"),
):
    """
    Receive an audio file, run the full pipeline.
    Returns either:
    - Complete result (is_interrupted=False)
    - Interrupt state (is_interrupted=True) waiting for HITL confirmation
    """
    import json

    # Parse chat history from form
    try:
        history = json.loads(chat_history)
    except Exception:
        history = []

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        thread_id = str(uuid.uuid4())
        logger.info(f"Processing audio for thread: {thread_id[:8]}")
        result = run_pipeline(tmp.name, thread_id, history)
        logger.info(f"Audio pipeline complete: {thread_id[:8]}")
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file only, without running the full pipeline."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        logger.info(f"Transcribing audio: {audio.filename}")
        from agent.stt import get_stt
        from config.settings import settings
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(tmp.name)
        logger.info(f"Transcription successful: '{transcript[:50]}...'")
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@app.post("/api/process_text", response_model=PipelineResponse)
async def process_text(body: TextProcessRequest):
    """Run the pipeline using direct text input."""
    thread_id = body.thread_id or str(uuid.uuid4())
    thread_id = body.thread_id or str(uuid.uuid4())
    logger.info(f"Processing text for thread {thread_id[:8]}... | Output Path: {body.output_path or 'default'}")
    try:
        from agent.graph import run_pipeline_text
        result = run_pipeline_text(body.text, thread_id, body.chat_history, body.output_path)
        logger.info(f"Text pipeline finished: {thread_id[:8]}")
    except ImportError:
        logger.error("ImportError: Text processing not implemented in graph.py")
        raise HTTPException(status_code=501, detail="Text processing not implemented in graph.py")
    except Exception as e:
        logger.error(f"Text processing error in thread {thread_id[:8]}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return PipelineResponse(
        thread_id=thread_id,
        transcript=body.text,
        intent_result=result.get("intent_result"),
        tool_results=result.get("tool_results", []),
        messages=result.get("messages", []),
        output_path=result.get("output_path"),
        error=result.get("error"),
        is_interrupted=result.get("is_interrupted", False),
        interrupt_data=result.get("interrupt_data"),
        confirmed=result.get("confirmed"),
    )


@app.post("/api/confirm", response_model=PipelineResponse)
async def confirm_action(body: ConfirmRequest):
    """
    Resume a paused (HITL-interrupted) pipeline.
    Pass confirmed=true to execute, false to cancel.
    """
    try:
        result = resume_pipeline(body.thread_id, body.confirmed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PipelineResponse(
        thread_id=body.thread_id,
        transcript=result.get("transcript", ""),
        intent_result=result.get("intent_result"),
        tool_results=result.get("tool_results", []),
        messages=result.get("messages", []),
        output_path=result.get("output_path"),
        error=result.get("error"),
        is_interrupted=result.get("is_interrupted", False),
        interrupt_data=result.get("interrupt_data"),
        confirmed=result.get("confirmed"),
    )


@app.post("/api/tts")
async def text_to_speech(body: TTSRequest):
    """Convert text to speech using gTTS."""
    from gtts import gTTS
    import io
    from fastapi.responses import StreamingResponse
    
    logger.info(f"Generating TTS for: '{body.text[:50]}...'")
    try:
        # Clean text of markdown before speaking
        clean_text = re.sub(r"```[\s\S]*?```", "", body.text) # Remove code blocks
        clean_text = re.sub(r"[`*#]", "", clean_text)       # Remove markdown chars
        
        tts = gTTS(text=clean_text, lang=body.lang)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all unique thread IDs from the checkpointer database."""
    import sqlite3
    from config.settings import settings
    db_path = str(settings.db_path)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Query unique thread IDs and their latest checkpoint ID (as a proxy for recency)
        cursor.execute("""
            SELECT thread_id, MAX(checkpoint_id) 
            FROM checkpoints 
            GROUP BY thread_id 
            ORDER BY MAX(checkpoint_id) DESC
        """)
        sessions = [{"thread_id": row[0]} for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        return []

@app.get("/api/sessions/{thread_id}", response_model=PipelineResponse)
async def get_session(thread_id: str):
    """Get the full state and history for a specific session."""
    try:
        from agent.graph import get_thread_history
        result = get_thread_history(thread_id)
        return PipelineResponse(
            thread_id=thread_id,
            transcript=result.get("transcript", ""),
            intent_result=result.get("intent_result"),
            tool_results=result.get("tool_results", []),
            messages=result.get("messages", []),
            output_path=result.get("output_path"),
            error=result.get("error"),
            is_interrupted=result.get("is_interrupted", False),
            interrupt_data=result.get("interrupt_data"),
            confirmed=result.get("confirmed"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
    )
