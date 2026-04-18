\"\"\"
server.py
─────────
ARIA — FastAPI backend.
Replaces app.py entirely. Serves the UI and exposes the pipeline API.

Run:  python server.py
UI:   http://localhost:8000
\"\"\"

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import (
    astream_pipeline, 
    astream_pipeline_text, 
    astream_resume_pipeline, 
    get_thread_history, 
    get_graph
)
from config.logging_config import logger
from config.settings import settings

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

# ── Streaming helper ──────────────────────────────────────────────────────────

async def event_generator(stream_iterator):
    \"\"\"Yields events from the graph stream as SSE.\"\"\"
    try:
        async for event in stream_iterator:
            # event is the state snapshot at each step
            snapshot = {
                \"transcript\": event.get(\"transcript\", \"\"),
                \"intent_result\": event.get(\"intent_result\"),
                \"tool_results\": event.get(\"tool_results\", []),
                \"messages\": event.get(\"messages\", []),
                \"error\": event.get(\"error\"),
                \"is_interrupted\": event.get(\"confirmed\") is None and event.get(\"intent_result\") is not None, 
            }
            yield f\"data: {json.dumps(snapshot)}\n\n\"
    except Exception as e:
        logger.error(f\"Stream error: {str(e)}\")
        yield f\"data: {json.dumps({'error': str(e)})}\n\n\"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(\"/\")
async def serve_ui():
    \"\"\"Serve the main UI.\"\"\"
    return FileResponse(str(UI_DIR / \"index.html\"))


@app.get(\"/health\")
async def health():
    return {\"status\": \"ok\", \"service\": \"ARIA\"}


@app.post(\"/api/process_stream\")
async def process_stream(
    audio: UploadFile = File(...),
    chat_history: str = Form(default=\"[]\"),
):
    \"\"\"Streaming version of process_audio.\"\"\"
    try:
        history = json.loads(chat_history)
    except:
        history = []

    suffix = Path(audio.filename or \"audio.wav\").suffix or \".wav\"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await audio.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    thread_id = str(uuid.uuid4())
    
    async def cleanup_and_stream():
        try:
            stream = astream_pipeline(tmp.name, thread_id, history)
            async for data in event_generator(stream):
                yield data
        finally:
            try: os.unlink(tmp.name)
            except: pass

    return StreamingResponse(cleanup_and_stream(), media_type=\"text/event-stream\")


@app.post(\"/api/process_text_stream\")
async def process_text_stream(body: TextProcessRequest):
    \"\"\"Streaming version of process_text.\"\"\"
    thread_id = body.thread_id or str(uuid.uuid4())
    stream = astream_pipeline_text(body.text, thread_id, body.chat_history, body.output_path)
    return StreamingResponse(event_generator(stream), media_type=\"text/event-stream\")


@app.post(\"/api/confirm_stream\")
async def confirm_stream(body: ConfirmRequest):
    \"\"\"Streaming version of confirm_action.\"\"\"
    stream = astream_resume_pipeline(body.thread_id, body.confirmed)
    return StreamingResponse(event_generator(stream), media_type=\"text/event-stream\")


@app.post(\"/api/transcribe\")
async def transcribe_audio(audio: UploadFile = File(...)):
    \"\"\"Transcribe audio file only.\"\"\"
    suffix = Path(audio.filename or \"audio.wav\").suffix or \".wav\"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        from agent.stt import get_stt
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(tmp.name)
        return {\"transcript\": transcript}
    except Exception as e:
        logger.error(f\"Transcription failed: {str(e)}\")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp.name)
        except: pass


@app.get(\"/api/sessions\")
async def list_sessions():
    \"\"\"List all unique thread IDs from the checkpointer database.\"\"\"
    db_path = str(settings.db_path)
    if not os.path.exists(db_path):
        return []
    
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(\"SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC\")
        sessions = [{\"thread_id\": row[0]} for row in cursor.fetchall()]
        conn.close()
        return sessions
    except:
        return []


@app.get(\"/api/sessions/{thread_id}\")
async def get_session(thread_id: str):
    \"\"\"Retrieve history and state for a specific thread.\"\"\"
    try:
        result = get_thread_history(thread_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f\"Session {thread_id} not found: {str(e)}\")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == \"__main__\":
    uvicorn.run(
        \"server:app\",
        host=\"0.0.0.0\",
        port=8000,
        reload=True,
        reload_dirs=[\".\"],
    )
