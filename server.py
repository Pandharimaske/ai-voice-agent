"""
server.py
─────────
ARIA — FastAPI backend (LangGraph 0.4.x + LangChain 0.3.x).

Run:  uv run python server.py
UI:   http://localhost:8000

Architecture:
  - FastAPI lifespan manages the AsyncSqliteSaver lifecycle cleanly.
  - Graph is compiled once at startup (no lazy-init globals).
  - SSE streaming for all agent interactions.
  - HITL resume uses Command(resume=True/False) via /api/confirm endpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
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
    init_graph,
)
from config.logging_config import logger
from config.settings import settings


# ── Lifespan: initialise graph + checkpointer once ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create the AsyncSqliteSaver and compile the graph.
    Shutdown: close the saver connection cleanly.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    logger.info("ARIA: Starting up…")
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as saver:
        init_graph(saver)
        logger.info(f"ARIA: Graph ready. DB → {settings.db_path}")
        yield
    logger.info("ARIA: Shutdown complete.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA Voice Agent", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ── Pydantic request models ────────────────────────────────────────────────────

class ConfirmRequest(BaseModel):
    thread_id: str
    confirmed: bool


class TextRequest(BaseModel):
    text: str
    chat_history: list = []
    thread_id: str | None = None
    output_path: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _save_upload(audio: UploadFile) -> str:
    """Save an uploaded audio file to a temp path."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await audio.read())
    tmp.flush()
    tmp.close()
    return tmp.name


async def _sse_gen(stream_gen, thread_id: str):
    """Wrap an async generator as SSE events."""
    try:
        async for event in stream_gen:
            event["thread_id"] = thread_id
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        logger.error(f"SSE stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(str(UI_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ARIA", "version": "3.0.0"}


# ── Streaming endpoints ────────────────────────────────────────────────────────

@app.post("/api/process_stream")
async def process_audio_stream(
    audio: UploadFile = File(...),
    thread_id: str = Form(default=""),
    output_path: str = Form(default=""),
    chat_history: str = Form(default="[]"),
):
    """Voice input → SSE stream of agent state updates."""
    tid = thread_id.strip() or str(uuid.uuid4())
    op  = output_path.strip() or None
    try:
        history = json.loads(chat_history)
    except Exception:
        history = []

    tmp_path = await _save_upload(audio)

    async def _stream_and_cleanup():
        try:
            async for chunk in _sse_gen(
                astream_pipeline(tmp_path, tid, history, op), tid
            ):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return StreamingResponse(_stream_and_cleanup(), media_type="text/event-stream")


@app.post("/api/process_text_stream")
async def process_text_stream(body: TextRequest):
    """Text input → SSE stream of agent state updates."""
    tid = body.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_gen(
            astream_pipeline_text(body.text, tid, body.chat_history, body.output_path),
            tid,
        ),
        media_type="text/event-stream",
    )


@app.post("/api/confirm")
async def confirm_action(body: ConfirmRequest):
    """
    Resume a graph paused by interrupt().
    Uses Command(resume=True/False) — the LangGraph 0.4 HITL resume pattern.
    """
    try:
        result = await astream_resume_pipeline(body.thread_id, body.confirmed)
        result["thread_id"] = body.thread_id
        return result
    except Exception as e:
        logger.error(f"Confirm error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── STT-only endpoint (for live voice preview in UI) ─────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio → return text only, without running the agent."""
    tmp_path = await _save_upload(audio)
    try:
        from agent.stt import get_stt
        api_key = settings.api_key_for(settings.stt.provider)
        stt = get_stt(settings.stt.provider, settings.stt.model, api_key)
        transcript = stt.transcribe(tmp_path)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Session management ─────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all past thread IDs with their last-updated timestamp."""
    import sqlite3
    db_path = str(settings.db_path)
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT thread_id, MAX(checkpoint_id) as last_id "
            "FROM checkpoints GROUP BY thread_id ORDER BY last_id DESC"
        )
        sessions = [{"thread_id": row[0]} for row in cur.fetchall()]
        conn.close()
        return sessions
    except Exception:
        return []


@app.get("/api/sessions/{thread_id}")
async def get_session(thread_id: str):
    """Return full serialised state for a specific thread."""
    try:
        return await get_thread_history(thread_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    """Delete all checkpoints for a thread from the SQLite DB."""
    import sqlite3
    db_path = str(settings.db_path)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="No database found.")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Discover checkpoint tables dynamically (schema differs across LG versions)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'checkpoint%'")
        tables = [row[0] for row in cur.fetchall()]

        total_deleted = 0
        for table in tables:
            cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            total_deleted += cur.rowcount

        conn.commit()
        conn.close()
        logger.info(f"Session deleted: {thread_id} ({total_deleted} rows across {tables})")
        return {"deleted": thread_id, "rows": total_deleted}
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Output folder browser ──────────────────────────────────────────────────────

@app.get("/api/output_files")
async def list_output_files(folder: str = "./output"):
    """Return a list of files in the given output folder."""
    try:
        target = Path(folder).resolve()
        # Safety: must be under cwd
        cwd = Path(".").resolve()
        if not str(target).startswith(str(cwd)):
            raise HTTPException(status_code=400, detail="Folder outside workspace.")
        if not target.exists():
            return []
        files = []
        for f in sorted(target.iterdir()):
            if f.is_file():
                files.append({
                    "name":     f.name,
                    "size":     f.stat().st_size,
                    "modified": f.stat().st_mtime,
                })
        return files
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
