"""FastAPI application entry point."""

import json
import os
import tempfile
import time
import shutil
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, verification
from app.config import SESSIONS_DIR, UPLOAD_DIR, CHROMA_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _cleanup_stale_sessions():
    """Delete session files, uploads, vectordb dirs, and reports older than SESSION_TTL.
    Also recover zombie sessions stuck in 'processing' or 'queued' state."""
    now = time.time()
    cleaned = 0
    zombie_fixed = 0

    # 1) Clean stale session JSON files
    for f in SESSIONS_DIR.glob("*.json"):
        if now - f.stat().st_mtime > SESSION_TTL_SECONDS:
            f.unlink(missing_ok=True)
            cleaned += 1

    # 2) Clean stale uploaded PDFs
    for f in UPLOAD_DIR.glob("*.pdf"):
        if now - f.stat().st_mtime > SESSION_TTL_SECONDS:
            f.unlink(missing_ok=True)
            cleaned += 1

    # 3) Clean stale ChromaDB vector store directories
    if CHROMA_DIR.exists():
        for d in CHROMA_DIR.iterdir():
            if d.is_dir():
                try:
                    if now - d.stat().st_mtime > SESSION_TTL_SECONDS:
                        shutil.rmtree(d, ignore_errors=True)
                        cleaned += 1
                except OSError:
                    pass

    # 4) Clean stale report files
    if REPORTS_DIR.exists():
        for f in REPORTS_DIR.glob("*_report.*"):
            if now - f.stat().st_mtime > SESSION_TTL_SECONDS:
                f.unlink(missing_ok=True)
                cleaned += 1

    # 5) Recover zombie sessions — any session stuck in processing/queued
    #    at startup must have been interrupted by a crash/restart
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") in ("processing", "queued"):
                data["status"] = "failed"
                data["error"] = "Server restarted during analysis. Please re-run."
                # Atomic write: temp file + os.replace
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(SESSIONS_DIR), suffix=".tmp", prefix="zombie_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                        tmp.write(json.dumps(data, indent=2, default=str, ensure_ascii=False))
                    os.replace(tmp_path, str(f))
                except BaseException:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
                zombie_fixed += 1
        except (json.JSONDecodeError, OSError):
            pass

    if cleaned:
        logger.info(f"Startup cleanup: removed {cleaned} stale file(s)/dir(s) older than 24h")
    if zombie_fixed:
        logger.info(f"Startup cleanup: recovered {zombie_fixed} zombie session(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: cleanup stale sessions on startup."""
    _cleanup_stale_sessions()
    yield


app = FastAPI(
    title="HATAD Land Intelligence Platform",
    description="The Bloomberg Terminal for Indian Land Due Diligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(verification.router, prefix="/api/analyze", tags=["Analysis"])


@app.get("/api/health")
async def health():
    return {"status": "operational", "platform": "HATAD Land Intelligence"}
