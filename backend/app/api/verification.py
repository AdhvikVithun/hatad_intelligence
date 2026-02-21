"""Analysis/verification endpoints with SSE progress streaming."""

import json
import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import UPLOAD_DIR, SESSIONS_DIR, REPORTS_DIR, MAX_CONCURRENT_ANALYSES
from app.pipeline.orchestrator import run_analysis, AnalysisSession
from app.pipeline.llm_client import check_ollama_status
from app.reports.generator import generate_pdf_report

router = APIRouter()
logger = logging.getLogger(__name__)

# Semaphore limits how many analysis pipelines run concurrently
_analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)


class AnalyzeRequest(BaseModel):
    filenames: list[str]


@router.post("/start")
async def start_analysis(request: AnalyzeRequest):
    """Start analysis on uploaded documents. Returns session_id.
    
    Use /api/analyze/{session_id}/stream for SSE progress updates.
    """
    # Validate files exist
    file_paths = []
    for fn in request.filenames:
        fp = UPLOAD_DIR / fn
        if not fp.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {fn}")
        file_paths.append(fp)

    if not file_paths:
        raise HTTPException(status_code=400, detail="No files specified")

    # Run analysis synchronously but stream results
    # We'll store the session and return its ID
    session = AnalysisSession()
    session.status = "queued"
    session.save()

    # Start analysis in background task
    asyncio.create_task(_run_analysis_bg(session.session_id, file_paths))

    return {
        "session_id": session.session_id,
        "status": "queued",
        "message": f"Analysis started for {len(file_paths)} document(s)",
    }


async def _run_analysis_bg(session_id: str, file_paths: list[Path]):
    """Background task to run the full analysis pipeline."""
    async with _analysis_semaphore:
        await _run_analysis_inner(session_id, file_paths)


async def _run_analysis_inner(session_id: str, file_paths: list[Path]):
    """Inner analysis runner (held under semaphore)."""
    try:
        last_update = None
        async for update in run_analysis(file_paths, session_id=session_id):
            last_update = update
            # Updates are saved to session file by orchestrator
    except Exception as e:
        logger.exception(f"Analysis pipeline failed for session {session_id}")
        # Load session, mark as failed
        try:
            session = AnalysisSession.load(session_id)
            session.status = "failed"
            session.error = str(e)
            session.save()
        except Exception:
            logger.error(f"Failed to save error state for session {session_id}")


@router.get("/{session_id}/stream")
async def stream_analysis(session_id: str):
    """SSE endpoint for streaming analysis progress."""
    
    async def event_stream():
        last_progress_count = 0
        poll_count = 0
        max_polls = 3600  # 60 min timeout (1 poll/sec) — large docs take 30-45 min
        heartbeat_interval = 15  # Send keepalive every 15s to prevent proxy/browser disconnect

        while poll_count < max_polls:
            try:
                session = AnalysisSession.load(session_id)
            except FileNotFoundError:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                return

            # Send new progress entries
            progress = session.progress
            if len(progress) > last_progress_count:
                for entry in progress[last_progress_count:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                last_progress_count = len(progress)
            elif poll_count % heartbeat_interval == 0 and poll_count > 0:
                # Keepalive heartbeat — prevents proxy/browser from dropping the SSE connection
                yield f": heartbeat {poll_count}s\n\n"

            # Check if done
            if session.status in ("completed", "failed"):
                final = {
                    "stage": "final",
                    "status": session.status,
                    "session_id": session.session_id,
                    "risk_score": session.risk_score,
                    "risk_band": session.risk_band,
                    "error": session.error,
                }
                yield f"data: {json.dumps(final)}\n\n"
                return

            await asyncio.sleep(1)
            poll_count += 1

        yield f"data: {json.dumps({'error': 'Timeout waiting for analysis (60 min limit)'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{session_id}/status")
async def get_analysis_status(session_id: str):
    """Get current status of an analysis session."""
    try:
        session = AnalysisSession.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.get("/{session_id}/results")
async def get_analysis_results(session_id: str):
    """Get full analysis results including verification and extracted data."""
    try:
        session = AnalysisSession.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Current status: {session.status}",
        )

    result = session.to_dict()
    if session.status == "failed":
        result["incomplete"] = True
    return result


@router.get("/{session_id}/memory-bank")
async def get_memory_bank(session_id: str):
    """Get the cross-document memory bank for a session."""
    try:
        session = AnalysisSession.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    mb = session.memory_bank
    if not mb:
        raise HTTPException(status_code=400, detail="Memory bank not yet populated")
    return mb


@router.get("/{session_id}/report/pdf")
async def download_pdf_report(session_id: str):
    """Generate and download PDF report for completed analysis."""
    try:
        session = AnalysisSession.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Current status: {session.status}",
        )

    import asyncio
    pdf_path = await asyncio.to_thread(generate_pdf_report, session.to_dict())
    
    return FileResponse(
        path=str(pdf_path),
        filename=f"HATAD_Report_{session_id}.pdf",
        media_type="application/pdf",
    )


@router.get("/{session_id}/report/html")
async def get_html_report(session_id: str):
    """Get HTML version of the report."""
    html_path = REPORTS_DIR / f"{session_id}_report.html"
    if not html_path.exists():
        # Generate it
        try:
            session = AnalysisSession.load(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        import asyncio
        await asyncio.to_thread(generate_pdf_report, session.to_dict())
    
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="Report not generated")


@router.get("/health/llm")
async def check_llm():
    """Check if Ollama LLM is available."""
    return await check_ollama_status()


@router.get("/sessions")
async def list_sessions():
    """List all analysis sessions."""
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "status": data.get("status"),
                "risk_score": data.get("risk_score"),
                "risk_band": data.get("risk_band"),
                "created_at": data.get("created_at"),
                "documents": [d.get("filename") for d in data.get("documents", [])],
            })
        except Exception:
            continue
    
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return {"sessions": sessions}


@router.get("/health/boot")
async def boot_check():
    """Comprehensive system health check for the bootloader.

    Returns a dict of subsystem statuses: 'ok', 'degraded', or 'fail'.
    """
    import shutil
    import httpx
    from app.config import (
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        EMBED_MODEL,
        SARVAM_ENABLED,
        UPLOAD_DIR,
        SESSIONS_DIR,
        REPORTS_DIR,
    )

    checks: dict[str, dict] = {}

    # 1. Backend API — if we got here, it's alive
    checks["backend"] = {"status": "ok", "label": "Backend API"}

    # 2. File system — writable temp dirs
    try:
        for d in [UPLOAD_DIR, SESSIONS_DIR, REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".boot_probe"
            test_file.write_text("ok")
            test_file.unlink()
        disk = shutil.disk_usage(str(UPLOAD_DIR))
        free_gb = disk.free / (1024 ** 3)
        checks["filesystem"] = {
            "status": "ok" if free_gb > 1 else "degraded",
            "label": "File System",
            "free_gb": round(free_gb, 1),
        }
        if free_gb <= 1:
            checks["filesystem"]["message"] = f"Low disk space: {free_gb:.1f} GB remaining"
    except Exception as e:
        checks["filesystem"] = {"status": "fail", "label": "File System", "message": str(e)}

    # 3. Ollama — is it running?
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            checks["ollama"] = {"status": "ok", "label": "Ollama Runtime", "models": len(models)}
    except Exception as e:
        checks["ollama"] = {
            "status": "fail",
            "label": "Ollama Runtime",
            "message": f"Cannot reach Ollama at {OLLAMA_BASE_URL}",
        }

    # 4. Reasoning model loaded?
    if checks["ollama"]["status"] == "ok":
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(OLLAMA_MODEL in n for n in models)
                checks["reasoning_model"] = {
                    "status": "ok" if available else "fail",
                    "label": "Reasoning Model",
                    "model": OLLAMA_MODEL,
                }
                if not available:
                    checks["reasoning_model"]["message"] = (
                        f"Model '{OLLAMA_MODEL}' not found. "
                        f"Run: ollama pull {OLLAMA_MODEL}"
                    )
        except Exception:
            checks["reasoning_model"] = {
                "status": "fail",
                "label": "Reasoning Model",
                "message": "Could not verify model availability",
            }
    else:
        checks["reasoning_model"] = {
            "status": "fail",
            "label": "Reasoning Model",
            "message": "Ollama is offline — cannot check model",
        }

    # 5. Embedding model
    if checks["ollama"]["status"] == "ok":
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(EMBED_MODEL in n for n in models)
                checks["embedding_model"] = {
                    "status": "ok" if available else "degraded",
                    "label": "Embedding Model",
                    "model": EMBED_MODEL,
                }
                if not available:
                    checks["embedding_model"]["message"] = (
                        f"Model '{EMBED_MODEL}' not found — RAG features unavailable. "
                        f"Run: ollama pull {EMBED_MODEL}"
                    )
        except Exception:
            checks["embedding_model"] = {
                "status": "degraded",
                "label": "Embedding Model",
                "message": "Could not verify embedding model",
            }
    else:
        checks["embedding_model"] = {
            "status": "degraded",
            "label": "Embedding Model",
            "message": "Ollama is offline — cannot check embedding model",
        }

    # 6. Sarvam OCR (optional)
    checks["sarvam"] = {
        "status": "ok" if SARVAM_ENABLED else "degraded",
        "label": "HATAD Vision (OCR)",
        "enabled": SARVAM_ENABLED,
    }
    if not SARVAM_ENABLED:
        checks["sarvam"]["message"] = "No API key configured — using local text extraction only"

    # Overall status
    statuses = [c["status"] for c in checks.values()]
    if any(s == "fail" for s in statuses):
        overall = "fail"
    elif any(s == "degraded" for s in statuses):
        overall = "degraded"
    else:
        overall = "ok"

    return {"overall": overall, "checks": checks}
