"""Analysis/verification endpoints with SSE progress streaming."""

import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import (
    UPLOAD_DIR, SESSIONS_DIR, REPORTS_DIR, MAX_CONCURRENT_ANALYSES,
    PROMPTS_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT,
)
from app.pipeline.orchestrator import run_analysis, AnalysisSession
from app.pipeline.llm_client import check_ollama_status, get_embeddings
from app.pipeline.memory_bank import MemoryBank
from app.pipeline.rag_store import RAGStore
from app.reports.generator import generate_pdf_report

router = APIRouter()
logger = logging.getLogger(__name__)

# Semaphore limits how many analysis pipelines run concurrently
_analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)


# ── Session load helper with retry (Windows file-lock resilience) ──

_LOAD_MAX_RETRIES = 4          # total attempts (1 initial + 3 retries)
_LOAD_BACKOFF_BASE = 0.4       # seconds — escalates: 0.4, 0.8, 1.6


async def _load_session_with_retry(session_id: str) -> AnalysisSession:
    """Load a session from disk, retrying on transient OS/file errors.

    Windows Defender and other real-time scanners can briefly lock freshly-
    written JSON files.  A short retry loop (≈ 3 s total) lets the lock
    clear before we give up.

    Raises:
        HTTPException 404  – session file does not exist
        HTTPException 503  – transient file error after all retries
    """
    last_exc: Exception | None = None
    for attempt in range(_LOAD_MAX_RETRIES):
        try:
            return AnalysisSession.load(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        except (PermissionError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            last_exc = exc
            wait = _LOAD_BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "Session %s load attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                session_id, attempt + 1, _LOAD_MAX_RETRIES,
                type(exc).__name__, exc, wait,
            )
            await asyncio.sleep(wait)

    # All retries exhausted
    logger.error(
        "Session %s: all %d load attempts failed — last error: %s",
        session_id, _LOAD_MAX_RETRIES, last_exc,
    )
    raise HTTPException(
        status_code=503,
        detail="Temporary file access error — please retry in a few seconds.",
    )


def _safe_json_response(data: dict, status_code: int = 200) -> JSONResponse:
    """Return a JSONResponse that mirrors session.save() serialization.

    Uses ``json.dumps(…, default=str)`` so datetime / Path / other
    non-native types are safely converted — matching what session.save()
    writes to disk.
    """
    content = json.loads(
        json.dumps(data, default=str, ensure_ascii=False)
    )
    return JSONResponse(content=content, status_code=status_code)


# ── Chat configuration ──
_CHAT_MAX_HISTORY_TURNS = 20  # Max user+assistant turn pairs to send to LLM
_CHAT_MAX_NARRATIVE_CHARS = 30_000  # Truncate narrative report in chat context


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
    session = await _load_session_with_retry(session_id)
    return _safe_json_response(session.to_dict())


@router.get("/{session_id}/results")
async def get_analysis_results(session_id: str):
    """Get full analysis results including verification and extracted data."""
    session = await _load_session_with_retry(session_id)

    if session.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Current status: {session.status}",
        )

    result = session.to_dict()
    if session.status == "failed":
        result["incomplete"] = True
    return _safe_json_response(result)


@router.get("/{session_id}/memory-bank")
async def get_memory_bank(session_id: str):
    """Get the cross-document memory bank for a session."""
    session = await _load_session_with_retry(session_id)

    mb = session.memory_bank
    if not mb:
        raise HTTPException(status_code=400, detail="Memory bank not yet populated")
    return _safe_json_response(mb)


@router.get("/{session_id}/report/pdf")
async def download_pdf_report(session_id: str):
    """Generate and download PDF report for completed analysis."""
    session = await _load_session_with_retry(session_id)

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Current status: {session.status}",
        )

    import asyncio
    import concurrent.futures as _cf
    # Playwright's sync_playwright() spawns a subprocess internally.
    # asyncio.to_thread() shares the event loop, which on Windows Py 3.13
    # cannot create subprocesses from a non-main thread.  Run in a
    # *process* pool instead so Playwright gets its own clean event loop.
    loop = asyncio.get_running_loop()
    with _cf.ProcessPoolExecutor(max_workers=1) as pool:
        pdf_path = await loop.run_in_executor(pool, generate_pdf_report, session.to_dict())
    
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
        session = await _load_session_with_retry(session_id)
        import asyncio
        import concurrent.futures as _cf
        loop = asyncio.get_running_loop()
        with _cf.ProcessPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, generate_pdf_report, session.to_dict())
    
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


# ═══════════════════════════════════════════════════
# CHAT ENDPOINT — Post-analysis conversational Q&A
# ═══════════════════════════════════════════════════

class ChatMessageIn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessageIn] = []


def _build_chat_system_prompt(
    session: AnalysisSession,
    mb_context: str,
    rag_evidence: str,
) -> str:
    """Build the system prompt for chat by filling the template."""
    try:
        template = (PROMPTS_DIR / "chat.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        template = "You are HATAD Intelligence, a land due diligence assistant."

    narrative = session.narrative_report or "No narrative report available."
    if len(narrative) > _CHAT_MAX_NARRATIVE_CHARS:
        narrative = narrative[:_CHAT_MAX_NARRATIVE_CHARS] + "\n\n[... report truncated for context window ...]"

    return template.format(
        doc_count=len(session.documents),
        risk_score=session.risk_score or "N/A",
        risk_band=session.risk_band or "N/A",
        memory_bank_context=mb_context or "No memory bank data available.",
        rag_evidence=rag_evidence or "No specific document excerpts retrieved.",
        narrative_report=narrative,
    )


@router.post("/{session_id}/chat")
async def chat_with_session(session_id: str, request: ChatRequest):
    """Streaming chat endpoint — answer user questions about a completed analysis.

    Returns SSE stream with NDJSON events:
      {"token": "..."} — content token
      {"thinking": "..."} — chain-of-thought chunk
      {"done": true, "content": "full response"} — completion signal
      {"error": "..."} — error
    """
    import httpx

    # ── Load and validate session ──
    session = await _load_session_with_retry(session_id)

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Chat is only available for completed analyses. Current status: {session.status}",
        )

    # ── Reconstruct memory bank ──
    mb_context = ""
    if session.memory_bank:
        try:
            mb = MemoryBank.from_dict(session.memory_bank)
            mb_context = mb.get_verification_context()
        except Exception as e:
            logger.warning(f"Chat: Failed to reconstruct memory bank: {e}")

    # ── Query RAG store for relevant evidence ──
    rag_evidence = ""
    try:
        rag_store = RAGStore(session_id)
        # Fix _indexed_count so query() doesn't short-circuit
        rag_store._indexed_count = rag_store._collection.count()
        if rag_store._indexed_count > 0:
            chunks = await rag_store.query(
                request.message, get_embeddings, n_results=6,
            )
            if chunks:
                rag_evidence = RAGStore.format_evidence(chunks, max_chars=6000)
    except Exception as e:
        logger.warning(f"Chat: RAG query failed (session {session_id}): {e}")

    # ── Build messages for Ollama ──
    system_prompt = _build_chat_system_prompt(session, mb_context, rag_evidence)

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (capped)
    history = request.history[-(_CHAT_MAX_HISTORY_TURNS * 2):]
    for msg in history:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})

    # Add the current user message
    messages.append({"role": "user", "content": request.message})

    # ── Stream response via SSE ──
    async def chat_stream():
        accumulated_content = ""
        accumulated_thinking = ""
        thinking_buf = ""
        THINKING_FLUSH = 400
        THINKING_MIN_INTERVAL = 1.5
        last_thinking_emit = 0.0

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT)) as client:
                body = {
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "repeat_penalty": 1.05,
                        "num_predict": 16384,
                    },
                    "think": True,
                }

                async with client.stream(
                    "POST", f"{OLLAMA_BASE_URL}/api/chat", json=body,
                ) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': f'LLM returned status {resp.status_code}'})}\n\n"
                        return

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        msg = chunk.get("message", {})

                        # Thinking tokens
                        think_delta = msg.get("thinking", "")
                        if think_delta:
                            accumulated_thinking += think_delta
                            thinking_buf += think_delta
                            now = time.time()
                            if (len(thinking_buf) >= THINKING_FLUSH
                                    and (now - last_thinking_emit) >= THINKING_MIN_INTERVAL):
                                yield f"data: {json.dumps({'thinking': thinking_buf})}\n\n"
                                thinking_buf = ""
                                last_thinking_emit = now

                        # Content tokens
                        content_delta = msg.get("content", "")
                        if content_delta:
                            accumulated_content += content_delta
                            yield f"data: {json.dumps({'token': content_delta})}\n\n"

                        if chunk.get("done"):
                            break

            # Flush remaining thinking
            if thinking_buf:
                yield f"data: {json.dumps({'thinking': thinking_buf})}\n\n"

            # Done signal
            yield f"data: {json.dumps({'done': True, 'content': accumulated_content})}\n\n"

            # ── Persist chat history ──
            try:
                sess = AnalysisSession.load(session_id)
                if not hasattr(sess, "chat_history") or not isinstance(sess.chat_history, list):
                    sess.chat_history = []
                sess.chat_history.append({
                    "role": "user",
                    "content": request.message,
                    "timestamp": datetime.now().isoformat(),
                })
                sess.chat_history.append({
                    "role": "assistant",
                    "content": accumulated_content,
                    "thinking": accumulated_thinking[:2000] if accumulated_thinking else "",
                    "timestamp": datetime.now().isoformat(),
                })
                sess.save()
            except Exception as save_err:
                logger.warning(f"Chat: Failed to persist chat history: {save_err}")

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to LLM — is Ollama running?'})}\n\n"
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'LLM response timed out'})}\n\n"
        except Exception as e:
            logger.exception(f"Chat stream error for session {session_id}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        chat_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
