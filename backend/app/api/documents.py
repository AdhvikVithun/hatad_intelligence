"""Document upload and management endpoints."""

import uuid
import logging
from pathlib import Path, PurePosixPath
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_FILES_PER_REQUEST = 10
PDF_MAGIC_BYTES = b"%PDF"


def _sanitize_filename(raw: str) -> str:
    """Strip path components and keep only the basename."""
    # Handle both Windows and POSIX paths embedded in filenames
    name = PurePosixPath(raw).name
    name = Path(name).name  # also handles backslashes
    return name or "document.pdf"


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more land documents (PDF) for analysis.
    
    Security: 50 MB limit, PDF magic-byte check, UUID filenames, no internal paths exposed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES_PER_REQUEST} files per upload")

    uploaded = []
    for file in files:
        if not file.filename:
            continue

        safe_name = _sanitize_filename(file.filename)

        # Validate file extension
        if not safe_name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: {safe_name}"
            )

        # Read with size limit (streaming read to avoid unbounded RAM)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB at a time
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {safe_name} exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit",
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        if not content:
            raise HTTPException(status_code=400, detail=f"Empty file: {safe_name}")

        # Validate PDF magic bytes
        if not content[:4].startswith(PDF_MAGIC_BYTES):
            raise HTTPException(
                status_code=400,
                detail=f"File does not appear to be a valid PDF: {safe_name}",
            )

        # Save with UUID filename to prevent path traversal
        file_id = uuid.uuid4().hex[:12]
        stem = Path(safe_name).stem
        dest = UPLOAD_DIR / f"{stem}_{file_id}.pdf"

        with open(dest, "wb") as f:
            f.write(content)

        logger.info(f"Uploaded: {safe_name} → {dest.name} ({len(content):,} bytes)")

        uploaded.append({
            "filename": dest.name,
            "original_name": safe_name,
            "size": len(content),
        })

    return {
        "uploaded": uploaded,
        "count": len(uploaded),
        "message": f"Successfully uploaded {len(uploaded)} document(s)",
    }


@router.get("/list")
async def list_uploaded_documents():
    """List all uploaded documents in the temp directory."""
    files = []
    for f in UPLOAD_DIR.glob("*.pdf"):
        files.append({
            "filename": f.name,
            "size": f.stat().st_size,
        })
    return {"files": files, "count": len(files)}


@router.delete("/clear")
async def clear_uploads():
    """Delete all uploaded documents."""
    count = 0
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink()
        count += 1
    return {"deleted": count, "message": f"Cleared {count} file(s)"}
