"""Application configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from backend root (before any os.getenv calls)
load_dotenv(BASE_DIR / ".env")
TEMP_DIR = BASE_DIR / "temp"
UPLOAD_DIR = TEMP_DIR / "uploads"
REPORTS_DIR = TEMP_DIR / "reports"
SESSIONS_DIR = TEMP_DIR / "sessions"
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "app" / "reports" / "templates"

# Create directories
for d in [TEMP_DIR, UPLOAD_DIR, REPORTS_DIR, SESSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:32b")

# Vision model — REMOVED (qwen3-vl unreliable; all intelligence via gpt-oss now)
# Sarvam provides OCR text, gpt-oss does all reasoning/extraction.
# Kept for reference only:
# VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:8b")

# Concurrency
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "1"))  # Parallel analysis pipelines
EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "2"))    # Parallel doc extractions (requires OLLAMA_NUM_PARALLEL>=N for real speedup)

# Processing
MAX_CHUNK_PAGES = 20  # Max pages per LLM chunk — doubled with 128K window
MAX_CHUNK_CHARS = 120000  # Hard char budget per extraction chunk (~30K tokens, fits 128K window)
LLM_TIMEOUT = 1200  # 20 min per LLM call — 128K input + CoT can take longer
LLM_MAX_RETRIES = 3
LLM_CONTEXT_WINDOW = 131072  # 128K tokens — matches Ollama desktop setting (qwen3:32b native)
LLM_MAX_INPUT_CHARS = 480000  # Safety cap: ~120K tokens input before truncation
LLM_WARN_INPUT_CHARS = 400000  # ~100K tokens — log a warning above this

# qwen3:32b advanced capabilities
LLM_USE_STRUCTURED_OUTPUTS = True   # Use JSON Schema format instead of "json"
LLM_USE_THINKING = True             # Enable chain-of-thought (think: true) — hybrid mode
LLM_USE_TOOLS = True                # Enable function/tool calling
LLM_TOOL_CALL_MAX_ROUNDS = 5       # Max tool call round-trips — raised for complex verification groups
LLM_MAX_CONCURRENT_CHUNKS = 3      # Max parallel LLM calls for chunked docs

# Classification thresholds (externalized from classifier.py)
CLASSIFY_MAX_CHARS = int(os.getenv("CLASSIFY_MAX_CHARS", "4000"))         # Max chars sent to LLM for classification
CLASSIFY_MAX_PAGES = int(os.getenv("CLASSIFY_MAX_PAGES", "2"))            # Best pages selected for first pass
CLASSIFY_MAX_TOKENS = int(os.getenv("CLASSIFY_MAX_TOKENS", "1024"))       # Max output tokens for classification
CLASSIFY_RETRY_MAX_CHARS = int(os.getenv("CLASSIFY_RETRY_MAX_CHARS", "8000"))  # Expanded context on retry
CLASSIFY_RETRY_MAX_PAGES = int(os.getenv("CLASSIFY_RETRY_MAX_PAGES", "4"))     # Expanded pages on retry

# RAG / Knowledge Base
RAG_ENABLED = True                  # Feature flag — disable to skip RAG entirely
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = 50               # Texts per embedding API call
RAG_CHUNK_SIZE = 1200               # Characters per text chunk for embedding (larger = more context)
RAG_CHUNK_OVERLAP = 200             # Overlap between adjacent chunks
RAG_TOP_K = 4                       # Chunks returned per retrieval query (fewer = less noise)
RAG_MIN_CHUNK_CHARS = 50            # Skip chunks shorter than this (noise filter)
RAG_MAX_DISTANCE = 0.45             # Cosine distance threshold — discard chunks above this
RAG_MMR_LAMBDA = 0.7                # MMR: balance relevance (λ) vs diversity (1-λ)
RAG_KEYWORD_BOOST = 0.15            # Hybrid retrieval: weight for keyword overlap bonus in scoring
RAG_PRE_INDEX = True                # Index raw OCR text before extraction (enables EC header injection)
CHROMA_DIR = TEMP_DIR / "vectordb"  # ChromaDB persistent storage
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Document-type-aware RAG profiles — overrides per doc type
# Each profile only specifies overrides; unspecified keys inherit from globals above.
# Set RAG_PROFILE_ENABLED=false to collapse all profiles to global defaults (instant rollback).
RAG_PROFILE_ENABLED = os.getenv("RAG_PROFILE_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RAG_DOC_PROFILES: dict[str, dict] = {
    "EC":          {"chunk_size": 800,  "top_k": 6, "mmr_lambda": 0.5},   # smaller chunks keep txns atomic; lambda favours diversity across time periods
    "SALE_DEED":   {"chunk_size": 1500, "overlap": 300},                   # long narratives need larger context windows
    "PATTA":       {"chunk_size": 600,  "overlap": 100, "top_k": 3},      # tabular — answer usually in 1-2 rows
    "A_REGISTER":  {"chunk_size": 600,  "overlap": 100, "top_k": 3},      # same structure as Patta
}

def get_rag_profile(doc_type: str = "") -> dict:
    """Return merged RAG parameters for a document type.

    Falls back to global defaults when RAG_PROFILE_ENABLED is False
    or when no profile exists for the doc type.
    """
    defaults = {
        "chunk_size": RAG_CHUNK_SIZE,
        "overlap": RAG_CHUNK_OVERLAP,
        "top_k": RAG_TOP_K,
        "mmr_lambda": RAG_MMR_LAMBDA,
        "max_distance": RAG_MAX_DISTANCE,
        "keyword_boost": RAG_KEYWORD_BOOST,
    }
    if not RAG_PROFILE_ENABLED or not doc_type:
        return defaults
    overrides = RAG_DOC_PROFILES.get(doc_type, {})
    return {**defaults, **overrides}

# Sarvam AI Document Intelligence — cloud OCR for Tamil documents
# Set SARVAM_API_KEY to enable; leave empty to disable (zero-cost default)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_ENABLED = bool(SARVAM_API_KEY)
SARVAM_LANGUAGE = "ta-IN"                    # Tamil (India)
SARVAM_POLL_INTERVAL = int(os.getenv("SARVAM_POLL_INTERVAL", "3"))      # Seconds between status polls
SARVAM_TIMEOUT = int(os.getenv("SARVAM_TIMEOUT", "120"))               # Max seconds per attempt
SARVAM_MAX_RETRIES = int(os.getenv("SARVAM_MAX_RETRIES", "2"))         # Retry failed/timed-out jobs
SARVAM_MAX_FILE_MB = int(os.getenv("SARVAM_MAX_FILE_MB", "40"))        # Skip files larger than this
SARVAM_CONCURRENCY = int(os.getenv("SARVAM_CONCURRENCY", "1"))          # Max concurrent Sarvam jobs (1 = serial, avoids 429)

# Adobe PDF Services — secondary cloud OCR (fallback when Sarvam fails/disabled)
# Set ADOBE_PDF_CLIENT_ID + ADOBE_PDF_CLIENT_SECRET to enable
ADOBE_PDF_CLIENT_ID = os.getenv("ADOBE_PDF_CLIENT_ID", "")
ADOBE_PDF_CLIENT_SECRET = os.getenv("ADOBE_PDF_CLIENT_SECRET", "")
ADOBE_PDF_ENABLED = bool(ADOBE_PDF_CLIENT_ID and ADOBE_PDF_CLIENT_SECRET)
ADOBE_PDF_TIMEOUT = int(os.getenv("ADOBE_PDF_TIMEOUT", "90"))          # Max seconds per job
ADOBE_PDF_MAX_FILE_MB = int(os.getenv("ADOBE_PDF_MAX_FILE_MB", "100"))  # Adobe supports up to 100MB

# Debug trace mode — set HATAD_TRACE=1 to get detailed deterministic engine logs
TRACE_ENABLED = os.getenv("HATAD_TRACE", "").strip().lower() in ("1", "true", "yes")

# Document types
DOCUMENT_TYPES = [
    "EC",                    # Encumbrance Certificate
    "PATTA",                 # Patta / Chitta
    "A_REGISTER",            # A-Register (அ-பதிவேடு) — village land register
    "SALE_DEED",             # Sale Deed
    "CHITTA",                # Chitta
    "ADANGAL",               # Adangal / Village Account
    "FMB",                   # Field Measurement Book
    "LAYOUT_APPROVAL",       # CMDA/DTCP Layout Approval
    "LEGAL_HEIR",            # Legal Heir Certificate
    "POA",                   # Power of Attorney
    "COURT_ORDER",           # Court Order/Decree
    "WILL",                  # Will / Testament
    "PARTITION_DEED",        # Partition Deed
    "GIFT_DEED",             # Gift Deed
    "RELEASE_DEED",          # Release Deed / Mortgage Release
    "OTHER",
]

# Tamil Nadu registration rates (configurable for policy changes)
TN_STAMP_DUTY_RATE = float(os.getenv("TN_STAMP_DUTY_RATE", "0.07"))    # 7% of property value
TN_REGISTRATION_FEE_RATE = float(os.getenv("TN_REGISTRATION_FEE_RATE", "0.04"))  # 4% (capped at ₹4L residential)

# Risk score bands
RISK_BANDS = {
    "LOW": {"min": 80, "max": 100, "color": "#1A7A3A", "label": "Low Risk"},
    "MEDIUM": {"min": 50, "max": 79, "color": "#C27A00", "label": "Medium Risk"},
    "HIGH": {"min": 20, "max": 49, "color": "#BF4A00", "label": "High Risk"},
    "CRITICAL": {"min": 0, "max": 19, "color": "#BF1C2E", "label": "Critical Risk"},
}
