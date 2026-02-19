"""Application configuration."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Vision model — for document extraction using page images (all types except EC)
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:8b")
VISION_TIMEOUT = 300              # Per-request timeout for vision calls
VISION_CONTEXT_WINDOW = 32768     # 32K tokens — sufficient for 8B vision extraction
VISION_DOC_TYPES = {                # Doc types eligible for vision fallback (when text extraction confidence is low)
    "EC", "PATTA", "CHITTA", "ADANGAL", "FMB", "SALE_DEED",
    "LAYOUT_APPROVAL", "LEGAL_HEIR", "POA", "COURT_ORDER",
    "WILL", "PARTITION_DEED", "GIFT_DEED", "RELEASE_DEED", "OTHER",
}

# Text-primary extraction with confidence-gated vision fallback
VISION_FALLBACK_ENABLED = True        # Feature flag — disable to skip vision fallback entirely
VISION_MAX_CONCURRENT = 1             # Max concurrent vision calls (1 = sequential, avoids GPU queue timeouts)
TEXT_EXTRACTION_CONFIDENCE_THRESHOLD = 0.7   # Below this → trigger vision fallback

# Processing
MAX_CHUNK_PAGES = 10  # Max pages per LLM chunk for large documents
LLM_TIMEOUT = 900  # 15 min per LLM call — model may think 70K+ chars before outputting JSON
LLM_MAX_RETRIES = 3
LLM_CONTEXT_WINDOW = 65536  # 64K tokens — match Ollama desktop setting
LLM_MAX_INPUT_CHARS = 200000  # Safety cap: ~50K tokens input before truncation
LLM_WARN_INPUT_CHARS = 160000  # ~40K tokens — log a warning above this

# gpt-oss:20b advanced capabilities
LLM_USE_STRUCTURED_OUTPUTS = True   # Use JSON Schema format instead of "json"
LLM_USE_THINKING = True             # Enable chain-of-thought (think: true)
LLM_USE_TOOLS = True                # Enable function/tool calling
LLM_TOOL_CALL_MAX_ROUNDS = 2       # Max tool call round-trips per LLM call
LLM_MAX_CONCURRENT_CHUNKS = 3      # Max parallel LLM calls for chunked docs

# RAG / Knowledge Base
RAG_ENABLED = True                  # Feature flag — disable to skip RAG entirely
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = 50               # Texts per embedding API call
RAG_CHUNK_SIZE = 1200               # Characters per text chunk for embedding (larger = more context)
RAG_CHUNK_OVERLAP = 200             # Overlap between adjacent chunks
RAG_TOP_K = 4                       # Chunks returned per retrieval query (fewer = less noise)
RAG_MIN_CHUNK_CHARS = 50            # Skip chunks shorter than this (noise filter)
RAG_MAX_DISTANCE = 0.45             # Cosine distance threshold — discard chunks above this
CHROMA_DIR = TEMP_DIR / "vectordb"  # ChromaDB persistent storage
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Sarvam AI Document Intelligence — cloud OCR for Tamil documents
# Set SARVAM_API_KEY to enable; leave empty to disable (zero-cost default)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_ENABLED = bool(SARVAM_API_KEY)
SARVAM_LANGUAGE = "ta-IN"                    # Tamil (India)
SARVAM_POLL_INTERVAL = 3                     # Seconds between job status polls
SARVAM_TIMEOUT = 120                         # Max seconds to wait for job completion

# Debug trace mode — set HATAD_TRACE=1 to get detailed deterministic engine logs
TRACE_ENABLED = os.getenv("HATAD_TRACE", "").strip().lower() in ("1", "true", "yes")

# Document types
DOCUMENT_TYPES = [
    "EC",                    # Encumbrance Certificate
    "PATTA",                 # Patta / Chitta
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

# Risk score bands
RISK_BANDS = {
    "LOW": {"min": 80, "max": 100, "color": "#00ff88", "label": "Low Risk"},
    "MEDIUM": {"min": 50, "max": 79, "color": "#ffaa00", "label": "Medium Risk"},
    "HIGH": {"min": 20, "max": 49, "color": "#ff6600", "label": "High Risk"},
    "CRITICAL": {"min": 0, "max": 19, "color": "#ff0033", "label": "Critical Risk"},
}
