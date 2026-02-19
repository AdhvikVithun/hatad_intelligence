"""RAG (Retrieval-Augmented Generation) vector store for document knowledge base.

Embeds all document pages into ChromaDB during ingestion, then provides
semantic retrieval during verification passes. The LLM calls the
search_documents tool to retrieve relevant source passages with page citations.

Architecture:
  - One ChromaDB collection per analysis session
  - Pages split into overlapping chunks (800 chars, 150 overlap) for granular retrieval
  - Embeddings via Ollama nomic-embed-text (768-dim, local)
  - Metadata: filename, page_number, chunk_index for citation
  - Brute-force extraction is NOT replaced — RAG only augments verification
"""

import logging
import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K,
    RAG_MIN_CHUNK_CHARS, RAG_MAX_DISTANCE,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""
    text: str
    filename: str
    page_number: int
    chunk_index: int
    score: float  # distance — lower is more similar

    def to_citation(self) -> str:
        return f"[{self.filename} p.{self.page_number}]"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }


class RAGStore:
    """Per-session vector store wrapping ChromaDB.

    Usage:
        store = RAGStore("session_abc123")
        await store.index_document("doc.pdf", pages, embed_fn)
        results = await store.query("who is the seller?", embed_fn)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.collection_name = f"session_{session_id}"
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR / session_id),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_count = 0
        self._doc_count = 0

    # ── Chunking ──

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE,
                    overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
        """Split text into overlapping chunks for embedding."""
        if not text or not text.strip():
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # Quality filter: skip chunks that are too short (noise, headers, etc.)
            stripped = chunk.strip()
            if stripped and len(stripped) >= RAG_MIN_CHUNK_CHARS:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ── Indexing ──

    async def index_document(
        self,
        filename: str,
        pages: list[dict],
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        on_progress: Callable | None = None,
    ) -> int:
        """Index all pages of a document into the vector store.

        Args:
            filename: Document filename
            pages: List of {"page_number": int, "text": str, ...}
            embed_fn: async function that takes list[str] → list[list[float]]
            on_progress: Optional progress callback

        Returns:
            Number of chunks indexed
        """
        all_chunks: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for page in pages:
            page_num = page["page_number"]
            text = page.get("text", "")
            if not text.strip():
                continue

            chunks = self._chunk_text(text)
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{filename}__p{page_num}__c{ci}"
                all_chunks.append(chunk)
                all_metas.append({
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": ci,
                })
                all_ids.append(chunk_id)

        if not all_chunks:
            logger.warning(f"RAGStore: No text chunks found for {filename}")
            return 0

        # Embed all chunks
        logger.info(f"RAGStore: Embedding {len(all_chunks)} chunks from {filename}")
        embeddings = await embed_fn(all_chunks)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metas,
        )

        self._indexed_count += len(all_chunks)
        self._doc_count += 1
        logger.info(f"RAGStore: Indexed {len(all_chunks)} chunks from {filename} "
                     f"(total: {self._indexed_count})")
        return len(all_chunks)

    # ── Retrieval ──

    async def query(
        self,
        question: str,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        n_results: int = RAG_TOP_K,
        filter_filename: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a question.

        Args:
            question: Natural language query
            embed_fn: Embedding function
            n_results: Number of chunks to return
            filter_filename: Optional — restrict search to one document

        Returns:
            List of RetrievedChunk, sorted by relevance (best first)
        """
        if self._indexed_count == 0:
            return []

        # Embed the query
        q_embeddings = await embed_fn([question])
        if not q_embeddings or not q_embeddings[0]:
            return []

        # Build filter
        where_filter = None
        if filter_filename:
            where_filter = {"filename": filter_filename}

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=q_embeddings,
                n_results=min(n_results, self._indexed_count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"RAGStore query failed: {e}")
            return []

        # Parse results with distance threshold filtering
        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Skip chunks that are too dissimilar (above distance threshold)
                if dist > RAG_MAX_DISTANCE:
                    continue
                chunks.append(RetrievedChunk(
                    text=doc,
                    filename=meta["filename"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    score=dist,
                ))

        return chunks

    def query_sync(
        self,
        question_embedding: list[float],
        n_results: int = RAG_TOP_K,
        filter_filename: str | None = None,
    ) -> list[RetrievedChunk]:
        """Synchronous query using pre-computed embedding (for tool calls)."""
        if self._indexed_count == 0:
            return []

        where_filter = None
        if filter_filename:
            where_filter = {"filename": filter_filename}

        try:
            results = self._collection.query(
                query_embeddings=[question_embedding],
                n_results=min(n_results, self._indexed_count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"RAGStore sync query failed: {e}")
            return []

        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Apply distance threshold to sync queries too
                if dist > RAG_MAX_DISTANCE:
                    continue
                chunks.append(RetrievedChunk(
                    text=doc,
                    filename=meta["filename"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    score=dist,
                ))

        return chunks

    # ── Formatting ──

    @staticmethod
    def format_evidence(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
        """Format retrieved chunks as evidence text for LLM consumption."""
        if not chunks:
            return ""

        lines = ["=== SOURCE EVIDENCE (from original documents) ==="]
        char_count = 0
        for chunk in chunks:
            citation = chunk.to_citation()
            entry = f"\n{citation}:\n{chunk.text}\n"
            if char_count + len(entry) > max_chars:
                lines.append(f"\n[... {len(chunks) - len(lines) + 1} more results truncated ...]")
                break
            lines.append(entry)
            char_count += len(entry)

        return "\n".join(lines)

    # ── Stats ──

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return {
            "session_id": self.session_id,
            "total_chunks": self._indexed_count,
            "documents_indexed": self._doc_count,
            "collection_count": self._collection.count(),
        }

    # ── Cleanup ──

    def cleanup(self):
        """Delete the session's vector store data."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"RAGStore: Cleaned up collection {self.collection_name}")
        except Exception as e:
            logger.warning(f"RAGStore cleanup failed: {e}")
