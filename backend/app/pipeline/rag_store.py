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
import re
from dataclasses import dataclass
from typing import Callable, Awaitable

import numpy as np
import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K,
    RAG_MIN_CHUNK_CHARS, RAG_MAX_DISTANCE, RAG_MMR_LAMBDA,
    RAG_KEYWORD_BOOST, get_rag_profile,
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
    transaction_id: str = ""  # stable EC transaction ID if available

    def to_citation(self) -> str:
        return f"[{self.filename} p.{self.page_number}]"

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }
        if self.transaction_id:
            d["transaction_id"] = self.transaction_id
        return d


def _build_where_filter(
    filter_filename: str | None = None,
    filter_transaction_type: str | None = None,
) -> dict | None:
    """Build a ChromaDB ``where`` filter from optional criteria."""
    conditions: list[dict] = []
    if filter_filename:
        conditions.append({"filename": filter_filename})
    if filter_transaction_type:
        conditions.append({"transaction_type": filter_transaction_type.strip().lower()})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


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
        doc_type: str = "",
    ) -> int:
        """Index all pages of a document into the vector store.

        Args:
            filename: Document filename
            pages: List of {"page_number": int, "text": str, ...}
            embed_fn: async function that takes list[str] → list[list[float]]
            on_progress: Optional progress callback
            doc_type: Document type (e.g. "EC", "SALE_DEED") for profile-based chunking

        Returns:
            Number of chunks indexed
        """
        profile = get_rag_profile(doc_type)
        chunk_size = profile["chunk_size"]
        overlap = profile["overlap"]

        all_chunks: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for page in pages:
            page_num = page["page_number"]
            text = page.get("text", "")
            if not text.strip():
                continue

            chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{filename}__p{page_num}__c{ci}"
                all_chunks.append(chunk)
                meta = {
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": ci,
                }
                # Include transaction_id in metadata if provided
                tid = page.get("transaction_id", "")
                if tid:
                    meta["transaction_id"] = tid
                all_metas.append(meta)
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

    async def index_transactions(
        self,
        filename: str,
        transactions: list[dict],
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        ec_header: dict | None = None,
    ) -> int:
        """Index EC transactions as individual semantic chunks.

        Each transaction becomes one embedding with rich metadata —
        enabling retrieval by transaction_type, survey_number, party name,
        or document_number.  This replaces the flat character-based chunking
        for post-extraction EC data with semantically meaningful units.

        Args:
            filename: Source EC filename
            transactions: List of extracted transaction dicts
            embed_fn: Async function list[str] → list[list[float]]
            ec_header: Optional EC-level context (ec_number, village, period)

        Returns:
            Number of transaction chunks indexed
        """
        if not transactions:
            return 0

        header_prefix = ""
        if ec_header:
            parts = []
            for k in ("ec_number", "village", "taluk", "period_from", "period_to"):
                v = ec_header.get(k)
                if v:
                    parts.append(f"{k.replace('_', ' ').title()}: {v}")
            if parts:
                header_prefix = f"[EC] {filename} | {' | '.join(parts)}\n"

        all_texts: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for i, txn in enumerate(transactions):
            # Build a natural-language text representation of this transaction
            lines = [header_prefix] if header_prefix else [f"[EC] {filename}"]

            txn_type = txn.get("transaction_type", "Unknown")
            doc_num = txn.get("document_number", "")
            date = txn.get("date", "")
            seller = txn.get("seller_or_executant", "")
            buyer = txn.get("buyer_or_claimant", "")
            survey = txn.get("survey_number", "")
            amount = txn.get("consideration_amount", "")
            remarks = txn.get("remarks", "")
            tid = txn.get("transaction_id", f"txn_{i}")

            lines.append(f"Transaction #{txn.get('row_number', i+1)}: {txn_type}")
            if doc_num:
                lines.append(f"Document No: {doc_num}")
            if date:
                lines.append(f"Date: {date}")
            if seller:
                lines.append(f"Seller/Executant: {seller}")
            if buyer:
                lines.append(f"Buyer/Claimant: {buyer}")
            if survey:
                lines.append(f"Survey No: {survey}")
            if amount:
                lines.append(f"Amount: {amount}")
            if remarks:
                lines.append(f"Remarks: {remarks}")

            text = "\n".join(lines)

            # Metadata for filtered retrieval
            meta = {
                "filename": filename,
                "page_number": txn.get("row_number", i + 1),
                "chunk_index": i,
                "transaction_id": tid,
                "transaction_type": txn_type.lower().strip(),
                "has_survey": bool(survey),
            }
            # Preserve page provenance from extraction (stamped by _run_chunked)
            source_pages = txn.get("source_pages")
            if source_pages:
                meta["source_page_start"] = source_pages[0]
                meta["source_page_end"] = source_pages[-1]

            chunk_id = f"{filename}__txn_{tid}"
            all_texts.append(text)
            all_metas.append(meta)
            all_ids.append(chunk_id)

        if not all_texts:
            return 0

        logger.info(f"RAGStore: Embedding {len(all_texts)} EC transactions from {filename}")
        embeddings = await embed_fn(all_texts)

        self._collection.upsert(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_texts,
            metadatas=all_metas,
        )

        self._indexed_count += len(all_texts)
        logger.info(f"RAGStore: Indexed {len(all_texts)} transactions from {filename} "
                     f"(total: {self._indexed_count})")
        return len(all_texts)

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
                    transaction_id=meta.get("transaction_id", ""),
                ))

        return chunks

    # ── Hybrid keyword helpers ──

    # Stopwords excluded from keyword scoring (common English + domain fillers)
    _STOP_WORDS: set[str] = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "had", "her", "was", "one", "our", "out", "has", "what", "who", "how",
        "this", "that", "with", "from", "they", "been", "have", "which",
        "their", "will", "each", "make", "does", "when", "where", "some",
        "then", "than", "them", "into", "also", "after", "other", "should",
        "could", "would", "about", "these", "there", "document", "documents",
        "find", "search", "check", "verify", "whether", "name", "value",
        # Tamil post-stemming stopwords — high-frequency functional words
        # that survive suffix stripping and add no retrieval signal.
        "இந்த", "அந்த", "என்ற", "அல்லது", "மற்றும்", "உள்ள",
        "ஒரு", "இது", "அது", "என்", "ஆன", "ஆக", "போல",
        "மேல", "கீழ",
    })

    # Tamil agglutinative suffixes — longest-first so greedy stripping
    # removes the maximal suffix (e.g. -களின் before -கள்).
    _TAMIL_SUFFIXES: tuple[str, ...] = (
        "களுக்கு", "களிடம்", "களின்", "களால்", "களை", "கள்",
        "த்தினால்", "த்திற்கு", "த்தின்", "த்தை", "த்து",
        "த்தில்", "னுடைய", "விடம்", "வுக்கு", "யிடம்",
        "யில்", "க்கு", "க்கான", "யின்", "யை", "ன்",
        "ில்", "ால்", "ிடம்", "ுடன்", "ுக்கு",
    )

    @staticmethod
    def _normalize_tamil_token(token: str) -> str:
        """Strip agglutinative suffixes from a Tamil token.

        Uses a longest-first greedy approach: scan the suffix list
        (sorted by length descending) and strip the first match.
        Only one pass — avoids over-stemming.
        """
        for suffix in RAGStore._TAMIL_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Extract significant keywords from a query string.

        Returns lowercased tokens (3+ chars, not stopwords).
        Tamil tokens are suffix-normalized before stopword check.
        Numeric / identifier-like tokens (contain digits) are always included
        regardless of stopword list.
        """
        tokens = re.findall(r"[\w/.-]{3,}", query)
        result: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            low = tok.lower()
            # Normalize Tamil agglutinative forms
            normalized = RAGStore._normalize_tamil_token(low)
            if normalized in seen:
                continue
            seen.add(normalized)
            has_digit = any(c.isdigit() for c in normalized)
            if has_digit or normalized not in RAGStore._STOP_WORDS:
                result.append(normalized)
        return result

    @staticmethod
    def _keyword_score(keywords: list[str], doc_text: str) -> float:
        """Fraction of *keywords* that appear in *doc_text* (case-insensitive).

        Numeric / identifier keywords receive 2× weight so that exact survey
        numbers or document numbers boost the score more than generic terms.
        Tamil tokens in doc_text are suffix-normalized before matching.
        Returns 0.0–1.0.
        """
        if not keywords:
            return 0.0
        doc_lower = doc_text.lower()
        # Build a set of normalized tokens from the document for Tamil matching
        doc_tokens = set()
        for tok in re.findall(r"[\w/.-]{3,}", doc_lower):
            doc_tokens.add(tok)
            doc_tokens.add(RAGStore._normalize_tamil_token(tok))
        total_weight = 0.0
        hit_weight = 0.0
        for kw in keywords:
            w = 2.0 if any(c.isdigit() for c in kw) else 1.0
            total_weight += w
            # Match if keyword appears as substring OR as normalized token
            if kw in doc_lower or kw in doc_tokens:
                hit_weight += w
        return hit_weight / total_weight if total_weight else 0.0

    def query_sync(
        self,
        question_embedding: list[float],
        n_results: int = RAG_TOP_K,
        filter_filename: str | None = None,
        filter_transaction_type: str | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Synchronous query using pre-computed embedding (for tool calls).

        Args:
            filter_transaction_type: Optional — restrict to chunks with this
                transaction_type metadata (e.g. "mortgage", "sale").
            query_text: Optional original query string for hybrid keyword
                re-ranking.  When provided, chunks containing exact query
                terms receive a score bonus.
        """
        if self._indexed_count == 0:
            return []

        where_filter = _build_where_filter(filter_filename, filter_transaction_type)

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

        # Extract keywords for hybrid scoring
        keywords = self._extract_keywords(query_text) if query_text else []

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
                # Hybrid: reduce distance (= boost) for keyword matches
                adjusted_dist = dist
                if keywords:
                    kw = self._keyword_score(keywords, doc)
                    adjusted_dist = max(0.0, dist - RAG_KEYWORD_BOOST * kw)
                chunks.append(RetrievedChunk(
                    text=doc,
                    filename=meta["filename"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    score=adjusted_dist,
                    transaction_id=meta.get("transaction_id", ""),
                ))

        # Re-sort after keyword adjustment
        if keywords and chunks:
            chunks.sort(key=lambda c: c.score)

        return chunks

    # ── MMR Retrieval ──

    def query_mmr_sync(
        self,
        question_embedding: list[float],
        n_results: int = RAG_TOP_K,
        filter_filename: str | None = None,
        filter_transaction_type: str | None = None,
        lambda_param: float = RAG_MMR_LAMBDA,
        query_text: str | None = None,
        doc_type: str = "",
    ) -> list[RetrievedChunk]:
        """MMR (Maximal Marginal Relevance) retrieval for diverse results.

        Fetches 3× candidates, then re-ranks using:
          score_i = λ · (sim(q, d_i) + kw_boost)  −  (1−λ) · max_{j∈S} sim(d_i, d_j)

        When *query_text* is provided, the relevance component includes a
        keyword overlap bonus (``RAG_KEYWORD_BOOST``) so that chunks
        containing exact identifiers (survey numbers, document numbers)
        are prioritised even when embedding similarity alone is mediocre.

        Args:
            filter_transaction_type: Optional — restrict to chunks with this
                transaction_type metadata (e.g. "mortgage", "sale").
            query_text: Optional original query string for hybrid keyword
                scoring.

        Returns `n_results` chunks balancing relevance + diversity.
        """
        # Apply doc-type profile overrides (if not explicitly overridden by caller)
        if doc_type:
            profile = get_rag_profile(doc_type)
            if n_results == RAG_TOP_K:
                n_results = profile["top_k"]
            if lambda_param == RAG_MMR_LAMBDA:
                lambda_param = profile["mmr_lambda"]

        if self._indexed_count == 0:
            return []

        # Fetch 3× candidates for MMR re-ranking
        candidate_count = min(n_results * 3, self._indexed_count)
        where_filter = _build_where_filter(filter_filename, filter_transaction_type)

        try:
            results = self._collection.query(
                query_embeddings=[question_embedding],
                n_results=candidate_count,
                where=where_filter,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            logger.error(f"RAGStore MMR query failed: {e}")
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        embeds_raw = results.get("embeddings")
        embeds = (
            embeds_raw[0]
            if embeds_raw is not None and len(embeds_raw) > 0 and len(embeds_raw[0]) > 0
            else None
        )

        # Extract keywords for hybrid scoring
        keywords = self._extract_keywords(query_text) if query_text else []

        # Apply distance threshold first
        candidates = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            if dist > RAG_MAX_DISTANCE:
                continue
            sim = 1.0 - dist  # cosine distance → similarity
            kw_bonus = (
                RAG_KEYWORD_BOOST * self._keyword_score(keywords, doc)
                if keywords else 0.0
            )
            candidates.append({
                "doc": doc, "meta": meta, "dist": dist,
                "sim": sim, "kw_bonus": kw_bonus, "idx": i,
                "embed": embeds[i] if embeds is not None else None,
            })

        if not candidates:
            return []

        # If no embeddings returned (ChromaDB config), fall back to distance ranking
        if embeds is None:
            # Still apply keyword boost when falling back
            candidates.sort(key=lambda c: c["dist"] - c["kw_bonus"])
            return [
                RetrievedChunk(
                    text=c["doc"], filename=c["meta"]["filename"],
                    page_number=c["meta"]["page_number"],
                    chunk_index=c["meta"]["chunk_index"],
                    score=max(0.0, c["dist"] - c["kw_bonus"]),
                    transaction_id=c["meta"].get("transaction_id", ""),
                )
                for c in candidates[:n_results]
            ]

        # MMR greedy selection — numpy-vectorized dot products
        selected: list[dict] = []
        remaining = list(candidates)

        # Pre-convert embeddings to numpy arrays for fast dot products
        for cand in remaining:
            if cand["embed"] is not None:
                cand["_np_embed"] = np.asarray(cand["embed"], dtype=np.float32)
            else:
                cand["_np_embed"] = None

        for _ in range(min(n_results, len(remaining))):
            best_score = -float('inf')
            best_idx = 0

            # Build matrix of selected embeddings for batch dot product
            sel_matrix = None
            if selected:
                sel_embeds = [s["_np_embed"] for s in selected if s["_np_embed"] is not None]
                if sel_embeds:
                    sel_matrix = np.stack(sel_embeds)  # (S, D)

            for ri, cand in enumerate(remaining):
                # Relevance to query (semantic + keyword boost)
                rel = cand["sim"] + cand["kw_bonus"]

                # Max similarity to already selected (diversity penalty)
                max_sim_to_selected = 0.0
                if sel_matrix is not None and cand["_np_embed"] is not None:
                    sims = sel_matrix @ cand["_np_embed"]  # (S,) dot products
                    max_sim_to_selected = float(sims.max())

                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = ri

            selected.append(remaining.pop(best_idx))

        return [
            RetrievedChunk(
                text=s["doc"], filename=s["meta"]["filename"],
                page_number=s["meta"]["page_number"],
                chunk_index=s["meta"]["chunk_index"],
                score=max(0.0, s["dist"] - s["kw_bonus"]),
                transaction_id=s["meta"].get("transaction_id", ""),
            )
            for s in selected
        ]

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
