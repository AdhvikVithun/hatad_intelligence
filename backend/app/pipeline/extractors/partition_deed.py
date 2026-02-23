"""Partition Deed extractor — text-based."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import PROMPTS_DIR, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_PARTITION_SCHEMA

logger = logging.getLogger(__name__)


class PartitionDeedExtractor(BaseExtractor):
    """Extract joint owners, partitioned shares, and consent from Partition Deeds."""

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_partition_deed.txt").read_text(encoding="utf-8")

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
        **kwargs,
    ) -> dict:
        name = filename or "Partition Deed"
        pages = extracted_text.get("pages", [])

        if len(pages) <= MAX_CHUNK_PAGES:
            return await self._extract_single(
                extracted_text["full_text"], name, on_progress
            )

        chunks = self._create_chunks(pages)
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)

        async def _limited(chunk_text: str, idx: int):
            async with sem:
                return await self._extract_single(
                    chunk_text, f"{name} chunk {idx+1}/{len(chunks)}", on_progress
                )

        tasks = [_limited(c, i) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge(results)

    async def _extract_single(self, text: str, label: str, on_progress) -> dict:
        prompt = f"Extract all details from this Partition Deed:\n\n{text}"
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_PARTITION_SCHEMA,
            task_label=f"{label} extraction ({len(text):,} chars)",
            on_progress=on_progress,
            think=True,
        )

    @staticmethod
    def _create_chunks(pages: list[dict]) -> list[str]:
        chunks = []
        for i in range(0, len(pages), MAX_CHUNK_PAGES):
            chunk_pages = pages[i : i + MAX_CHUNK_PAGES]
            chunks.append(
                "\n\n".join(
                    f"--- PAGE {p['page_number']} ---\n{p['text']}"
                    for p in chunk_pages
                )
            )
        return chunks

    @staticmethod
    def _merge(results: list) -> dict:
        merged: dict = {}
        all_owners: list = []
        all_shares: list = []
        for r in results:
            if isinstance(r, Exception) or not isinstance(r, dict):
                continue
            all_owners.extend(r.pop("joint_owners", []))
            all_shares.extend(r.pop("partitioned_shares", []))
            for k, v in r.items():
                if v is None or v == "" or v == [] or v == {}:
                    continue
                existing = merged.get(k)
                if existing is None or existing == "" or existing == [] or existing == {}:
                    merged[k] = v
        if all_owners:
            merged["joint_owners"] = all_owners
        if all_shares:
            merged["partitioned_shares"] = all_shares
        return merged
