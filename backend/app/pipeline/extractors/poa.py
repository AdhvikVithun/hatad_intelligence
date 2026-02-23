"""Power of Attorney (POA) extractor — text-based."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import PROMPTS_DIR
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_POA_SCHEMA

logger = logging.getLogger(__name__)


class POAExtractor(BaseExtractor):
    """Extract principal, agent, powers, and property from Power of Attorney."""

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_poa.txt").read_text(encoding="utf-8")

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
        **kwargs,
    ) -> dict:
        name = filename or "Power of Attorney"
        full_text = extracted_text.get("full_text", "")

        prompt = f"Extract all details from this Power of Attorney document:\n\n{full_text}"
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_POA_SCHEMA,
            task_label=f"{name} extraction ({len(full_text):,} chars)",
            on_progress=on_progress,
            think=True,
        )
