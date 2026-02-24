"""Quick standalone Sarvam API test — upload a PDF, poll, report timing."""
import asyncio
import os
import tempfile
import time
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


async def test():
    from sarvamai import AsyncSarvamAI
    from app.config import SARVAM_API_KEY

    f = r"temp\uploads\download (4)_70b94699b35d.pdf"
    print(f"File: {f}")
    print(f"Size: {os.path.getsize(f) / 1024:.1f} KB")

    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    t0 = time.time()
    job = await client.document_intelligence.create_job(
        language="ta-IN", output_format="html"
    )
    print(f"[{time.time()-t0:.1f}s] Job created: {job.job_id}")

    await job.upload_file(f)
    print(f"[{time.time()-t0:.1f}s] Uploaded")

    await job.start()
    print(f"[{time.time()-t0:.1f}s] Started processing")

    # Poll manually with status + page details
    state = "Unknown"
    for i in range(40):  # 40 × 3s = 120s max
        await asyncio.sleep(3)
        status = await job.get_status()
        state = status.job_state
        elapsed = time.time() - t0

        detail_str = ""
        if status.job_details and len(status.job_details) > 0:
            d = status.job_details[0]
            pp = getattr(d, "pages_processed", "?")
            tp = getattr(d, "total_pages", "?")
            ps = getattr(d, "pages_succeeded", "?")
            pf = getattr(d, "pages_failed", "?")
            detail_str = f"  processed={pp}/{tp} ok={ps} fail={pf}"

        print(f"[{elapsed:.1f}s] Poll {i+1}: state={state}{detail_str}")

        if state in ("Completed", "PartiallyCompleted", "Failed"):
            break

    total = time.time() - t0
    print(f"\nFINAL STATE: {state} after {total:.1f}s")

    if state in ("Completed", "PartiallyCompleted"):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.zip"
            await job.download_output(str(out))
            size = out.stat().st_size
            print(f"Downloaded output: {size / 1024:.1f} KB")

            # Parse and show text preview
            import zipfile
            with zipfile.ZipFile(out) as zf:
                print(f"ZIP contents: {zf.namelist()}")
                html_files = [n for n in zf.namelist() if n.endswith(".html")]
                if html_files:
                    html = zf.read(html_files[0]).decode("utf-8", errors="replace")
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    print(f"\nExtracted text ({len(text)} chars):")
                    print(text[:1000])
                    print("..." if len(text) > 1000 else "")
    elif state == "Failed":
        err = getattr(status, "error_message", "unknown")
        print(f"Error: {err}")

    try:
        await client._client_wrapper.httpx_client.httpx_client.aclose()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(test())
