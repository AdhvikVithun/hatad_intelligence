"""Quick standalone test: Adobe PDF Services Extract API.

Usage:
    cd backend
    python _test_adobe_extract.py
    python _test_adobe_extract.py --file "path/to/your.pdf"
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env
from dotenv import load_dotenv
load_dotenv()


async def main():
    from app.pipeline.adobe_ocr import adobe_extract_text, _adobe_extract_sync

    # Choose test file
    if "--file" in sys.argv:
        idx = sys.argv.index("--file") + 1
        test_pdf = Path(sys.argv[idx])
    else:
        # Default: use a small upload PDF
        uploads = Path("temp/uploads")
        if uploads.exists():
            pdfs = sorted(uploads.glob("download*.pdf"))
            if not pdfs:
                pdfs = sorted(uploads.glob("*.pdf"))
            test_pdf = pdfs[0] if pdfs else None
        else:
            test_pdf = None

    if not test_pdf or not test_pdf.exists():
        print(f"ERROR: No test PDF found. Use --file <path>")
        sys.exit(1)

    size_kb = test_pdf.stat().st_size / 1024
    print(f"Test file: {test_pdf.name} ({size_kb:.1f} KB)")
    print(f"Adobe enabled: {os.getenv('ADOBE_PDF_CLIENT_ID', '')[:8]}...")
    print()

    # Progress callback
    async def progress(stage, msg, detail):
        print(f"  [{stage}] {msg}")

    # Test async API
    print("=" * 60)
    print("TEST 1: Async adobe_extract_text()")
    print("=" * 60)
    t0 = time.time()
    result = await adobe_extract_text(test_pdf, on_progress=progress)
    elapsed = time.time() - t0

    if result is None:
        print(f"\nResult: None (failed or disabled) — {elapsed:.1f}s")
    else:
        print(f"\n✓ Success in {elapsed:.1f}s")
        print(f"  Total pages: {result['total_pages']}")
        print(f"  Full text length: {len(result['full_text']):,} chars")
        print(f"  Extraction quality: {result['extraction_quality']}")
        print(f"  Element count: {result['metadata'].get('element_count', '?')}")
        print()
        
        # Show first 500 chars of text
        preview = result['full_text'][:500]
        print(f"  Text preview:\n  {'─' * 40}")
        for line in preview.split('\n'):
            print(f"  {line}")
        print(f"  {'─' * 40}")
        
        # Per-page summary
        print(f"\n  Per-page summary:")
        for page in result['pages']:
            pn = page['page_number']
            chars = page['quality']['char_count']
            q = page['quality']['quality']
            tables = len(page.get('tables', []))
            print(f"    Page {pn}: {chars:,} chars, quality={q}, tables={tables}")


if __name__ == "__main__":
    asyncio.run(main())
