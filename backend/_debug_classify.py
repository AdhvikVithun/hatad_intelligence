"""Debug: check which keyword patterns match the misclassified EC."""
import sys, re
sys.path.insert(0, '.')
from app.pipeline.ingestion import extract_text_from_pdf
from app.pipeline.classifier import _KEYWORD_PATTERNS, _get_keyword_hints
from pathlib import Path

uploads = Path('temp/uploads')
matches = list(uploads.glob('APP_8400001_TXN_486713205_TMPLT_8400004_*.pdf'))
if not matches:
    print('No matching files found')
    sys.exit(1)

fp = matches[0]
print(f'File: {fp.name} ({fp.stat().st_size/1024:.1f} KB)')
data = extract_text_from_pdf(fp, skip_ocr=True)
text = data['full_text']
print(f'Text length: {len(text):,} chars')
print()

# Check each keyword pattern
print("=== KEYWORD PATTERN MATCHES ===")
for doc_type, pattern in _KEYWORD_PATTERNS:
    m = pattern.search(text)
    if m:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        context = text[start:end].replace('\n', ' ')
        print(f"  {doc_type:20s} matched: '{m.group()}' in context: ...{context}...")

# Check the get_keyword_hints result
single, all_matches = _get_keyword_hints(text, fp.name)
print(f"\n_get_keyword_hints -> single={single}, all={all_matches}")

# Show first 2000 chars
print("\n=== TEXT PREVIEW (first 2000 chars) ===")
print(text[:2000])
