"""Regenerate PDF report for session b7796cf7 to test template changes."""
import sys, json
sys.path.insert(0, ".")
from app.reports.generator import generate_pdf_report

session_path = "temp/sessions/b7796cf7.json"
data = json.loads(open(session_path, encoding="utf-8").read())
pdf = generate_pdf_report(data)
print(f"PDF generated: {pdf} ({pdf.stat().st_size:,} bytes)")
