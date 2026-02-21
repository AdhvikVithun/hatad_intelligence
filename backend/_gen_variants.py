"""Generate 4 PDFs at different risk levels to verify color mapping."""
import sys, os, copy
sys.path.insert(0, os.path.dirname(__file__))

from gen_sample_pdf import session_data
from app.reports.generator import generate_pdf_report

variants = [
    ("low_risk",  92, "LOW"),
    ("med_risk",  65, "MEDIUM"),
    ("high_risk", 35, "HIGH"),
    ("crit_risk", 12, "CRITICAL"),
]

for sid, score, band in variants:
    data = copy.deepcopy(session_data)
    data["session_id"] = sid
    data["risk_score"] = score
    data["risk_band"] = band
    pdf = generate_pdf_report(data)
    print(f"{band:10s} (score={score:3d}) -> {pdf.name}")
