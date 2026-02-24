"""Quick test: show before/after sanitization for every check in the session."""
import json, sys
sys.path.insert(0, ".")
from app.reports.generator import _sanitize_explanation

data = json.loads(open("temp/sessions/b7796cf7.json", encoding="utf-8").read())
checks = data.get("verification_result", {}).get("checks", [])
changed = 0
for c in checks:
    exp = c.get("explanation", "")
    if not exp:
        continue
    clean = _sanitize_explanation(exp)
    if clean != exp:
        changed += 1
        cid = c.get("check_id", "?")
        st = c.get("status", "?")
        print(f"=== {cid} ({st}) ===")
        print(f"BEFORE ({len(exp)} chars):")
        print(exp[:300])
        if len(exp) > 300:
            print("...")
        print(f"\nAFTER ({len(clean)} chars):")
        print(clean[:300])
        if len(clean) > 300:
            print("...")
        print()

print(f"Total checks: {len(checks)}, sanitised: {changed}")
