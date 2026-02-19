#!/usr/bin/env python3
"""CLI tool to run deterministic checks & memory bank on a saved session.

Usage:
    python run_checks.py <session_id_or_file>            # Run checks
    python run_checks.py <session_id_or_file> --trace    # Run with HATAD_TRACE
    python run_checks.py --list                          # List available sessions
    python run_checks.py <session_id> --json             # Output raw JSON

Examples:
    python run_checks.py 0973f73e
    python run_checks.py temp/sessions/0973f73e.json --trace
    python run_checks.py --list
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import SESSIONS_DIR


def list_sessions():
    """List all available session files with summary info."""
    if not SESSIONS_DIR.exists():
        print("No sessions directory found.")
        return

    files = sorted(SESSIONS_DIR.glob("*.json"))
    if not files:
        print("No session files found.")
        return

    print(f"\n{'Session ID':<14} {'Docs':>4}  {'Types':<35} {'Status':<12} {'Risk'}")
    print("─" * 80)

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ed = data.get("extracted_data", {})
            types = [v.get("document_type", "?") for v in ed.values()]
            status = data.get("status", "?")
            risk = data.get("risk_score")
            risk_str = f"{risk}/100" if risk is not None else "—"
            print(f"{f.stem:<14} {len(types):>4}  {', '.join(types):<35} {status:<12} {risk_str}")
        except Exception as e:
            print(f"{f.stem:<14}  ERROR: {e}")

    print()


def load_session(session_ref: str) -> dict:
    """Load a session by ID (prefix) or full path."""
    # Try as direct path first
    path = Path(session_ref)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # Try as session ID (prefix match)
    matches = list(SESSIONS_DIR.glob(f"{session_ref}*.json"))
    if len(matches) == 1:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    elif len(matches) > 1:
        print(f"Ambiguous ID '{session_ref}' — matches: {[m.stem for m in matches]}")
        sys.exit(1)
    else:
        print(f"Session '{session_ref}' not found.")
        sys.exit(1)


def run_checks(session_data: dict, trace: bool = False, output_json: bool = False):
    """Run deterministic checks and memory bank analysis on session data."""
    if trace:
        os.environ["HATAD_TRACE"] = "1"
        import importlib
        import app.config
        importlib.reload(app.config)

    import logging
    if trace:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    from app.pipeline.deterministic import run_deterministic_checks
    from app.pipeline.memory_bank import MemoryBank

    ed = session_data.get("extracted_data", {})
    if not ed:
        print("No extracted_data in session.")
        return

    # ── Run deterministic checks ──
    det_checks = run_deterministic_checks(ed)

    # ── Run memory bank ──
    bank = MemoryBank()
    for filename, doc in ed.items():
        doc_type = doc.get("document_type", "OTHER")
        data = doc.get("data", {})
        bank.ingest_document(filename, doc_type, data)
    conflicts = bank.detect_conflicts()

    if output_json:
        output = {
            "deterministic_checks": det_checks,
            "memory_bank_conflicts": [c.to_dict() for c in conflicts],
            "memory_bank_summary": bank.get_summary(),
        }
        print(json.dumps(output, indent=2, default=str))
        return

    # ── Pretty print results ──
    doc_types = [v.get("document_type", "?") for v in ed.values()]
    print(f"\n{'═' * 70}")
    print(f"  HATAD Check Runner — {len(ed)} document(s): {', '.join(doc_types)}")
    print(f"{'═' * 70}\n")

    # Deterministic checks
    if det_checks:
        print(f"  DETERMINISTIC CHECKS ({len(det_checks)} results)")
        print(f"  {'─' * 60}")
        for c in det_checks:
            status_icon = {"FAIL": "✗", "WARNING": "⚠", "INFO": "ℹ", "PASS": "✓"}.get(c["status"], "?")
            sev_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(c["severity"], "⚪")
            print(f"  {status_icon} {sev_color} [{c['rule_code']}] {c['rule_name']}")
            print(f"    {c['explanation'][:120]}")
            if c.get("evidence"):
                print(f"    Evidence: {c['evidence'][:100]}")
            print()
    else:
        print("  No deterministic check issues found.\n")

    # Memory bank
    print(f"  MEMORY BANK ({bank.get_summary()['total_facts']} facts)")
    print(f"  {'─' * 60}")
    if conflicts:
        print(f"  {len(conflicts)} conflict(s) detected:\n")
        for c in conflicts:
            sev_icon = {"HIGH": "🟠", "WARNING": "🟡", "CRITICAL": "🔴"}.get(c.severity, "⚪")
            print(f"  {sev_icon} [{c.category}/{c.key}] {c.description}")
            print()
    else:
        print("  No conflicts detected.\n")

    # Cross-references
    xrefs = bank.get_cross_references()
    inconsistent = [x for x in xrefs if not x["consistent"]]
    if inconsistent:
        print(f"  INCONSISTENT CROSS-REFERENCES ({len(inconsistent)})")
        print(f"  {'─' * 60}")
        for x in inconsistent[:10]:
            print(f"  ✗ {x['category']}/{x['key']} across {', '.join(x['sources'])}")
            for v in x["values"]:
                print(f"    {v['source']}: {str(v['value'])[:80]}")
            print()

    # Summary
    fail_count = sum(1 for c in det_checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in det_checks if c["status"] == "WARNING")
    print(f"{'═' * 70}")
    print(f"  Summary: {fail_count} FAIL, {warn_count} WARNING, {len(conflicts)} conflicts")
    print(f"{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="HATAD CLI — Run deterministic checks on saved sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("session", nargs="?", help="Session ID (prefix) or JSON file path")
    parser.add_argument("--list", action="store_true", help="List available sessions")
    parser.add_argument("--trace", action="store_true", help="Enable HATAD_TRACE debug output")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of pretty print")

    args = parser.parse_args()

    if args.list:
        list_sessions()
        return

    if not args.session:
        parser.print_help()
        return

    session_data = load_session(args.session)
    run_checks(session_data, trace=args.trace, output_json=args.json)


if __name__ == "__main__":
    main()
