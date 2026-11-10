#!/usr/bin/env python
"""Bundle the submission into a single zip (code, results, report, notebook, data).

    python package_submission.py            -> ../hybrid_quantum_campaign_submission.zip
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "hybrid_quantum_campaign_submission.zip"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TOP = "hybrid_quantum_campaign_submission"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts) or p.suffix == ".pyc":
            continue
        z.write(p, f"{TOP}/{p.relative_to(ROOT)}")
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
