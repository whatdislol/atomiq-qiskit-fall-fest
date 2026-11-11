#!/usr/bin/env python
"""End-to-end runner for the Hybrid Quantum-Classical Term Deposit Campaign Pipeline.

    python run_all.py                 # all stages (about 30-40 min on 2 CPU cores)
    python run_all.py --stages 1      # only Stage 1
    python run_all.py --stages 2 3    # Stages 2-3 (reuse results/stage1_predictions.csv)
    python run_all.py --quick         # smoke test (~10 min): 1 CV repeat, fewer restarts / bootstrap draws;
                                      # writes to ./results_quick and never touches ./results

Outputs are written to ./results (see README.md for the file list).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from src.qcampaign import config, utils, data as data_mod, figures
from src.qcampaign.pipeline import run_stage1, run_stage2, run_stage3, export_results, run_uncertainty

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()


    if args.quick:  # never overwrite the submitted results with smoke-test output
        config.RESULTS = config.ROOT / "results_quick"
        config.FIGURES = config.RESULTS / "figures"
        print("[quick] smoke test: 1 CV repeat, fewer restarts/bootstraps; writing to", config.RESULTS)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    utils.write_environment(quick=args.quick)
    d = data_mod.load()
    print("[data]", d.checks)
    print("[data]", d.id_mapping_note)
    t0 = time.time()

    if 1 in args.stages:
        run_stage1(d, quick=args.quick, n_jobs=args.jobs)
    if 2 in args.stages:
        run_stage2(d, quick=args.quick)
    if 3 in args.stages:
        run_stage3(d, quick=args.quick)
    if 4 in args.stages:
        run_uncertainty(d, quick=args.quick, n_jobs=args.jobs)
    try:
        export_results(d, root=config.RESULTS if args.quick else config.ROOT)
        figures.all_figures()
    except FileNotFoundError as e:
        print("[export] skipped (run all stages first):", e)
    print(f"[done] total {time.time() - t0:.0f}s -> {config.RESULTS}")


if __name__ == "__main__":
    main()
