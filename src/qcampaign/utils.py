"""Reproducibility helpers: environment capture and JSON I/O."""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

import numpy as np

from . import config

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "openpyxl", "matplotlib",
            "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "qiskit-machine-learning"]


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_environment(quick: bool = False):
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "seeds": config.SEEDS,
        "quick_mode": quick,
        "cv": {"outer": f"{config.OUTER_SPLITS}x{config.OUTER_REPEATS}", "inner": config.INNER_SPLITS},
        "shots": {"qaoa": config.QAOA_SHOTS, "stage3": config.STAGE3_SHOTS},
        "monte_carlo_samples": config.MC_SAMPLES,
        "noisy_backends": config.NOISY_BACKENDS,
    }
    dump_json(env, config.RESULTS / "environment.json")
    return env


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_Enc)


def load_json(path):
    with open(path) as f:
        return json.load(f)
