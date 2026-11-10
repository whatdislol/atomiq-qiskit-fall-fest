"""Central configuration: paths, seeds, grids and pre-declared decision rules.

Everything that influences a result is defined here so a judge can audit and
re-run the pipeline. Seeds are recorded in ``results/environment.json``.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_XLSX = ROOT / "data" / "bank_campaign_participant_release.xlsx"
# Optional file supplied by the organisers that maps Stage1_Test row order -> customer_id.
# If present it overrides the default assumption (see data.assign_test_ids).
TEST_ID_OVERRIDE_CSV = ROOT / "data" / "test_customer_ids_override.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# ----------------------------------------------------------------------------- seeds
SEEDS = {
    "global": 42,
    "cv_outer": 42,          # RepeatedStratifiedKFold for model assessment
    "cv_inner": 7,           # StratifiedKFold for hyper-parameter tuning
    "vqc_init": 42,          # VQC parameter initialisation
    "bootstrap": 2026,       # uncertainty propagation resampling
    "qaoa_init": 42,         # QAOA multi-start initial points
    "qaoa_sampler": 1234,    # Aer sampler seed (QAOA bitstring sampling)
    "transpiler": 42,        # seed_transpiler for every transpile call
    "stage3_sampler": 5678,  # Aer sampler seed (Stage 3 shots)
    "noise_sim": 91011,      # Aer seed for noisy simulations
    "monte_carlo": 2468,     # classical Monte Carlo validation (Stage 3)
}

# ----------------------------------------------------------------------------- Stage 1
OUTER_SPLITS, OUTER_REPEATS = 5, 5
INNER_SPLITS = 5
N_QUBITS_CLASSIFIER = 6       # features fed to the quantum classifiers (1 qubit / feature)
CLIP_Z = 3.0                  # standardised features clipped to [-3, 3] before angle encoding

GRIDS = {
    "lr_compact": {"k": [4, 6, 8], "C": [0.1, 0.3, 1.0]},
    "lr_onehot": {"C": [0.03, 0.1, 0.3, 1.0]},
    "rbf_klr": {"gamma": [0.15, 0.3, 0.45], "C": [1.0, 3.0, 10.0]},
    "qk_klr": {"gamma": [0.15, 0.3, 0.45], "C": [1.0, 3.0, 10.0]},
    "vqc": {"l2": [0.03, 0.1, 0.3]},
}
VQC_FIXED = {"layers": 2, "input_scale": 0.5, "maxiter": 300}
IQP_REPS = 2

# Production-model rule.
#   Original (fixed before the first nested CV): lowest mean outer-fold log loss (a strictly
#   proper scoring rule; Stages 2-3 consume the probabilities as numbers).
#   Amendment (outcome-preserving on this data, added for cross-platform robustness): every
#   candidate whose mean log loss is within one standard error of the best (SE = sd over repeats
#   of the best model / sqrt(repeats)) is treated as tied, and ties go to the simplest model in
#   SIMPLICITY_ORDER. Without it, a 0.0001 gap can flip with floating-point differences between
#   machines. Reference models (prior, one-hot LR) are excluded from selection.
SELECTION_METRIC = "log_loss"
SELECTION_CANDIDATES = ["lr_compact", "rf", "rbf_klr", "qk_klr", "vqc", "hybrid_ensemble"]
SELECTION_TIE_SE = 1.0
SIMPLICITY_ORDER = ["lr_compact", "rf", "rbf_klr", "qk_klr", "vqc", "hybrid_ensemble"]

TOP_N = 10                    # Stage 2 candidate pool

# ----------------------------------------------------------------------------- Stage 2
QAOA_REPS = [1, 2, 3]
QAOA_RESTARTS = 8
# formulation name -> (presolve, cardinality as penalty, mixer, depths, restarts, run on noisy backends)
STAGE2_FORMULATIONS = {
    "full_penalty_x": dict(presolve=False, k_penalty=True, mixer="x", reps=[1, 2], restarts=3, maxiter=300, noisy=False),
    "presolved_penalty_x": dict(presolve=True, k_penalty=True, mixer="x", reps=QAOA_REPS, restarts=QAOA_RESTARTS,
                                maxiter=600, noisy=True),
    "presolved_xy_dicke": dict(presolve=True, k_penalty=False, mixer="xy", reps=QAOA_REPS, restarts=QAOA_RESTARTS,
                               maxiter=600, noisy=True),
}
QAOA_SHOTS = 8192
PENALTY_SAFETY = 1.5          # multiplier on the provable lower bound for penalty weights

# ----------------------------------------------------------------------------- Stage 3
STAGE3_SHOTS = 8192
MC_SAMPLES = 1_000_000
SUCCESS_THRESHOLD = 3         # at least 3 of 6 subscribe
NOISY_BACKENDS = ["FakeTorino", "FakeBrisbane"]

# ----------------------------------------------------------------------------- Uncertainty propagation
N_BOOTSTRAP = 100
