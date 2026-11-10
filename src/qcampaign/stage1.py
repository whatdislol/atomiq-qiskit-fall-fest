"""Stage 1 - subscription probability prediction.

Protocol
--------
1. Nested cross-validation on Stage1_Train only: outer 5-fold x 5 repeats
   (RepeatedStratifiedKFold, seed 42) for assessment; every model tunes its own
   hyper-parameters with an inner stratified 5-fold CV on the outer-training part
   (criterion: pooled log loss). Preprocessing, feature selection and the modal month
   are fitted inside each training split, so validation folds never leak.
2. Metrics are computed per repeat on the pooled 300 out-of-fold predictions and
   averaged over the 5 repeats (mean +/- sd).
3. The production model is chosen with the pre-declared rule in ``config``
   (lowest mean log loss) - no test information is involved.
4. The production model is refitted on all 300 training customers (inner-CV tuning
   again) and the 100 test customers are predicted exactly once.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config
from .data import FEATURES, CampaignData
from .metrics import all_metrics, rank_order, summarise
from .models.classical import LRCompact, LROneHot, PriorModel, RandomForest, RBFKernelLR
from .models.quantum import HybridEnsemble, QuantumKernelKLR, QuantumVQC

FACTORIES = {
    "prior": PriorModel,
    "lr_onehot": LROneHot,
    "lr_compact": LRCompact,
    "rf": RandomForest,
    "rbf_klr": RBFKernelLR,
    "qk_klr": QuantumKernelKLR,
    "vqc": QuantumVQC,
}
MODEL_LABELS = {
    "prior": "Base rate (reference)",
    "lr_onehot": "Logistic regression, all one-hot fields (reference)",
    "lr_compact": "Logistic regression, engineered top-k features [classical baseline]",
    "rf": "Random forest, engineered features [classical]",
    "rbf_klr": "Gaussian-kernel logistic regression, 6 features [classical control]",
    "qk_klr": "Quantum fidelity kernel (IQP, 6 qubits) + kernel LR [quantum]",
    "vqc": "Data re-uploading VQC (6 qubits, 2 layers) + linear head [quantum]",
    "hybrid_ensemble": "Hybrid ensemble: mean(LR, quantum kernel, VQC) [hybrid]",
}


def _fit_fold(Xtr, ytr, Xva, names):
    out, params, times = {}, {}, {}
    fitted = {}
    for n in names:
        t = time.time()
        m = FACTORIES[n]().fit(Xtr, ytr)
        times[n] = time.time() - t
        out[n] = m.predict_proba(Xva)[:, 1]
        params[n] = m.best_params_
        fitted[n] = m
    if all(c in fitted for c in HybridEnsemble.components):
        out["hybrid_ensemble"] = HybridEnsemble(fitted).predict_proba(Xva)[:, 1]
        params["hybrid_ensemble"] = {"members": list(HybridEnsemble.components)}
    return out, params, times


def nested_cv(data: CampaignData, names=tuple(FACTORIES), n_jobs: int = 2, repeats=None):
    X = data.train[FEATURES].reset_index(drop=True)
    y = data.train["y"].to_numpy()
    repeats = repeats or config.OUTER_REPEATS
    rskf = RepeatedStratifiedKFold(n_splits=config.OUTER_SPLITS, n_repeats=repeats,
                                   random_state=config.SEEDS["cv_outer"])
    splits = list(rskf.split(X, y))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_fold)(X.iloc[a], y[a], X.iloc[b], list(names)) for a, b in splits)
    wall = time.time() - t0
    models = list(res[0][0])
    oof = {m: np.zeros((repeats, len(y))) for m in models}
    fold_params, fold_times = [], []
    for i, ((a, b), (pred, params, times)) in enumerate(zip(splits, res)):
        r = i // config.OUTER_SPLITS
        for m in models:
            oof[m][r, b] = pred[m]
        fold_params.append({"fold": i, "repeat": r, **{m: params[m] for m in params}})
        fold_times.append({"fold": i, **times})
    rows = []
    for m in models:
        for r in range(repeats):
            rows.append({"model": m, "repeat": r, **all_metrics(y, oof[m][r])})
    per_repeat = pd.DataFrame(rows)
    summary = summarise(per_repeat)
    summary.insert(0, "description", [MODEL_LABELS.get(m, m) for m in summary.index])
    return {"summary": summary, "per_repeat": per_repeat, "oof": oof, "fold_params": fold_params,
            "fold_times": pd.DataFrame(fold_times), "wall_seconds": wall, "y": y,
            "customer_id": data.train["customer_id"].to_numpy()}


def select_model(summary: pd.DataFrame, n_repeats: int | None = None) -> str:
    """Lowest mean log loss; candidates within SELECTION_TIE_SE standard errors of the best are
    tied, and ties go to the simplest model (config.SIMPLICITY_ORDER)."""
    met = config.SELECTION_METRIC
    cand = summary.loc[[m for m in config.SELECTION_CANDIDATES if m in summary.index]]
    best = cand[met].idxmin()
    n = n_repeats or config.OUTER_REPEATS
    se = float(cand.loc[best, met + "_sd"]) / np.sqrt(n) if n > 1 else 0.0
    tied = [m for m in cand.index if cand.loc[m, met] <= cand.loc[best, met] + config.SELECTION_TIE_SE * se]
    return str(min(tied, key=config.SIMPLICITY_ORDER.index))


def selection_report(summary: pd.DataFrame, n_repeats: int) -> dict:
    met = config.SELECTION_METRIC
    cand = summary.loc[[m for m in config.SELECTION_CANDIDATES if m in summary.index]]
    best = cand[met].idxmin()
    se = float(cand.loc[best, met + "_sd"]) / np.sqrt(n_repeats) if n_repeats > 1 else 0.0
    tied = [m for m in cand.index if cand.loc[m, met] <= cand.loc[best, met] + config.SELECTION_TIE_SE * se]
    return {"lowest_log_loss": str(best), "standard_error": se, "tied_within_1se": tied,
            "chosen_simplest_of_tied": select_model(summary, n_repeats)}


def fit_production(data: CampaignData, name: str):
    X = data.train[FEATURES]
    y = data.train["y"].to_numpy()
    if name == "hybrid_ensemble":
        fitted = {c: FACTORIES[c]().fit(X, y) for c in HybridEnsemble.components}
        return HybridEnsemble(fitted), fitted
    m = FACTORIES[name]().fit(X, y)
    return m, {name: m}


def predict_test(data: CampaignData, model) -> pd.DataFrame:
    """Single prediction pass over the 100 test customers + ranking with the tie rule."""
    p = model.predict_proba(data.test[FEATURES])[:, 1]
    ids = data.test["customer_id"].to_numpy()
    order = rank_order(ids, p)
    rank = np.empty(len(p), int)
    rank[order] = np.arange(1, len(p) + 1)
    out = pd.DataFrame({"customer_id": ids, "predicted_probability": p, "rank": rank})
    out["top10_flag"] = (out["rank"] <= config.TOP_N).astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def save_cv(cv: dict, chosen: str):
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    cv["summary"].to_csv(config.RESULTS / "stage1_cv_metrics.csv", float_format="%.5f")
    cv["per_repeat"].to_csv(config.RESULTS / "stage1_cv_per_repeat.csv", index=False, float_format="%.5f")
    oof = pd.DataFrame({"customer_id": cv["customer_id"], "y": cv["y"]})
    for m, a in cv["oof"].items():
        oof[m] = a.mean(0)  # mean OOF probability over repeats
    oof.to_csv(config.RESULTS / "stage1_oof_predictions_train.csv", index=False, float_format="%.6f")
    with open(config.RESULTS / "stage1_cv_details.json", "w") as f:
        json.dump({"chosen_model": chosen, "selection_rule": f"min mean {config.SELECTION_METRIC} over "
                   f"{config.SELECTION_CANDIDATES}; ties within {config.SELECTION_TIE_SE} SE -> simplest "
                   f"({config.SIMPLICITY_ORDER})",
                   "selection": selection_report(cv["summary"], len(cv["per_repeat"].repeat.unique())), "outer": f"{config.OUTER_SPLITS}-fold x "
                   f"{len(cv['per_repeat'].repeat.unique())} repeats", "inner": f"{config.INNER_SPLITS}-fold",
                   "wall_seconds": cv["wall_seconds"], "fold_params": cv["fold_params"],
                   "mean_fit_seconds": cv["fold_times"].drop(columns="fold").mean().to_dict()},
                  f, indent=1, default=str)
