"""Evaluation metrics used by the judges plus calibration diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def rank_order(ids, p) -> np.ndarray:
    """Indices sorted by probability descending; ties -> lower customer_id first (spec rule)."""
    ids = np.asarray(ids, dtype=str)
    p = np.asarray(p, float)
    return np.lexsort((ids, -p))


def precision_recall_at_k(y, p, k, ids=None):
    y = np.asarray(y)
    ids = np.arange(len(y)).astype(str) if ids is None else ids
    top = rank_order(ids, p)[:k]
    hits = y[top].sum()
    return hits / k, (hits / y.sum() if y.sum() else np.nan)


def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal a=0, b=1."""
    p = np.clip(p, EPS, 1 - EPS)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def ece(y, p, bins=5):
    """Expected calibration error with equal-frequency bins (few positives -> few bins)."""
    p = np.asarray(p); y = np.asarray(y)
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def all_metrics(y, p, k_abs=10, k_frac=0.10) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k_rel = max(1, int(round(k_frac * len(y))))
    p10, r10 = precision_recall_at_k(y, p, k_abs)
    pk, rk = precision_recall_at_k(y, p, k_rel)
    a, b = calibration_slope_intercept(y, p)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
        f"precision@{k_abs}": p10, f"recall@{k_abs}": r10,
        "precision@10pct": pk, "recall@10pct": rk,
        "calib_intercept": a, "calib_slope": b, "ece": ece(y, p),
        "mean_pred": float(p.mean()), "base_rate": float(y.mean()),
    }


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    g = per_fold.drop(columns=["repeat"], errors="ignore").groupby("model")
    mean, std = g.mean(), g.std()
    out = mean.copy()
    for c in mean.columns:
        out[c + "_sd"] = std[c]
    return out
