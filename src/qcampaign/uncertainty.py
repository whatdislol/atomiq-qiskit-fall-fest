"""Error propagation across the linked stages (training-data bootstrap).

Stage 1 probabilities are estimates; Stages 2-3 inherit their error. To quantify how much,
we refit the production model on B bootstrap resamples of the 300 training customers (stratified,
hyper-parameters frozen at the production values), re-predict the 100 test customers and push
every replicate through the *exact* classical versions of Stages 2 and 3:

    bootstrap train -> p_test^(b) -> top-10^(b) -> optimal 6^(b) (brute force) -> P(>=3)^(b)

Reported: top-10 membership frequency per customer, how often our six are selected, the
distribution of P(>=3) for (a) our fixed six re-scored with p^(b) and (b) each replicate's own six.
This is an analysis of the pipeline's sensitivity; it does NOT alter the submitted Stage 1-3 results.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, utils
from .data import FEATURES, CampaignData
from .metrics import rank_order
from .qaoa import SelectionProblem
from .stage3 import exact_at_least


def _frozen_factories(card):
    """Factories that refit each production component with its hyper-parameters frozen.
    Default arguments bind each component's own params (avoids late-binding closures)."""
    from .models.classical import LRCompact
    from .models.quantum import QuantumKernelKLR, QuantumVQC
    from .stage1 import FACTORIES
    comp = card["components"]
    out = {}
    if "lr_compact" in comp:
        out["lr_compact"] = lambda bp=comp["lr_compact"]["best_params"]: _with_grid(
            LRCompact(), {"k": [bp["k"]], "C": [bp["C"]]})
    if "qk_klr" in comp:
        out["qk_klr"] = lambda bp=comp["qk_klr"]["best_params"]: _with_grid(
            QuantumKernelKLR(), {"gamma": [bp["gamma"]], "C": [bp["C"]]})
    if "vqc" in comp:
        out["vqc"] = lambda bp=comp["vqc"]["best_params"]: _with_grid(QuantumVQC(), {"l2": [bp["l2"]]})
    for name in ("rf", "rbf_klr"):
        if name in comp:
            out[name] = lambda n=name, bp=comp[name]["best_params"]: _with_grid(
                FACTORIES[n](), {k: [v] for k, v in bp.items()})
    return out


def _with_grid(model, grid):
    model.grid = grid
    return model


def _one(b, Xtr, ytr, Xte, ids_te, stage2, cons, card, seed):
    rng = np.random.default_rng(seed + b)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
    facs = _frozen_factories(card)
    probs = [facs[n]().fit(Xtr.iloc[idx].reset_index(drop=True), ytr[idx]).predict_proba(Xte)[:, 1] for n in facs]
    p = np.mean(probs, axis=0)
    order = rank_order(ids_te, p)
    top = order[:config.TOP_N]
    s2 = stage2.set_index("customer_id").loc[ids_te[top]]
    v = p[top] * s2.expected_deposit_value.to_numpy() - s2.contact_cost.to_numpy()
    prob = SelectionProblem(list(ids_te[top]), v, s2.contact_cost.to_numpy(float), s2.capacity_units.to_numpy(float),
                            cons["K"], cons["budget"], cons["capacity"])
    feas, _ = prob.brute_force()
    x = np.array(feas[0]["x"])
    sel = ids_te[top][x == 1]
    return {"p": p, "top10": list(ids_te[top]), "selected": list(sel),
            "p_ge3_own": exact_at_least(p[top][x == 1], config.SUCCESS_THRESHOLD)}


def run(d: CampaignData, quick: bool = False, n_jobs: int = 2):
    t0 = time.time()
    card = utils.load_json(config.RESULTS / "stage1_model_card.json")
    s2 = utils.load_json(config.RESULTS / "stage2_details.json")["summary"]
    s3 = utils.load_json(config.RESULTS / "stage3_details.json")
    B = 10 if quick else config.N_BOOTSTRAP
    Xtr = d.train[FEATURES].reset_index(drop=True)
    ytr = d.train["y"].to_numpy()
    Xte = d.test[FEATURES]
    ids_te = d.test["customer_id"].to_numpy()
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(b, Xtr, ytr, Xte, ids_te, d.stage2, d.constraints, card,
                                                config.SEEDS["bootstrap"]) for b in range(B))
    P = np.array([r["p"] for r in res])                      # B x 100
    our6 = s2["selected_ids"]
    idx6 = [list(ids_te).index(c) for c in our6]
    fixed6 = np.array([exact_at_least(P[b, idx6], config.SUCCESS_THRESHOLD) for b in range(B)])
    own = np.array([r["p_ge3_own"] for r in res])
    top_freq = pd.Series([c for r in res for c in r["top10"]]).value_counts() / B
    sel_freq = pd.Series([c for r in res for c in r["selected"]]).value_counts() / B
    same_six = float(np.mean([set(r["selected"]) == set(our6) for r in res]))
    preds = pd.read_csv(config.RESULTS / "stage1_predictions.csv")
    tab = preds[["customer_id", "predicted_probability", "rank", "top10_flag"]].copy()
    tab["p_boot_mean"] = [P[:, list(ids_te).index(c)].mean() for c in tab.customer_id]
    tab["p_boot_q05"] = [np.quantile(P[:, list(ids_te).index(c)], 0.05) for c in tab.customer_id]
    tab["p_boot_q95"] = [np.quantile(P[:, list(ids_te).index(c)], 0.95) for c in tab.customer_id]
    tab["top10_frequency"] = tab.customer_id.map(top_freq).fillna(0.0)
    tab["selected_frequency"] = tab.customer_id.map(sel_freq).fillna(0.0)
    tab.to_csv(config.RESULTS / "uncertainty_customers.csv", index=False, float_format="%.5f")
    out = {
        "n_bootstrap": B, "resampling": "stratified bootstrap of the 300 training customers; hyper-parameters frozen",
        "our_top10_mean_membership": float(tab.loc[tab.top10_flag == 1, "top10_frequency"].mean()),
        "our_six_selected_identically": same_six,
        "our_six_mean_selection_frequency": float(tab.set_index("customer_id").loc[our6, "selected_frequency"].mean()),
        "p_ge3_point_estimate": s3["exact_poisson_binomial"],
        "p_ge3_fixed_six": {"mean": fixed6.mean(), "sd": fixed6.std(ddof=1), "q05": np.quantile(fixed6, .05),
                            "q50": np.median(fixed6), "q95": np.quantile(fixed6, .95)},
        "p_ge3_reoptimised_six": {"mean": own.mean(), "sd": own.std(ddof=1), "q05": np.quantile(own, .05),
                                  "q50": np.median(own), "q95": np.quantile(own, .95)},
        "quantum_shot_noise_sd": s3["quantum_standard_error"],
        "ratio_model_sd_to_shot_sd": fixed6.std(ddof=1) / s3["quantum_standard_error"],
        "fixed_six_draws": fixed6, "reoptimised_six_draws": own, "seconds": time.time() - t0,
    }
    utils.dump_json(out, config.RESULTS / "uncertainty_propagation.json")
    print(f"[uncertainty] B={B}: our six re-selected {same_six:.0%}; P(>=3) fixed-six "
          f"{out['p_ge3_fixed_six']['mean']:.3f} [{out['p_ge3_fixed_six']['q05']:.3f}, {out['p_ge3_fixed_six']['q95']:.3f}]; "
          f"model sd / shot sd = {out['ratio_model_sd_to_shot_sd']:.1f} ({time.time() - t0:.0f}s)")
    return out
