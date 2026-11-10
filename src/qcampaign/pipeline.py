"""Stage runners used by run_all.py and the notebook. Each stage reads the previous
stage's result file, so the customer_id chain (Stage 1 -> 2 -> 3) is explicit and auditable."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from . import config, utils
from .data import CampaignData


# ============================================================================ Stage 1
def run_stage1(d: CampaignData, quick=False, n_jobs=2):
    from . import stage1
    t = time.time()
    cv = stage1.nested_cv(d, repeats=1 if quick else config.OUTER_REPEATS, n_jobs=n_jobs)
    chosen = stage1.select_model(cv["summary"], n_repeats=1 if quick else config.OUTER_REPEATS)
    stage1.save_cv(cv, chosen)
    rep = stage1.selection_report(cv["summary"], 1 if quick else config.OUTER_REPEATS)
    print(f"[stage1] nested CV done in {cv['wall_seconds']:.0f}s; lowest log loss: {rep['lowest_log_loss']}; "
          f"tied within 1 SE ({rep['standard_error']:.4f}): {rep['tied_within_1se']}; chosen (simplest): {chosen}")
    cols = ["roc_auc", "pr_auc", "brier", "log_loss", "precision@10pct", "recall@10pct", "calib_slope"]
    print(cv["summary"][cols].round(4).to_string())

    model, components = stage1.fit_production(d, chosen)
    preds = stage1.predict_test(d, model)  # the single prediction pass over the test set
    preds.to_csv(config.RESULTS / "stage1_predictions.csv", index=False, float_format="%.6f")

    card = {"chosen_model": chosen, "id_mapping": d.id_mapping_note, "components": {}}
    for name, m in components.items():
        c = {"best_params": m.best_params_, "inner_cv_logloss": getattr(m, "inner_scores_", None)}
        if hasattr(m, "selected_features"):
            c["selected_features"] = m.selected_features()
        if hasattr(m, "coefficients"):
            c["coefficients"] = m.coefficients()
        if name == "vqc":
            c["vqc_parameters"] = m.vqc_.params_.tolist()
            c["vqc_optimizer"] = m.vqc_.opt_result_
            c["n_parameters"] = m.vqc_.n_params()
        card["components"][name] = c
    card["top10"] = preds.loc[preds.top10_flag == 1, ["customer_id", "predicted_probability", "rank"]].to_dict("records")
    card["expected_hits_in_top10"] = float(preds.loc[preds.top10_flag == 1, "predicted_probability"].sum())
    card["mean_test_probability"] = float(preds.predicted_probability.mean())
    utils.dump_json(card, config.RESULTS / "stage1_model_card.json")
    (config.RESULTS / "models").mkdir(exist_ok=True)
    joblib.dump({"chosen": chosen, "components": components}, config.RESULTS / "models" / "stage1_production.joblib")
    print(f"[stage1] top-10: {preds.loc[preds.top10_flag == 1, 'customer_id'].tolist()}  ({time.time() - t:.0f}s)")
    return preds


# ============================================================================ Stage 2
def run_stage2(d: CampaignData, quick=False):
    from . import stage2
    return stage2.run(d, quick=quick)


# ============================================================================ Stage 3
def run_stage3(d: CampaignData, quick=False):
    from . import stage3
    return stage3.run(d, quick=quick)


# ============================================================================ Uncertainty propagation
def run_uncertainty(d: CampaignData, quick=False, n_jobs=2):
    from . import uncertainty
    return uncertainty.run(d, quick=quick, n_jobs=n_jobs)


# ============================================================================ top-level result files
def export_results(d: CampaignData, root=None):
    """Write the three judge-facing files at the repository root (one row per customer,
    customer_id as key). Summary values are repeated on each row so each file is self-contained."""
    root = config.ROOT if root is None else root
    # ---- Stage 1
    s1 = pd.read_csv(config.RESULTS / "stage1_predictions.csv")
    s1[["customer_id", "predicted_probability", "rank", "top10_flag"]].to_csv(root / "stage1_results.csv", index=False,
                                                                          float_format="%.6f")
    pd.DataFrame({"stage1_test_row": np.arange(1, len(d.test) + 1), "customer_id": d.test.customer_id}).to_csv(
        (config.ROOT / "data" if root == config.ROOT else root) / "test_customer_ids_used.csv", index=False)
    # ---- Stage 2
    det = utils.load_json(config.RESULTS / "stage2_details.json")
    sm = det["summary"]
    c = pd.read_csv(config.RESULTS / "stage2_candidates.csv")
    out = pd.DataFrame({
        "customer_id": c.customer_id, "stage1_rank": c["rank"], "predicted_probability": c.predicted_probability,
        "expected_deposit_value": c.expected_deposit_value, "contact_cost": c.contact_cost,
        "capacity_units": c.capacity_units, "customer_segment": c.customer_segment,
        "expected_net_value": c.expected_net_value, "qubit_index": c.qubit,
        "selected_quantum": c.selected, "selected_classical_optimum": c.selected_exact_optimum,
    })
    sel_run = sm["selected_by"]
    out["total_contact_cost_selected"] = sm["budget_used"]
    out["budget_limit"] = sm["budget_limit"]
    out["total_capacity_selected"] = sm["capacity_used"]
    out["capacity_limit"] = sm["capacity_limit"]
    out["objective_value_quantum"] = sm["objective_value"]
    out["objective_value_classical_optimum"] = sm["classical_optimum_objective"]
    out["gap_absolute"] = sm["gap_absolute"]
    out["gap_relative"] = sm["gap_relative"]
    out["best_bitstring_q9_to_q0"] = sm["selected_bitstring_decision_register"]
    tb = pd.read_csv(config.RESULTS / "stage2_top_bitstrings.csv", dtype={"bitstring": str})
    out["top3_measured_bitstrings"] = "; ".join(f"{b}:{n}" for b, n in zip(tb.bitstring.head(3), tb["count"].head(3)))
    out["selected_by"] = sel_run
    out.to_csv(root / "stage2_results.csv", index=False, float_format="%.6f")
    # ---- Stage 3
    s3 = utils.load_json(config.RESULTS / "stage3_details.json")
    q = pd.read_csv(config.RESULTS / "stage3_qubits.csv")
    shots = s3["shots"]
    counts = s3["ideal_counts"]
    n = len(q)
    ones = [sum(v for b, v in counts.items() if b.replace(" ", "")[::-1][i] == "1") for i in range(n)]
    o = pd.DataFrame({
        "customer_id": q.customer_id, "qubit": [f"q{i}" for i in q.qubit], "stage1_rank": q["rank"],
        "predicted_probability": q.predicted_probability, "ry_theta_rad": q.theta_ry,
        "shots_measured_1": ones, "sampled_p1": q.marginal_p1_sampled_ideal,
    })
    o["shots_total"] = shots
    o["shots_success_ge3"] = s3["quantum_good_shots"]
    o["quantum_estimate"] = s3["quantum_estimate_ideal_shots"]
    o["quantum_std_error"] = s3["quantum_standard_error"]
    o["exact_probability"] = s3["exact_poisson_binomial"]
    o["quantum_abs_error"] = s3["quantum_abs_error"]
    o["monte_carlo_estimate"] = s3["monte_carlo_estimate"]
    o["monte_carlo_samples"] = s3["monte_carlo_samples"]
    o["monte_carlo_abs_error"] = s3["monte_carlo_abs_error"]
    o["statevector_probability"] = s3["statevector_probability"]
    for b, v in s3["noisy"].items():
        o[f"noisy_{b}_unmitigated"] = v["estimate_unmitigated"]
        o[f"noisy_{b}_readout_mitigated"] = v["estimate_readout_mitigated"]
    o["bit_order"] = "little-endian: bitstring rightmost char = q0"
    o.to_csv(root / "stage3_results.csv", index=False, float_format="%.6f")
    print("[export] wrote stage1_results.csv, stage2_results.csv, stage3_results.csv")
