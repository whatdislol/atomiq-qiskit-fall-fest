"""Figures for the README and presentation, regenerated from saved result files only.

Palette: validated reference categorical slots (blue / orange / aqua) used as a fixed family
encoding - classical = blue, quantum = orange, hybrid = aqua - plus neutral ink for references.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, utils

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
FAMILY = {"prior": MUTED, "lr_onehot": MUTED, "lr_compact": BLUE, "rf": BLUE, "rbf_klr": BLUE,
          "qk_klr": ORANGE, "vqc": ORANGE, "hybrid_ensemble": AQUA}
SHORT = {"prior": "Base rate", "lr_onehot": "LR one-hot (ref.)", "lr_compact": "LR engineered",
         "rf": "Random forest", "rbf_klr": "RBF kernel LR", "qk_klr": "Quantum kernel LR",
         "vqc": "VQC (re-uploading)", "hybrid_ensemble": "Hybrid ensemble"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _family_legend(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color=c, lw=0, markersize=7, label=l)
         for c, l in [(BLUE, "Classical"), (ORANGE, "Quantum"), (AQUA, "Hybrid"), (MUTED, "Reference")]]
    ax.legend(handles=h, loc=loc, fontsize=8.5, handletextpad=0.3)


def stage1_cv():
    s = pd.read_csv(config.RESULTS / "stage1_cv_metrics.csv", index_col=0)
    chosen = utils.load_json(config.RESULTS / "stage1_cv_details.json")["chosen_model"]
    order = ["prior", "lr_onehot", "rf", "lr_compact", "rbf_klr", "qk_klr", "vqc", "hybrid_ensemble"]
    order = [m for m in order if m in s.index]
    prior_brier = s.loc["prior", "brier"]
    s["brier_skill"] = 1 - s["brier"] / prior_brier
    s["brier_skill_sd"] = s["brier_sd"] / prior_brier
    panels = [("pr_auc", "PR-AUC  (higher is better)"), ("roc_auc", "ROC-AUC"),
              ("brier_skill", "Brier skill vs base rate"), ("precision@10pct", "Precision @ top 10%")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, (m, t) in zip(axes, panels):
        for i, name in enumerate(order):
            ax.errorbar(s.loc[name, m], i, xerr=s.loc[name, m + "_sd"], fmt="o", color=FAMILY[name],
                        ms=7 if name == chosen else 5.5, mec=INK if name == chosen else FAMILY[name],
                        mew=1.4 if name == chosen else 0, elinewidth=1.4, capsize=0)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SHORT[m] + ("  (chosen)" if m == chosen else "") for m in order])
    _family_legend(axes[-1])
    fig.suptitle("Stage 1 - nested cross-validation on the 300 training customers (5x5 outer folds, mean +/- sd over repeats)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "stage1_cv_comparison.png")


def stage1_calibration():
    oof = pd.read_csv(config.RESULTS / "stage1_oof_predictions_train.csv")
    chosen = utils.load_json(config.RESULTS / "stage1_cv_details.json")["chosen_model"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    series = [("lr_compact", BLUE, "s"), ("qk_klr", ORANGE, "^"), ("vqc", VIOLET, "D"), ("hybrid_ensemble", AQUA, "o")]
    for name, col, mk in series:
        p, y = oof[name].to_numpy(), oof["y"].to_numpy()
        q = np.quantile(p, np.linspace(0, 1, 6))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 4)
        xs = [p[idx == b].mean() for b in range(5)]
        ys = [y[idx == b].mean() for b in range(5)]
        ax.plot(xs, ys, marker=mk, color=col, lw=2, ms=6.5, label=SHORT[name] + (" (chosen)" if name == chosen else ""))
    lim = max(0.45, max(ys) + 0.05)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted probability (quintile bins)")
    ax.set_ylabel("Observed subscription rate")
    ax.set_title("Out-of-fold reliability (quintile bins, 35 positives)")
    ax.legend(fontsize=8.5, loc="upper left")
    _save(fig, "stage1_calibration.png")


def stage1_ranking():
    pr = pd.read_csv(config.RESULTS / "stage1_predictions.csv").sort_values("rank")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    c = np.where(pr.top10_flag == 1, AQUA, AXIS)
    ax.bar(pr["rank"], pr.predicted_probability, color=c, width=0.8)
    ax.set_xlabel("Test customers ranked by predicted probability")
    ax.set_ylabel("P(subscribe)")
    ax.set_title("Stage 1 - test-set probabilities; the top 10 (highlighted) become Stage 2 candidates")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0, 101)
    for _, r in pr.head(10).iterrows():
        pass
    ax.annotate(f"top 10: {pr.head(10).predicted_probability.min():.3f} - {pr.head(10).predicted_probability.max():.3f}",
                xy=(10.5, pr.head(10).predicted_probability.min()), xytext=(22, pr.predicted_probability.max() * 0.85),
                fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    _save(fig, "stage1_test_ranking.png")


FORM = {"full_penalty_x": ("Full QUBO, 19 qubits (X mixer)", BLUE, "s"),
        "presolved_penalty_x": ("Presolved QUBO, 10 qubits (X mixer)", ORANGE, "^"),
        "presolved_xy_dicke": ("Dicke + XY mixer, 10 qubits", AQUA, "o")}


def stage2_qaoa():
    det = utils.load_json(config.RESULTS / "stage2_details.json")
    runs = pd.read_csv(config.RESULTS / "stage2_runs.csv")
    base = det["summary"]["random_baselines"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    allp = []
    for v, (lab, col, mk) in FORM.items():
        if v not in det["optimisation"]:
            continue
        o = det["optimisation"][v]
        ps = sorted(int(k) for k in o); allp += ps
        ax.plot(ps, [o[str(p)]["quality_exact"]["p_optimal"] for p in ps], marker=mk, color=col, lw=2, ms=7, label=lab)
    ax.axhline(base["uniform_K_subsets_p_optimal"], color=MUTED, ls="--", lw=1)
    ax.text(max(allp), base["uniform_K_subsets_p_optimal"] * 1.2, "uniform over the 210 six-subsets", color=MUTED, fontsize=8, ha="right")
    ax.axhline(base["uniform_bits_p_optimal"], color=MUTED, ls=":", lw=1)
    ax.text(max(allp), base["uniform_bits_p_optimal"] * 1.2, "uniform over 1024 bitstrings", color=MUTED, fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(sorted(set(allp)))
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(measure the exact optimum)")
    ax.set_title("Ideal QAOA: probability of the optimal six")
    ax.legend(fontsize=8.5, loc="center right")
    ax = axes[1]
    labels, vals, cols = [], [], []
    for v, (lab, col, mk) in FORM.items():
        sub = runs[runs.variant == v]
        if not len(sub):
            continue
        pmax = sub.p.max()
        modes = [("ideal_shots", "ideal shots")] + [(f"noisy_{b}", b.replace("Fake", "") + " noisy") for b in config.NOISY_BACKENDS] + \
                [(f"noisy_{b}_readout_mitigated", b.replace("Fake", "") + " noisy + RM") for b in config.NOISY_BACKENDS]
        for mode, ml in modes:
            r = sub[(sub.p == pmax) & (sub["mode"] == mode)]
            if len(r):
                labels.append(f"{lab.split(',')[0]} p={pmax}: {ml}")
                vals.append(r.p_optimal_sampled.iloc[0]); cols.append(col)
    yy = np.arange(len(vals))[::-1]
    ax.barh(yy, vals, color=cols, height=0.65)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8)
    for i, v in zip(yy, vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(base["uniform_K_subsets_p_optimal"], color=MUTED, ls="--", lw=1)
    ax.set_xlabel(f"fraction of {config.QAOA_SHOTS} shots that decode to the optimum")
    ax.set_title("Finite shots, hardware noise, readout mitigation (RM)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "stage2_qaoa.png")


def stage2_bitstrings():
    tb = pd.read_csv(config.RESULTS / "stage2_top_bitstrings.csv", dtype={"bitstring": str, "selection": str}).head(12)
    det = utils.load_json(config.RESULTS / "stage2_details.json")
    opt_obj = det["summary"]["classical_optimum_objective"]
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [AQUA if (f and abs(o - opt_obj) < 1e-9) else (BLUE if f else AXIS) for f, o in zip(tb.feasible, tb.objective)]
    ax.barh(range(len(tb))[::-1], tb["count"], color=cols, height=0.7)
    ax.set_yticks(range(len(tb))[::-1])
    ax.set_yticklabels([f"{b}  obj {o:6.3f}" for b, o in zip(tb.bitstring, tb.objective)], fontsize=8, family="DejaVu Sans Mono")
    ax.set_xlabel("counts")
    ax.set_title(f"Most frequent measured bitstrings - {det['summary']['selected_by']}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AQUA, label="optimal & feasible"), Patch(color=BLUE, label="feasible"),
                       Patch(color=AXIS, label="infeasible")], fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "stage2_bitstrings.png")


def stage3():
    r = utils.load_json(config.RESULTS / "stage3_details.json")
    exact = r["exact_poisson_binomial"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax = axes[0]
    pmf = np.array(r["pmf_number_of_subscribers"])
    shots = r["shots"]
    samp = np.zeros(len(pmf))
    for b, c in r["ideal_counts"].items():
        samp[b.count("1")] += c / shots
    k = np.arange(len(pmf))
    ax.bar(k, samp, width=0.7, color=[ORANGE if i >= config.SUCCESS_THRESHOLD else "#f2c9b6" for i in k],
           label=f"quantum circuit, {shots} shots\n(dark = success, >= 3)")
    ax.scatter(k, pmf, marker="_", s=420, linewidths=2.2, color=INK, zorder=3, label="exact (Poisson-binomial)")
    ax.axvline(config.SUCCESS_THRESHOLD - 0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("number of the 6 customers who subscribe")
    ax.set_ylabel("probability")
    ax.set_title("Distribution of subscribers")
    ax.set_ylim(0, pmf.max() * 1.32)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    est = [("Statevector (exact circuit)", r["statevector_probability"], 0, INK2),
           (f"Ideal Aer, {shots} shots", r["quantum_estimate_ideal_shots"], 1.96 * r["quantum_standard_error"], ORANGE),
           (f"Classical Monte Carlo, {r['monte_carlo_samples']:,}", r["monte_carlo_estimate"], 1.96 * r["monte_carlo_standard_error"], BLUE)]
    for b, v in r["noisy"].items():
        se = np.sqrt(v["estimate_unmitigated"] * (1 - v["estimate_unmitigated"]) / shots)
        est.append((f"{b.replace('Fake', '')} noisy, unmitigated", v["estimate_unmitigated"], 1.96 * se, RED))
        est.append((f"{b.replace('Fake', '')} noisy, readout-mitigated", v["estimate_readout_mitigated"], 1.96 * se, VIOLET))
    q = r["amplitude_estimation"]["ideal"]
    est.append((f"MLAE (ideal), {q['oracle_calls']:,} oracle calls", q["estimate"], 1.96 * q["sd_fisher"], YELLOW))
    for i, (lab, v, e, c) in enumerate(est[::-1]):
        ax.errorbar(v, i, xerr=e, fmt="o", color=c, ms=6, elinewidth=1.5)
    ax.axvline(exact, color=AQUA, lw=2)
    ax.set_title(f"Estimates vs exact value {exact:.4f} (green line)")
    ax.set_yticks(range(len(est))); ax.set_yticklabels([e[0] for e in est[::-1]], fontsize=8.5)
    ax.set_xlabel("P(at least 3 subscribe)  (bars: 95% interval)")
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    conv = pd.read_csv(config.RESULTS / "stage3_shot_convergence.csv")
    ax.loglog(conv.shots, conv.rmse, marker="o", color=ORANGE, lw=2, label="empirical RMSE (20 seeds)")
    ax.loglog(conv.shots, conv.theory_sd, color=MUTED, ls="--", lw=1.2, label=r"$\sqrt{P(1-P)/N}$")
    ax.set_xlabel("shots N"); ax.set_ylabel("RMSE vs exact")
    ax.set_title("Shot-noise convergence")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "stage3_results.png")


def uncertainty():
    path = config.RESULTS / "uncertainty_propagation.json"
    if not path.exists():
        return
    u = utils.load_json(path)
    r = utils.load_json(config.RESULTS / "stage3_details.json")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    d = np.array(u["fixed_six_draws"])
    ax.hist(d, bins=25, color=BLUE, alpha=0.9, label=f"model uncertainty (bootstrap B={u['n_bootstrap']})")
    se = r["quantum_standard_error"]
    ax.axvspan(r["exact_poisson_binomial"] - 1.96 * se, r["exact_poisson_binomial"] + 1.96 * se, color=ORANGE, alpha=0.25,
               label=f"quantum shot noise, 95% ({r['shots']} shots)")
    ax.axvline(r["exact_poisson_binomial"], color=INK, lw=1.5, label="submitted estimate")
    ax.set_xlabel("P(at least 3 of our six subscribe)")
    ax.set_ylabel("bootstrap replicates")
    ax.set_title("Error propagation: Stage 1 uncertainty dominates Stage 3 shot noise")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", visible=False)
    _save(fig, "uncertainty_propagation.png")


def all_figures():
    _style()
    for f in (stage1_cv, stage1_calibration, stage1_ranking, stage2_qaoa, stage2_bitstrings, stage3, uncertainty):
        try:
            f()
        except FileNotFoundError as e:
            print(f"[figures] skipped {f.__name__}: {e}")
