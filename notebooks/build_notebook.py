"""Generate (and optionally execute) notebooks/end_to_end_pipeline.ipynb.

    python notebooks/build_notebook.py --execute
"""
import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Hybrid Quantum-Classical Term Deposit Campaign Pipeline - end-to-end notebook

This notebook walks through the three linked stages. Heavy computations (5x5 nested CV, QAOA optimisation, noisy
simulations, bootstrap) are produced by `python run_all.py` and loaded from `results/`; every stage also has **live
cells** that recompute key quantities on the spot so the numbers can be checked without trusting the files.

Set `RERUN_PIPELINE = True` to regenerate everything from scratch first (about 35 minutes on 2 CPU cores).

**customer_id chain:** Stage 1 writes `stage1_predictions.csv` -> Stage 2 reads its top 10 -> Stage 3 reads Stage 2's six."""),
    code("""import sys, os, json, warnings, subprocess
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
RERUN_PIPELINE = False
if RERUN_PIPELINE:
    subprocess.run([sys.executable, str(ROOT / "run_all.py")], check=True, cwd=ROOT)
from qcampaign import config, data, utils
R = config.RESULTS
print(json.dumps(utils.load_json(R / "environment.json"), indent=1)[:1500])"""),
    md("## 0. Data, integrity checks and the customer_id assumption"),
    code("""d = data.load()
print(d.checks)
print()
print(d.id_mapping_note)
d.train.head()"""),
    md("""## 1. Stage 1 - subscription prediction

Candidates: classical baselines (engineered logistic regression, random forest, Gaussian-kernel LR control), two
quantum models (fidelity quantum kernel with an IQP feature map on 6 qubits + kernel logistic regression; data
re-uploading VQC on 6 qubits) and a hybrid ensemble. Nested CV: outer 5-fold x 5 repeats, inner 5-fold tuning on log
loss; all preprocessing/selection fitted inside training folds. Production model chosen by the **pre-declared** rule
(lowest mean log loss)."""),
    code("""cv = pd.read_csv(R / "stage1_cv_metrics.csv", index_col=0)
cols = ["roc_auc", "pr_auc", "brier", "log_loss", "precision@10pct", "recall@10pct", "calib_slope"]
display(cv[cols + [c + "_sd" for c in ["roc_auc", "pr_auc", "brier", "log_loss"]]].round(4).sort_values("log_loss"))
print("chosen:", utils.load_json(R / "stage1_cv_details.json")["chosen_model"])
display(Image(R / "figures" / "stage1_cv_comparison.png"))"""),
    md("**Live check:** refit the production model on all training data and confirm it reproduces the submitted test probabilities."),
    code("""from qcampaign import stage1
card = utils.load_json(R / "stage1_model_card.json")
model, comps = stage1.fit_production(d, card["chosen_model"])
live = stage1.predict_test(d, model)
sub = pd.read_csv(R / "stage1_predictions.csv")
m = live.merge(sub, on="customer_id", suffixes=("_live", "_file"))
print("max |p_live - p_file| =", float((m.predicted_probability_live - m.predicted_probability_file).abs().max()),
      "(the file stores 6 decimals)")
print("same ranking:", (m.rank_live == m.rank_file).all())
prod = comps[card["chosen_model"]]
print("selected features:", prod.selected_features())
print("standardised coefficients:", {k: round(float(v), 3) for k, v in prod.coefficients().items()})
sub.head(12)"""),
    code("""display(Image(R / "figures" / "stage1_test_ranking.png")); display(Image(R / "figures" / "stage1_calibration.png"))"""),
    md("""## 2. Stage 2 - constrained selection (QAOA vs brute force)

Objective: maximise sum (p_i * deposit_i - cost_i) over exactly 6 of the top 10, subject to budget <= 25 and capacity <= 22.
Three formulations: full QUBO with slack registers (18 qubits here), presolved QUBO (10 qubits; inequalities proven
implied by the cardinality constraint for this candidate set), and constraint-preserving QAOA (Dicke state + XY mixer)."""),
    code("""from qcampaign import stage2
from qcampaign.qaoa import build_qubo, verify_ground_state
top = stage2.load_candidates(d)
prob = stage2.make_problem(top, d.constraints)
feas, n_all = prob.brute_force()
print(f"live brute force: {len(feas)} feasible of {n_all}; optimum", [i for i, s in zip(prob.ids, feas[0]['x']) if s],
      round(feas[0]['objective'], 4))
full = build_qubo(prob, config.PENALTY_SAFETY, include_k_penalty=True, presolve=False)
print("full QUBO qubits:", full.n_vars, "slack registers:", full.slack, "penalty A =", round(full.penalties['A'], 3))
print(f"ground-state check over 2^{full.n_vars} states:", verify_ground_state(full, feas))
top"""),
    code("""det = utils.load_json(R / "stage2_details.json")
print(json.dumps(det["summary"], indent=1))
display(pd.read_csv(R / "stage2_runs.csv").round(4))
display(Image(R / "figures" / "stage2_qaoa.png")); display(Image(R / "figures" / "stage2_bitstrings.png"))"""),
    code("""print("Optimised QAOA parameters (gamma in normalised-energy units):")
for v, o in det["optimisation"].items():
    for p, r in o.items():
        print(f"  {v:22s} p={p}: gammas={np.round(r['gammas'], 4).tolist()} betas={np.round(r['betas'], 4).tolist()} "
              f"P(opt)={r['quality_exact']['p_optimal']:.4f}")
pd.read_csv(R / "stage2_top_bitstrings.csv", dtype={"bitstring": str, "selection": str}).head(10)"""),
    md("""## 3. Stage 3 - P(at least 3 of 6 subscribe)

One qubit per customer, RY(2 arcsin sqrt(p)) so P(1) = p. Qubit order = ascending Stage-1 rank; Qiskit bitstrings are
little-endian (rightmost character = q0)."""),
    code("""from qcampaign import stage3
s3 = utils.load_json(R / "stage3_details.json")
sel = stage3.load_selection(d)
display(sel)
p = sel.predicted_probability.to_numpy()
print("live exact Poisson-binomial:", stage3.exact_at_least(p, 3))
print(stage3.prep_circuit(p).draw(output="text"))
keys = ["exact_poisson_binomial", "statevector_probability", "quantum_estimate_ideal_shots", "quantum_ci95_wilson",
        "quantum_abs_error", "monte_carlo_estimate", "monte_carlo_abs_error"]
print(json.dumps({k: s3[k] for k in keys}, indent=1))
print({b: {k: round(v[k], 5) for k in ["estimate_unmitigated", "estimate_readout_mitigated"]} for b, v in s3["noisy"].items()})
display(Image(R / "figures" / "stage3_results.png"))"""),
    md("## 4. Error propagation across stages (bootstrap of the training data)"),
    code("""u = utils.load_json(R / "uncertainty_propagation.json")
print(json.dumps({k: v for k, v in u.items() if not k.endswith("draws")}, indent=1))
display(Image(R / "figures" / "uncertainty_propagation.png"))"""),
    md("## 5. Correctness tests"),
    code("""r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", str(ROOT / "tests")],
                   capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-800:])"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    out = HERE / "end_to_end_pipeline.ipynb"
    if a.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, out)
    print("wrote", out)
