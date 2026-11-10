"""Stage 2 - constrained customer selection with QAOA, benchmarked against brute force.

Decision rule (fixed in advance, never uses the brute-force answer):
    the reported selection is the feasible sampled bitstring with the highest objective among
    the ideal finite-shot Aer executions of all QAOA formulations at every depth. Feasibility and
    objective are computed classically for each measured bitstring (standard sample-and-decode).
The brute-force optimum is computed only afterwards, for the gap and for verification.
"""
from __future__ import annotations

import time
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config, hardware, utils
from .data import CampaignData
from .qaoa import (QAOASimulator, SelectionProblem, build_qubo, cvar, qaoa_circuit, verify_ground_state)

CVAR_ALPHA = 0.25   # CVaR objective (Barkoutsos et al. 2020) used for the variational loop


def load_candidates(d: CampaignData, preds: pd.DataFrame | None = None) -> pd.DataFrame:
    preds = pd.read_csv(config.RESULTS / "stage1_predictions.csv") if preds is None else preds
    top = preds[preds.top10_flag == 1].sort_values("rank").copy()
    assert len(top) == config.TOP_N
    top = top.merge(d.stage2, on="customer_id", how="left", validate="one_to_one")
    assert top[["contact_cost", "expected_deposit_value", "capacity_units"]].notna().all().all()
    top["expected_net_value"] = top.predicted_probability * top.expected_deposit_value - top.contact_cost
    top["qubit"] = np.arange(len(top))  # candidate in Stage-1 rank order r -> qubit r-1
    return top.reset_index(drop=True)


def make_problem(top: pd.DataFrame, cons: dict) -> SelectionProblem:
    return SelectionProblem(top.customer_id.tolist(), top.expected_net_value.to_numpy(float),
                            top.contact_cost.to_numpy(float), top.capacity_units.to_numpy(float),
                            cons["K"], cons["budget"], cons["capacity"])


# ---------------------------------------------------------------------------------------------
def decision_marginals(qubo, probs):
    """Aggregate a full-register distribution onto decision vectors (slack bits ignored)."""
    N = qubo.problem.n
    idx = np.arange(len(probs))
    dec = idx & ((1 << N) - 1)
    agg = np.bincount(dec, weights=probs, minlength=2 ** N)
    return agg  # index = sum_i x_i 2^i


def solution_table(prob: SelectionProblem):
    N = prob.n
    idx = np.arange(2 ** N)
    X = ((idx[:, None] >> np.arange(N)) & 1)
    feas = (X.sum(1) == prob.K) & (X @ prob.cost <= prob.budget + 1e-9) & (X @ prob.cap <= prob.capacity + 1e-9)
    obj = X @ prob.v
    return X, feas, obj


def quality(prob, agg, opt_obj):
    X, feas, obj = solution_table(prob)
    p_opt = float(agg[feas & np.isclose(obj, opt_obj)].sum())
    p_feas = float(agg[feas].sum())
    exp_obj_feas = float((agg[feas] * obj[feas]).sum() / max(p_feas, 1e-15))
    return {"p_optimal": p_opt, "p_feasible": p_feas, "mean_objective_given_feasible": exp_obj_feas,
            "approx_ratio_given_feasible": exp_obj_feas / opt_obj if opt_obj else np.nan}


def optimise_variant(qubo, mixer, reps_list, restarts, seed, scale, maxiter=600):
    """Classical outer loop: COBYLA on CVaR_alpha(energy), multi-start + INTERP warm starts."""
    E = qubo.energy_vector() / scale
    sim = QAOASimulator(E, qubo.n_vars, "x" if mixer == "x" else "xy", qubo.problem.n, qubo.problem.K)
    rng = np.random.default_rng(seed)
    order = np.argsort(E, kind="stable")
    out, prev = {}, None
    for p in reps_list:
        starts = []
        if prev is not None:  # INTERP (Zhou et al. 2020): linear interpolation of the depth p-1 schedule
            g, b = prev
            xs = np.linspace(0, 1, p)
            xp = np.linspace(0, 1, p - 1) if p > 1 else np.array([0.0])
            starts.append(np.r_[np.interp(xs, xp, g), np.interp(xs, xp, b)])
        ramp = np.r_[np.linspace(0.1, 0.8, p) * 0.5, np.linspace(0.8, 0.1, p) * 0.5]  # annealing-like ramp
        starts.append(ramp)
        while len(starts) < restarts:
            starts.append(np.r_[rng.uniform(0, np.pi / 2, p), rng.uniform(0, np.pi / 2, p)])
        best, trace = None, []
        nfev_total = 0
        for s0 in starts:
            f = lambda th: cvar(E, sim.probs(th[:p], th[p:]), CVAR_ALPHA, order)
            r = minimize(f, s0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
            nfev_total += r.nfev
            trace.append(float(r.fun))
            if best is None or r.fun < best.fun:
                best = r
        g, b = best.x[:p], best.x[p:]
        probs = sim.probs(g, b)
        prev = (g, b)
        out[p] = {"gammas": g, "betas": b, "cvar": float(best.fun), "expected_energy": float(E @ probs),
                  "restart_values": trace, "n_function_evals": int(nfev_total), "probs": probs}
    return out


def counts_quality(qubo, counts, opt_obj):
    prob = qubo.problem
    shots = sum(counts.values())
    rows = []
    for bstr, c in counts.items():
        x = qubo.decode(bstr)
        e = prob.evaluate(x)
        rows.append({"bitstring": bstr.replace(" ", ""), "count": c, "freq": c / shots,
                     "selection": "".join(map(str, x)), **e})
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    feas = df[df.feasible]
    best = feas.sort_values(["objective", "count"], ascending=[False, False]).iloc[0] if len(feas) else None
    return df, {
        "shots": shots,
        "p_optimal_sampled": float(df.loc[df.feasible & np.isclose(df.objective, opt_obj), "freq"].sum()),
        "p_feasible_sampled": float(feas.freq.sum()),
        "distinct_bitstrings": int(len(df)),
        "best_feasible_selection": None if best is None else best.selection,
        "best_feasible_bitstring": None if best is None else best.bitstring,
        "best_feasible_objective": None if best is None else float(best.objective),
        "best_feasible_count": None if best is None else int(best["count"]),
        "most_frequent_feasible_selection": None if not len(feas) else feas.iloc[0].selection,
        "most_frequent_feasible_objective": None if not len(feas) else float(feas.iloc[0].objective),
    }


def mitigated_quality(qubo, counts, mats, opt_obj):
    p = hardware.mitigate_counts(counts, mats)
    return quality(qubo.problem, decision_marginals(qubo, p), opt_obj)


# ---------------------------------------------------------------------------------------------
def run(d: CampaignData, quick: bool = False):
    t0 = time.time()
    top = load_candidates(d)
    prob = make_problem(top, d.constraints)
    feasible, n_total = prob.brute_force()
    specs = config.STAGE2_FORMULATIONS
    shots = config.QAOA_SHOTS
    MIX = {name: spec["mixer"] for name, spec in specs.items()}

    # --- formulations ---------------------------------------------------------------------
    variants = {name: build_qubo(prob, config.PENALTY_SAFETY, include_k_penalty=spec["k_penalty"],
                                 presolve=spec["presolve"]) for name, spec in specs.items()}
    form = {}
    for name, q in variants.items():
        info = {"n_qubits": q.n_vars, "decision_qubits": prob.n, "slack": {k: {"start_qubit": s, "weights": w}
                for k, (s, w) in q.slack.items()}, "penalty": q.penalties, "redundant_constraints": q.redundant}
        info.update({"mixer": MIX[name], "presolve": specs[name]["presolve"],
                     "cardinality_handling": "penalty" if specs[name]["k_penalty"] else "XY mixer + Dicke state (exact)"})
        if specs[name]["k_penalty"]:
            info["ground_state_check"] = verify_ground_state(q, feasible)
        else:  # restricted to the Hamming-weight-K subspace that the XY mixer preserves
            E = q.energy_vector()
            N = prob.n
            w = np.array([bin(i & ((1 << N) - 1)).count("1") for i in range(len(E))])
            sub = np.where(w == prob.K)[0]
            g = sub[np.argmin(E[sub])]
            info["ground_state_check"] = {
                "subspace": f"Hamming weight {prob.K} on decision qubits ({comb(N, prob.K)} selections x slack)",
                "ground_state_bitstring": format(g, f"0{q.n_vars}b"),
                "ground_state_is_constrained_optimum": bool(tuple(q.decode(format(g, f"0{q.n_vars}b"))) == tuple(feasible[0]["x"]))}
        h, J, off = q.ising()
        info["ising"] = {"h": h, "J_nonzero": {f"{i},{j}": J[i, j] for i in range(q.n_vars) for j in range(i + 1, q.n_vars)
                                              if abs(J[i, j]) > 1e-12}, "offset": off}
        info["qubo"] = {"linear": q.lin, "quadratic_nonzero": {f"{i},{j}": q.Q[i, j] for i in range(q.n_vars)
                                                             for j in range(i + 1, q.n_vars) if abs(q.Q[i, j]) > 1e-12},
                        "constant": q.const}
        form[name] = info

    # --- QAOA optimisation (exact statevector, classical outer loop) -----------------------
    opt = {}
    for name, q in variants.items():
        h, J, _ = q.ising()
        scale = max(np.abs(h).max(), np.abs(J).max())  # normalise so gamma is O(1)
        t = time.time()
        reps_list = [1] if quick else specs[name]["reps"]
        restarts = 2 if quick else specs[name]["restarts"]
        res = optimise_variant(q, MIX[name], reps_list, restarts, config.SEEDS["qaoa_init"], scale, specs[name]["maxiter"])
        for p, r in res.items():
            r["quality_exact"] = quality(prob, decision_marginals(q, r["probs"]), feasible[0]["objective"])
            r["energy_scale"] = scale
        opt[name] = res
        print(f"[stage2] {name}: optimised p={reps_list} in {time.time() - t:.0f}s; "
              + ", ".join(f"p={p}: P(opt)={r['quality_exact']['p_optimal']:.3f} P(feas)={r['quality_exact']['p_feasible']:.3f}"
                          for p, r in res.items()))

    # --- execution as Qiskit circuits on Aer: ideal (shots) -------------------------------
    runs, all_counts = [], {}
    ideal = hardware.ideal_simulator(config.SEEDS["qaoa_sampler"])
    from qiskit import transpile
    for name, q in variants.items():
        for p, r in opt[name].items():
            qc = qaoa_circuit(q, r["gammas"] / r["energy_scale"], r["betas"], MIX[name])
            tqc = transpile(qc, ideal, seed_transpiler=config.SEEDS["transpiler"], optimization_level=1)
            counts = hardware.run_counts(ideal, tqc, shots)
            df, cq = counts_quality(q, counts, feasible[0]["objective"])
            all_counts[(name, p, "ideal")] = df
            runs.append({"variant": name, "p": p, "mode": "ideal_shots", **cq})

    # --- decision rule: best feasible sampled bitstring over all ideal runs ----------------
    ideal_runs = [r for r in runs if r["best_feasible_objective"] is not None]
    pick = max(ideal_runs, key=lambda r: (r["best_feasible_objective"], r["p_optimal_sampled"]))
    x_sel = np.array([int(c) for c in pick["best_feasible_selection"]])
    sel_eval = prob.evaluate(x_sel)

    # --- hardware-like execution: transpile + noise (deepest p of each variant) ------------
    hw = {}
    for bname in config.NOISY_BACKENDS:
        backend = hardware.fake_backend(bname)
        for name, q in variants.items():
            depths = sorted(opt[name]) if specs[name]["noisy"] else [max(opt[name])]
            for p_hw in depths:
                r = opt[name][p_hw]
                qc = qaoa_circuit(q, r["gammas"] / r["energy_scale"], r["betas"], MIX[name])
                tqc, rep = hardware.transpile_report(qc, backend)
                key = f"{name}|p={p_hw}|{bname}"
                if not specs[name]["noisy"]:
                    hw[key] = {"transpile": rep, "note": "noisy execution skipped (19 qubits; depth too large to be informative)"}
                    print(f"[stage2] {name} p={p_hw} on {bname}: depth {rep['original_depth']}->{rep['transpiled_depth']}, "
                          f"2q {rep['transpiled_gate_counts']['two_qubit']}, swaps {rep['swap_count_after_routing']} (transpile only)")
                    continue
                sim = hardware.noisy_simulator(backend)
                counts = hardware.run_counts(sim, tqc, shots)
                df, cq = counts_quality(q, counts, feasible[0]["objective"])
                meas_phys = [rep["physical_layout"][i] for i in range(q.n_vars)]
                mats = hardware.calibrate_readout(backend, meas_phys, shots, config.SEEDS["noise_sim"] + 1)
                mq = mitigated_quality(q, counts, mats, feasible[0]["objective"])
                hw[key] = {"transpile": rep, "noisy_unmitigated": cq, "noisy_readout_mitigated": mq}
                runs.append({"variant": name, "p": p_hw, "mode": f"noisy_{bname}", **cq})
                runs.append({"variant": name, "p": p_hw, "mode": f"noisy_{bname}_readout_mitigated",
                             "p_optimal_sampled": mq["p_optimal"], "p_feasible_sampled": mq["p_feasible"]})
                all_counts[(name, p_hw, bname)] = df
                print(f"[stage2] {name} p={p_hw} on {bname}: depth {rep['original_depth']}->{rep['transpiled_depth']}, "
                      f"2q {rep['transpiled_gate_counts']['two_qubit']}, swaps {rep['swap_count_after_routing']}, "
                      f"P(opt) noisy {cq['p_optimal_sampled']:.4f} / mitigated {mq['p_optimal']:.4f}")

    # --- penalty-design sensitivity (presolved penalty formulation, exact simulation) ----------
    sweep = []
    vmax = float(np.max(np.abs(prob.v)))
    p_sw = 1 if quick else 3
    for ratio in ([0.5, 1.5] if quick else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]):
        qs = build_qubo(prob, include_k_penalty=True, presolve=True, penalty=ratio * vmax)
        chk = verify_ground_state(qs, feasible)
        h, J, _ = qs.ising()
        sc = max(np.abs(h).max(), np.abs(J).max())
        rs = optimise_variant(qs, "x", list(range(1, p_sw + 1)), 2 if quick else 4, config.SEEDS["qaoa_init"], sc)[p_sw]
        qq = quality(prob, decision_marginals(qs, rs["probs"]), feasible[0]["objective"])
        sweep.append({"penalty_over_max_abs_v": ratio, "A": ratio * vmax, "p": p_sw,
                      "ground_state_is_optimum": chk["ground_state_is_constrained_optimum"], **qq})
    pd.DataFrame(sweep).to_csv(config.RESULTS / "stage2_penalty_sweep.csv", index=False, float_format="%.6f")
    print("[stage2] penalty sweep:", [(r["penalty_over_max_abs_v"], r["ground_state_is_optimum"], round(r["p_optimal"], 4)) for r in sweep])

    # --- outputs ---------------------------------------------------------------------------
    opt_x = np.array(feasible[0]["x"])
    gap_abs = feasible[0]["objective"] - sel_eval["objective"]
    top_out = top.copy()
    top_out["selected"] = x_sel
    top_out["selected_exact_optimum"] = opt_x
    cols = ["rank", "customer_id", "predicted_probability", "expected_deposit_value", "contact_cost",
            "capacity_units", "customer_segment", "expected_net_value", "qubit", "selected", "selected_exact_optimum"]
    top_out[cols].to_csv(config.RESULTS / "stage2_candidates.csv", index=False, float_format="%.6f")

    n_feas = len(feasible)
    summary = {
        "selected_ids": [i for i, s in zip(prob.ids, x_sel) if s],
        "selected_bitstring_decision_register": "".join(map(str, x_sel[::-1])),
        "selection_vector_rank_order": "".join(map(str, x_sel)),
        "selected_by": f"{pick['variant']} p={pick['p']} (ideal Aer, {shots} shots)",
        "objective_value": sel_eval["objective"],
        "budget_used": sel_eval["cost"], "budget_limit": prob.budget,
        "capacity_used": sel_eval["capacity"], "capacity_limit": prob.capacity,
        "n_selected": sel_eval["n_selected"], "feasible": sel_eval["feasible"],
        "classical_optimum_ids": [i for i, s in zip(prob.ids, opt_x) if s],
        "classical_optimum_objective": feasible[0]["objective"],
        "gap_absolute": gap_abs,
        "gap_relative": gap_abs / abs(feasible[0]["objective"]),
        "second_best_objective": feasible[1]["objective"] if n_feas > 1 else None,
        "n_feasible_selections": n_feas, "n_assignments_enumerated": n_total,
        "random_baselines": {"uniform_bits_p_optimal": 1 / 2 ** prob.n, "uniform_bits_p_feasible": n_feas / 2 ** prob.n,
                             "uniform_K_subsets_p_optimal": 1 / comb(prob.n, prob.K),
                             "uniform_K_subsets_p_feasible": n_feas / comb(prob.n, prob.K)},
    }
    details = {
        "summary": summary, "formulations": form,
        "optimisation": {v: {str(p): {k: r[k] for k in ("gammas", "betas", "cvar", "expected_energy", "restart_values",
                                                           "n_function_evals", "quality_exact", "energy_scale")}
                             for p, r in res.items()} for v, res in opt.items()},
        "optimiser": {"method": "COBYLA", "objective": f"CVaR_{CVAR_ALPHA} of normalised energy",
                      "restarts": {k: v["restarts"] for k, v in specs.items()}, "depths": {k: v["reps"] for k, v in specs.items()},
                      "init": "INTERP from depth p-1, linear ramp, uniform random", "seed": config.SEEDS["qaoa_init"]},
        "runs": runs, "hardware": hw, "top_feasible_solutions": [
            {"ids": [i for i, s in zip(prob.ids, f["x"]) if s], **{k: f[k] for k in ("objective", "cost", "capacity")}}
            for f in feasible[:10]],
        "angle_convention": "circuit uses gamma_circuit = gamma / energy_scale on the un-normalised Ising H; "
                            "RZ(2 gamma h_i), RZZ(2 gamma J_ij), RX(2 beta), XXPlusYY(2 beta)",
        "bit_order": "Qiskit little-endian: rightmost bitstring character = qubit 0 = Stage-1 rank 1 candidate",
    }
    utils.dump_json(details, config.RESULTS / "stage2_details.json")
    pd.DataFrame(runs).to_csv(config.RESULTS / "stage2_runs.csv", index=False, float_format="%.6f")
    # top measured bitstrings of the selected run
    key = (pick["variant"], pick["p"], "ideal")
    tb = all_counts[key].head(30).copy()
    tb["selected_ids"] = tb.selection.apply(lambda s: " ".join(i for i, c in zip(prob.ids, s) if c == "1"))
    tb.to_csv(config.RESULTS / "stage2_top_bitstrings.csv", index=False, float_format="%.6f")
    (config.RESULTS / "counts").mkdir(exist_ok=True)
    for (v, p, mode), df in all_counts.items():
        df.to_csv(config.RESULTS / "counts" / f"stage2_{v}_p{p}_{mode}.csv", index=False, float_format="%.6f")
    print(f"[stage2] selected {summary['selected_ids']} objective {summary['objective_value']:.4f}; "
          f"optimum {summary['classical_optimum_objective']:.4f}; gap {gap_abs:.2e} ({time.time() - t0:.0f}s)")
    return summary
