"""Stage 3 - probability that at least 3 of the 6 selected customers subscribe.

Core method (required by the brief)
-----------------------------------
One qubit per selected customer, ``RY(theta_i)|0>`` with ``theta_i = 2 arcsin(sqrt(p_i))`` so that
P(measure 1) = p_i exactly; qubits are independent (product state), mirroring the independence
assumption. Customer -> qubit order: ascending Stage-1 rank (qubit 0 = best-ranked selected
customer). Qiskit bitstrings are little-endian (qubit 0 is the RIGHTMOST character); success means
Hamming weight >= 3, which is order-invariant, but the per-qubit marginals are checked against p_i
with the correct ordering.

Validation: exact Poisson-binomial (dynamic programming), exact statevector sum, classical Monte
Carlo, finite-shot ideal sampling, noisy fake-backend sampling (unmitigated and readout-mitigated).

Extension (exploratory): Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020) on an oracle
that coherently counts the subscribers into a 3-qubit register and flags weight >= 3.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit.quantum_info import Statevector

from . import config, hardware, utils
from .data import CampaignData


# ------------------------------------------------------------------------------------------ exact
def poisson_binomial_pmf(p) -> np.ndarray:
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.convolve(pmf, [1 - pi, pi])
    return pmf


def exact_at_least(p, k) -> float:
    return float(poisson_binomial_pmf(p)[k:].sum())


def wilson_ci(phat, n, z=1.959964):
    den = 1 + z ** 2 / n
    c = (phat + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


# ------------------------------------------------------------------------------------------ circuits
def angles(p):
    return 2 * np.arcsin(np.sqrt(np.asarray(p, float)))


def prep_circuit(p, measure=True) -> QuantumCircuit:
    qc = QuantumCircuit(len(p), name="customer_state_prep")
    for i, t in enumerate(angles(p)):
        qc.ry(float(t), i)
    if measure:
        qc.measure_all()
    return qc


def success_from_counts(counts, k):
    shots = sum(counts.values())
    good = sum(v for b, v in counts.items() if b.replace(" ", "").count("1") >= k)
    return good / shots, good, shots


def marginals_from_counts(counts, n):
    shots = sum(counts.values())
    m = np.zeros(n)
    for b, v in counts.items():
        b = b.replace(" ", "")[::-1]  # reverse -> index i is qubit i
        for i in range(n):
            m[i] += v * (b[i] == "1")
    return m / shots


def success_from_prob_vector(pvec, n, k):
    idx = np.arange(len(pvec))
    w = np.array([bin(i).count("1") for i in idx])
    return float(pvec[w >= k].sum())


# ------------------------------------------------------------------------------------------ QAE oracle
def counting_oracle(p, k=3) -> QuantumCircuit:
    """A|0> : RY state prep on 6 data qubits, coherent Hamming-weight counter (3 qubits),
    flag qubit = [weight >= k]. Specialised to n <= 7 and k = 3:
    weight >= 3  <=>  c2 OR (c1 AND c0), and the two terms are mutually exclusive for weight <= 6."""
    n = len(p)
    assert n <= 6 and k == 3
    qc = QuantumCircuit(n + 4, name="A_count_ge3")
    data, c0, c1, c2, flag = list(range(n)), n, n + 1, n + 2, n + 3
    for i, t in enumerate(angles(p)):
        qc.ry(float(t), i)
    for i in data:  # controlled increment of the 3-bit counter (c2 c1 c0)
        qc.mcx([i, c0, c1], c2)
        qc.ccx(i, c0, c1)
        qc.cx(i, c0)
    qc.cx(c2, flag)
    qc.ccx(c1, c0, flag)
    return qc


def mlae(p, k, schedule, shots, seed, backend=None):
    """Maximum-likelihood amplitude estimation with Grover powers ``schedule``."""
    A = counting_oracle(p, k)
    flag = A.num_qubits - 1
    oracle = QuantumCircuit(A.num_qubits)
    oracle.z(flag)
    Q = grover_operator(oracle, state_preparation=A)
    sim = hardware.ideal_simulator(seed) if backend is None else hardware.noisy_simulator(backend, seed)
    hits, circs_info = [], []
    for m in schedule:
        qc = QuantumCircuit(A.num_qubits, 1)
        qc.compose(A, inplace=True)
        for _ in range(m):
            qc.compose(Q, inplace=True)
        qc.measure(flag, 0)
        tqc = transpile(qc, sim, seed_transpiler=config.SEEDS["transpiler"], optimization_level=1)
        c = sim.run(tqc, shots=shots).result().get_counts()
        hits.append(c.get("1", 0))
        circs_info.append({"m": m, "depth": tqc.depth(), "ops": dict(tqc.count_ops())})
    hits = np.array(hits); ms = np.array(schedule)
    th = np.linspace(1e-6, np.pi / 2 - 1e-6, 200001)
    ll = np.zeros_like(th)
    for m, h in zip(ms, hits):
        pr = np.clip(np.sin((2 * m + 1) * th) ** 2, 1e-12, 1 - 1e-12)
        ll += h * np.log(pr) + (shots - h) * np.log(1 - pr)
    t_hat = th[np.argmax(ll)]
    a_hat = float(np.sin(t_hat) ** 2)
    fisher = 4 * shots * np.sum((2 * ms + 1) ** 2)
    sd = float(abs(np.sin(2 * t_hat)) / np.sqrt(fisher))
    oracle_calls = int(shots * np.sum(2 * ms + 1))
    return {"schedule": schedule, "shots_per_circuit": shots, "hits": hits.tolist(), "estimate": a_hat,
            "sd_fisher": sd, "ci95": [a_hat - 1.96 * sd, a_hat + 1.96 * sd], "oracle_calls": oracle_calls,
            "circuits": circs_info, "n_qubits": A.num_qubits}


# ------------------------------------------------------------------------------------------ runner
def load_selection(d: CampaignData):
    det = utils.load_json(config.RESULTS / "stage2_details.json")
    sel_ids = det["summary"]["selected_ids"]
    preds = pd.read_csv(config.RESULTS / "stage1_predictions.csv")
    sel = preds[preds.customer_id.isin(sel_ids)].sort_values("rank").reset_index(drop=True)
    assert len(sel) == 6 and set(sel.customer_id) == set(sel_ids)
    sel["qubit"] = np.arange(len(sel))
    sel["theta_ry"] = angles(sel.predicted_probability)
    return sel


def run(d: CampaignData, quick: bool = False):
    t0 = time.time()
    k = config.SUCCESS_THRESHOLD
    sel = load_selection(d)
    p = sel.predicted_probability.to_numpy()
    n = len(p)
    pmf = poisson_binomial_pmf(p)
    exact = float(pmf[k:].sum())

    # 1) ideal statevector (exact quantum)
    sv = Statevector(prep_circuit(p, measure=False))
    sv_prob = success_from_prob_vector(sv.probabilities(), n, k)
    sv_marg = [float(sv.probabilities([i])[1]) for i in range(n)]

    # 2) ideal finite-shot Aer sampling
    shots = config.STAGE3_SHOTS
    ideal = hardware.ideal_simulator(config.SEEDS["stage3_sampler"])
    qc = prep_circuit(p)
    tq = transpile(qc, ideal, seed_transpiler=config.SEEDS["transpiler"], optimization_level=1)
    counts = ideal.run(tq, shots=shots).result().get_counts()
    q_est, q_good, _ = success_from_counts(counts, k)
    q_marg = marginals_from_counts(counts, n)

    # shot-convergence study (independent seeds)
    conv = []
    for s in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        ests = []
        for r in range(5 if quick else 20):
            simr = hardware.ideal_simulator(config.SEEDS["stage3_sampler"] + 1000 * r + s)
            c = simr.run(tq, shots=s).result().get_counts()
            ests.append(success_from_counts(c, k)[0])
        ests = np.array(ests)
        conv.append({"shots": s, "mean_estimate": ests.mean(), "rmse": float(np.sqrt(np.mean((ests - exact) ** 2))),
                     "theory_sd": float(np.sqrt(exact * (1 - exact) / s))})

    # 3) classical Monte Carlo
    rng = np.random.default_rng(config.SEEDS["monte_carlo"])
    N = 20_000 if quick else config.MC_SAMPLES
    draws = rng.random((N, n)) < p
    mc = float((draws.sum(1) >= k).mean())

    # 4) noisy hardware-like simulation (+ readout mitigation)
    noisy = {}
    for bname in config.NOISY_BACKENDS:
        backend = hardware.fake_backend(bname)
        tqc, rep = hardware.transpile_report(qc, backend)
        nsim = hardware.noisy_simulator(backend)
        c = nsim.run(tqc, shots=shots).result().get_counts()
        un, _, _ = success_from_counts(c, k)
        meas_phys = [rep["physical_layout"][i] for i in range(n)]
        mats = hardware.calibrate_readout(backend, meas_phys, shots, config.SEEDS["noise_sim"] + 2)
        pm = hardware.mitigate_counts(c, mats)
        mit = success_from_prob_vector(pm, n, k)
        noisy[bname] = {"transpile": rep, "counts": dict(sorted(c.items(), key=lambda kv: -kv[1])),
                        "estimate_unmitigated": un, "estimate_readout_mitigated": mit,
                        "abs_error_unmitigated": abs(un - exact), "abs_error_mitigated": abs(mit - exact),
                        "marginals_unmitigated": marginals_from_counts(c, n).tolist(),
                        "readout_confusion_matrices": [m.tolist() for m in mats]}

    # 5) amplitude estimation extension
    A = counting_oracle(p, k)
    a_sv = float(Statevector(A).probabilities([A.num_qubits - 1])[1])
    sched = [0, 1, 2, 4] if quick else [0, 1, 2, 4, 8]
    qae_shots = 256
    qae = mlae(p, k, sched, qae_shots, config.SEEDS["stage3_sampler"] + 7)
    qae["oracle_statevector_probability"] = a_sv
    qae["abs_error"] = abs(qae["estimate"] - exact)
    # equal-budget comparison: direct sampling with the same number of state preparations
    budget = qae["oracle_calls"]
    sim_b = hardware.ideal_simulator(config.SEEDS["stage3_sampler"] + 99)
    cb = sim_b.run(tq, shots=budget).result().get_counts()
    direct_same_budget = success_from_counts(cb, k)[0]
    qae["direct_sampling_same_budget"] = {"shots": budget, "estimate": direct_same_budget,
                                          "abs_error": abs(direct_same_budget - exact),
                                          "theory_sd": float(np.sqrt(exact * (1 - exact) / budget))}
    # QAE on noisy hardware model (shows the depth cost)
    qae_noisy = mlae(p, k, [0, 1, 2], qae_shots, config.SEEDS["noise_sim"] + 3,
                     backend=hardware.fake_backend(config.NOISY_BACKENDS[0]))
    qae_noisy["abs_error"] = abs(qae_noisy["estimate"] - exact)
    _, rep_q1 = hardware.transpile_report(_q_power_circuit(p, k, 1), hardware.fake_backend(config.NOISY_BACKENDS[0]))

    # ------------------------------------------------------------------------------- outputs
    se = float(np.sqrt(q_est * (1 - q_est) / shots))
    res = {
        "selected_ids": sel.customer_id.tolist(),
        "qubit_order": {f"q{i}": cid for i, cid in enumerate(sel.customer_id)},
        "bit_order_note": "Qiskit little-endian: in a bitstring the RIGHTMOST character is q0.",
        "probabilities": dict(zip(sel.customer_id, p.tolist())),
        "ry_angles": dict(zip(sel.customer_id, sel.theta_ry.tolist())),
        "success_condition": f"Hamming weight >= {k}",
        "exact_poisson_binomial": exact,
        "pmf_number_of_subscribers": pmf.tolist(),
        "expected_subscribers": float(p.sum()),
        "statevector_probability": sv_prob, "statevector_minus_exact": sv_prob - exact,
        "statevector_marginals": sv_marg,
        "quantum_estimate_ideal_shots": q_est, "quantum_good_shots": q_good, "shots": shots,
        "quantum_standard_error": se, "quantum_ci95_wilson": wilson_ci(q_est, shots),
        "quantum_abs_error": abs(q_est - exact), "quantum_z_score": (q_est - exact) / np.sqrt(exact * (1 - exact) / shots),
        "sampled_marginals": q_marg.tolist(), "marginal_max_abs_dev": float(np.abs(q_marg - p).max()),
        "monte_carlo_estimate": mc, "monte_carlo_samples": N, "monte_carlo_abs_error": abs(mc - exact),
        "monte_carlo_standard_error": float(np.sqrt(mc * (1 - mc) / N)),
        "circuit": {"logical_qubits": n, "depth": qc.remove_final_measurements(inplace=False).depth(),
                    "ops": dict(qc.count_ops()), "qasm_like": [f"ry({t:.6f}) q[{i}]" for i, t in enumerate(sel.theta_ry)]},
        "ideal_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "noisy": noisy, "shot_convergence": conv,
        "amplitude_estimation": {"ideal": qae, "noisy_" + config.NOISY_BACKENDS[0]: qae_noisy,
                                 "grover_power_1_transpiled": rep_q1},
    }
    utils.dump_json(res, config.RESULTS / "stage3_details.json")

    rows = sel[["customer_id", "rank", "predicted_probability", "qubit", "theta_ry"]].copy()
    rows["marginal_p1_statevector"] = sv_marg
    rows["marginal_p1_sampled_ideal"] = q_marg
    rows.to_csv(config.RESULTS / "stage3_qubits.csv", index=False, float_format="%.6f")
    pd.DataFrame(conv).to_csv(config.RESULTS / "stage3_shot_convergence.csv", index=False, float_format="%.6f")
    pd.DataFrame([{"bitstring_q5..q0": b, "count": c, "n_subscribers": b.count("1"), "success": int(b.count("1") >= k)}
                  for b, c in res["ideal_counts"].items()]).to_csv(config.RESULTS / "stage3_ideal_counts.csv", index=False)
    print(f"[stage3] exact {exact:.6f} | statevector {sv_prob:.6f} | ideal {shots} shots {q_est:.6f} (+/-{se:.4f}) | "
          f"MC {mc:.6f} | " + " | ".join(f"{b}: {v['estimate_unmitigated']:.4f}->{v['estimate_readout_mitigated']:.4f}"
                                        for b, v in noisy.items())
          + f" | MLAE {qae['estimate']:.4f} ({time.time() - t0:.0f}s)")
    return res


def _q_power_circuit(p, k, m):
    A = counting_oracle(p, k)
    oracle = QuantumCircuit(A.num_qubits); oracle.z(A.num_qubits - 1)
    Q = grover_operator(oracle, state_preparation=A)
    qc = QuantumCircuit(A.num_qubits, 1)
    qc.compose(A, inplace=True)
    for _ in range(m):
        qc.compose(Q, inplace=True)
    qc.measure(A.num_qubits - 1, 0)
    return qc
