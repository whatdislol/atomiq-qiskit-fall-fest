#!/usr/bin/env python3
"""Atomiq Stage 3: exact independent Bernoulli probability + 6-qubit shot simulation.

Default input is the fixed N17 Stage-2 winning six customers in q0..q5 order.
Only the standard library is required for exact/Monte Carlo modes.
Qiskit + qiskit-aer are required for --backend aer.

Examples:
 python atomiq_stage3_N17.py --backend numpy --shots 8192 --seed 42
 python atomiq_stage3_N17.py --backend aer --shots 8192 --seed 42
"""
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

CUSTOMERS = [
    ("C0081", 0.849352),
    ("C0079", 0.729566),
    ("C0031", 0.701745),
    ("C0327", 0.264112),
    ("C0374", 0.226869),
    ("C0222", 0.182735),
]


def exact_distribution(probs):
    dist = [1.0]
    for p in probs:
        new = [0.0] * (len(dist) + 1)
        for k, q in enumerate(dist):
            new[k] += q * (1.0 - p)
            new[k + 1] += q * p
        dist = new
    return dist


def outcome_probabilities(probs):
    n = len(probs)
    mapping = {}
    for m in range(1 << n):
        pr = 1.0
        for i, p in enumerate(probs):
            pr *= p if (m >> i) & 1 else 1 - p
        mapping[format(m, f"0{n}b")] = pr  # q[n-1]...q0
    return mapping


def sample_ideal(probs, shots, seed):
    # Mathematically identical computational-basis distribution to the Ry circuit.
    rng = random.Random(seed)
    counts = Counter()
    for _ in range(shots):
        bitmask = sum((rng.random() < p) << i for i, p in enumerate(probs))
        counts[format(bitmask, f"0{len(probs)}b")] += 1
    return counts


def sample_aer(probs, shots, seed):
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SystemExit("Qiskit Aer unavailable. Install qiskit and qiskit-aer, or use --backend numpy") from exc
    qc = QuantumCircuit(len(probs), len(probs))
    for i, p in enumerate(probs):
        qc.ry(2.0 * math.asin(math.sqrt(p)), i)
    qc.measure(range(len(probs)), range(len(probs)))
    backend = AerSimulator(method="statevector")
    compiled = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = Counter({str(k).replace(" ", ""): int(v) for k, v in result.get_counts().items()})
    return counts, compiled


def wilson_interval(n_yes, n_total, z=1.95996398454):
    p = n_yes / n_total
    den = 1 + z*z/n_total
    mid = (p + z*z/(2*n_total))/den
    delta = z * math.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total))/den
    return mid - delta, mid + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["numpy", "aer"], default="numpy")
    parser.add_argument("--outdir", default="stage3_N17_results")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ids, probs = map(list, zip(*CUSTOMERS))
    dist = exact_distribution(probs)
    exact = sum(dist[3:])
    p_by_string = outcome_probabilities(probs)
    assert math.isclose(sum(dist), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(p_by_string.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(v for b, v in p_by_string.items() if b.count("1") >= 3), exact, abs_tol=1e-12)

    circuit_info = None
    if args.backend == "aer":
        counts, circuit = sample_aer(probs, args.shots, args.seed)
        circuit_info = {"depth": circuit.depth(), "count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()}, "num_qubits": circuit.num_qubits}
        with (outdir / "stage3_circuit.txt").open("w", encoding="utf-8") as f:
            f.write(str(circuit.draw(output="text")))
    else:
        counts = sample_ideal(probs, args.shots, args.seed)
    yes = sum(count for bits, count in counts.items() if bits.count("1") >= 3)
    estimate = yes / args.shots
    ci_lo, ci_hi = wilson_interval(yes, args.shots)

    with (outdir / "stage3_selected_customers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "qubit", "stage1_probability", "ry_angle_radians"])
        for i, (cid, p) in enumerate(CUSTOMERS):
            w.writerow([cid, f"q{i}", f"{p:.6f}", f"{2*math.asin(math.sqrt(p)):.12f}"])

    with (outdir / "stage3_outcome_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subscriptions", "exact_probability", "measured_count", "measured_probability", "success_at_least_3"])
        for k, pr in enumerate(dist):
            count = sum(v for bits, v in counts.items() if bits.count("1") == k)
            w.writerow([k, f"{pr:.12f}", count, f"{count/args.shots:.12f}", int(k >= 3)])

    with (outdir / "stage3_bitstring_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bitstring_q5_to_q0", "subscriber_count", "ideal_probability", "measured_count", "success_at_least_3"])
        for bits, theoretical in sorted(p_by_string.items()):
            w.writerow([bits, bits.count("1"), f"{theoretical:.12f}", counts.get(bits, 0), int(bits.count("1") >= 3)])

    result = {
        "model": "N17 + Friend Hybrid, fixed gamma=0.2 (provided Stage-1 predictions)",
        "selected_customers_in_qubit_order_q0_to_q5": ids,
        "stage1_probabilities_q0_to_q5": probs,
        "bitstring_convention": "bitstring q5...q0; rightmost bit corresponds to C0081",
        "circuit": "Ry(theta_i), theta_i = 2*arcsin(sqrt(p_i)), followed by measurement; no entangling gates",
        "assumption": "Independent customer responses conditional on fixed Stage-1 probabilities",
        "event": "number of measured 1 bits >= 3",
        "exact_probability": exact,
        "exact_distribution_k_0_to_6": dist,
        "backend": args.backend,
        "shots": args.shots,
        "seed": args.seed,
        "success_shots": yes,
        "estimated_probability": estimate,
        "absolute_error": abs(estimate - exact),
        "standard_error_approx_at_exact_p": math.sqrt(exact*(1-exact)/args.shots),
        "wilson_95_percent_interval": [ci_lo, ci_hi],
        "circuit_info": circuit_info,
        "measurement_is_hardware": False,
    }
    (outdir / "stage3_submission.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "stage3_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in ("exact_probability", "estimated_probability", "success_shots", "shots", "absolute_error", "standard_error_approx_at_exact_p", "backend", "seed"):
            w.writerow([key, result[key]])
    print(f"Selected customers (q0..q5): {ids}")
    print(f"Exact P(>=3) = {exact:.12f} ({100*exact:.4f}%)")
    print(f"{args.backend} shots: {yes}/{args.shots} = {estimate:.12f}")
    print(f"Absolute error = {abs(estimate-exact):.8f}; Wilson 95% CI = [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Saved files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
