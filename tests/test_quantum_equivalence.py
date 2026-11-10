"""Correctness checks for every claim of exactness made in the report.  Run:  python -m pytest -q tests"""
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qcampaign.metrics import rank_order  # noqa: E402
from qcampaign.models.vqc import DataReuploadingVQC  # noqa: E402
from qcampaign.qaoa import (QAOASimulator, SelectionProblem, bounded_slack_weights, build_qubo,  # noqa: E402
                            dicke_state_circuit, dicke_vector, qaoa_circuit, verify_ground_state)
from qcampaign.stage3 import counting_oracle, exact_at_least, poisson_binomial_pmf, prep_circuit  # noqa: E402


def toy_problem(seed=3, N=7, K=4):
    rng = np.random.default_rng(seed)
    return SelectionProblem([f"C{i}" for i in range(N)], rng.uniform(-1, 8, N), rng.integers(1, 5, N).astype(float),
                            rng.integers(1, 5, N).astype(float), K, 13.0, 10.0)


def test_vqc_adjoint_gradient_matches_finite_differences():
    m = DataReuploadingVQC(4, 2, 1e-2)
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((12, 4)), (rng.random(12) < .3).astype(float)
    P = rng.standard_normal(m.n_params())
    _, g = m._loss_grad(P, X, y)
    num = np.array([(m._loss_grad(P + 1e-6 * e, X, y)[0] - m._loss_grad(P - 1e-6 * e, X, y)[0]) / 2e-6
                    for e in np.eye(len(P))])
    assert np.abs(num - g).max() < 1e-7


def test_vqc_simulator_matches_qiskit_statevector():
    m = DataReuploadingVQC(4, 2, input_scale=1.0)
    rng = np.random.default_rng(1)
    m.params_ = rng.standard_normal(m.n_params())
    X = rng.standard_normal((3, 4))
    Z = m.z_expectations(X)
    for i in range(3):
        sv = Statevector(m.build_qiskit_circuit(X[i]))
        zq = [sv.expectation_value(SparsePauliOp("I" * (3 - q) + "Z" + "I" * q)).real for q in range(4)]
        assert np.allclose(zq, Z[i], atol=1e-12)


@pytest.mark.parametrize("n,k", [(4, 2), (6, 3), (10, 6)])
def test_dicke_state(n, k):
    sv = Statevector(dicke_state_circuit(n, k)).data
    assert abs(abs(np.vdot(dicke_vector(n, k), sv)) ** 2 - 1) < 1e-10


def test_bounded_slack_covers_range_exactly():
    for U in range(1, 40):
        w = bounded_slack_weights(U)
        reach = {sum(b * x for b, x in zip(bits, w)) for bits in np.ndindex(*([2] * len(w)))}
        assert reach == set(range(U + 1))


def test_qubo_ground_state_is_constrained_optimum_and_ising_exact():
    prob = toy_problem()
    feas, _ = prob.brute_force()
    q = build_qubo(prob, 1.5, include_k_penalty=True)
    assert verify_ground_state(q, feas)["ground_state_is_constrained_optimum"]
    E = q.energy_vector()
    h, J, off = q.ising()
    n = q.n_vars
    Z = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    assert np.allclose(off + Z @ h + np.einsum("bi,ij,bj->b", Z, J, Z), E, atol=1e-8)


@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_qaoa_simulator_matches_qiskit_circuit(mixer):
    prob = toy_problem(N=6, K=3)
    q = build_qubo(prob, 1.5, include_k_penalty=(mixer == "x"))
    sim = QAOASimulator(q.energy_vector(), q.n_vars, mixer, prob.n, prob.K)
    g, b = np.array([0.07, 0.11]), np.array([0.6, 0.3])
    psi = sim.state(g, b)
    sv = Statevector(qaoa_circuit(q, g, b, mixer, measure=False)).data
    assert abs(abs(np.vdot(psi, sv)) ** 2 - 1) < 1e-9
    if mixer == "xy":  # Hamming weight of the decision register is conserved
        w = np.array([bin(i & ((1 << prob.n) - 1)).count("1") for i in range(len(psi))])
        assert abs(np.sum(np.abs(psi[w == prob.K]) ** 2) - 1) < 1e-10


def test_stage3_circuit_and_oracle_are_exact():
    p = np.array([0.42, 0.35, 0.30, 0.28, 0.25, 0.22])
    exact = exact_at_least(p, 3)
    probs = Statevector(prep_circuit(p, measure=False)).probabilities()
    w = np.array([bin(i).count("1") for i in range(64)])
    assert abs(probs[w >= 3].sum() - exact) < 1e-12
    for i in range(6):  # qubit i carries customer i (little-endian marginal)
        assert abs(Statevector(prep_circuit(p, measure=False)).probabilities([i])[1] - p[i]) < 1e-12
    A = counting_oracle(p)
    sv = Statevector(A)
    assert abs(sv.probabilities([A.num_qubits - 1])[1] - exact) < 1e-12
    assert np.allclose(sv.probabilities([6, 7, 8])[:7], poisson_binomial_pmf(p), atol=1e-12)


def test_tie_break_lower_customer_id_first():
    ids = np.array(["C0010", "C0002", "C0005"])
    p = np.array([0.3, 0.3, 0.5])
    assert list(ids[rank_order(ids, p)]) == ["C0005", "C0002", "C0010"]


def test_poisson_binomial_against_enumeration():
    p = np.array([0.1, 0.5, 0.7, 0.2])
    tot = 0.0
    for bits in np.ndindex(2, 2, 2, 2):
        b = np.array(bits)
        pr = np.prod(np.where(b == 1, p, 1 - p))
        tot += pr * (b.sum() >= 2)
    assert abs(tot - exact_at_least(p, 2)) < 1e-12
