"""QUBO construction and QAOA solvers for the constrained selection problem.

Problem (Stage 2)
-----------------
    maximise   sum_i v_i x_i,            v_i = p_i * D_i - c_i
    s.t.       sum_i x_i            = K            (exactly six customers)
               sum_i c_i x_i       <= budget
               sum_i h_i x_i       <= capacity     x_i in {0, 1}

QUBO (minimised, all penalties quadratic):
    E(x, s) = - sum_i v_i x_i
              + A_K  (sum_i x_i - K)^2
              + A_B  (sum_i c_i x_i + sum_j w^B_j sB_j - budget)^2     [only if the budget can bind]
              + A_H  (sum_i h_i x_i + sum_j w^H_j sH_j - capacity)^2   [only if the capacity can bind]

Slack variables use a *bounded* binary encoding w = (1, 2, 4, ..., 2^{m-2}, U - (2^{m-1} - 1)) so the
slack range is exactly [0, U], with U = limit - (minimum possible usage of K customers).

Penalty weights (both rules are provable and never use the optimum V*):
  * general: every residual is an integer, so a violated constraint costs at least A. Any violating
    state has energy >= -sum(v+) + A while the optimum has -V* >= -sum(v+); A > sum|v_i| suffices.
  * cardinality only (after presolve): a state with |x| = K + d gains at most |d| * max|v| in
    objective but pays A d^2, so A > max|v_i| suffices (much tighter -> better QAOA landscape).
We use PENALTY_SAFETY (1.5) times the bound and verify the ground state by exhaustive enumeration.

Two QAOA variants (Farhi et al. 2014; Hadfield et al. 2019):
  * "x_mixer":  |+>^n start, transverse-field mixer, all constraints as penalties.
  * "xy_mixer": decision register starts in the Dicke state |D^N_K> and evolves under a
                parity-ring XY mixer, which conserves Hamming weight -> sum x_i = K holds
                *by construction* (no K-penalty needed); slack qubits keep an X mixer.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


# ============================================================================ problem / QUBO
@dataclass
class SelectionProblem:
    ids: list
    v: np.ndarray           # expected net value
    cost: np.ndarray
    cap: np.ndarray
    K: int
    budget: float
    capacity: float

    @property
    def n(self):
        return len(self.ids)

    def evaluate(self, x):
        x = np.asarray(x, int)
        return {"n_selected": int(x.sum()), "cost": float(self.cost @ x), "capacity": float(self.cap @ x),
                "objective": float(self.v @ x),
                "feasible": bool(x.sum() == self.K and self.cost @ x <= self.budget + 1e-9
                                 and self.cap @ x <= self.capacity + 1e-9)}

    def brute_force(self):
        """Exhaustive search over all 2^N assignments (N = 10 -> 1024)."""
        rows = []
        for bits in itertools.product([0, 1], repeat=self.n):
            e = self.evaluate(bits)
            rows.append({"x": bits, **e})
        feas = [r for r in rows if r["feasible"]]
        feas.sort(key=lambda r: -r["objective"])
        return feas, len(rows)


def bounded_slack_weights(U: int) -> list[int]:
    if U <= 0:
        return []
    m = int(np.floor(np.log2(U))) + 1
    w = [2 ** j for j in range(m - 1)]
    w.append(U - (2 ** (m - 1) - 1))
    assert sum(w) == U and all(x > 0 for x in w)
    return w


@dataclass
class QUBO:
    problem: SelectionProblem
    Q: np.ndarray                       # upper-triangular quadratic coefficients
    lin: np.ndarray
    const: float
    n_vars: int
    slack: dict = field(default_factory=dict)  # name -> (start index, weights)
    penalties: dict = field(default_factory=dict)
    redundant: dict = field(default_factory=dict)
    include_k_penalty: bool = True

    def energy_vector(self) -> np.ndarray:
        """E for every computational basis state; index = sum_q x_q 2^q (Qiskit little-endian)."""
        n = self.n_vars
        idx = np.arange(2 ** n)
        X = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
        return self.const + X @ self.lin + ((X @ self.Q) * X).sum(1)

    def decode(self, bitstring: str):
        """Qiskit bitstring (qubit 0 rightmost) -> decision vector x of length N."""
        b = bitstring.replace(" ", "")[::-1]
        return np.array([int(b[i]) for i in range(self.problem.n)])

    def ising(self):
        """Return (h, J, offset) with x = (1 - z)/2:  E = offset + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
        n = self.n_vars
        h = np.zeros(n); J = np.zeros((n, n)); off = self.const
        for i in range(n):
            off += self.lin[i] / 2; h[i] -= self.lin[i] / 2
        for i in range(n):
            for j in range(i + 1, n):
                q = self.Q[i, j] + self.Q[j, i]
                off += q / 4; h[i] -= q / 4; h[j] -= q / 4; J[i, j] += q / 4
            q = self.Q[i, i]  # x_i^2 = x_i
            off += q / 2; h[i] -= q / 2
        return h, J, off


def build_qubo(prob: SelectionProblem, safety: float = 1.5, include_k_penalty: bool = True,
               presolve: bool = True, penalty: float | None = None) -> QUBO:
    """``presolve=True`` drops an inequality only when it is *provably* implied by sum x = K
    (the K largest coefficients already fit). ``presolve=False`` keeps both inequalities with
    slack registers - the general formulation, used to demonstrate the penalty machinery."""
    N, K = prob.n, prob.K
    # which inequality constraints can bind for a K-subset?
    max_cost = np.sort(prob.cost)[-K:].sum(); min_cost = np.sort(prob.cost)[:K].sum()
    max_cap = np.sort(prob.cap)[-K:].sum(); min_cap = np.sort(prob.cap)[:K].sum()
    redundant = {"budget": bool(max_cost <= prob.budget), "capacity": bool(max_cap <= prob.capacity)}
    terms = []  # list of (coeff vector over decision vars, slack weights, rhs, weight, name)
    if include_k_penalty:
        terms.append((np.ones(N), [], K, None, "cardinality"))
    if not (presolve and redundant["budget"]):
        U = int(prob.budget - min_cost)
        terms.append((prob.cost.astype(float), bounded_slack_weights(U), prob.budget, None, "budget"))
    if not (presolve and redundant["capacity"]):
        U = int(prob.capacity - min_cap)
        terms.append((prob.cap.astype(float), bounded_slack_weights(U), prob.capacity, None, "capacity"))
    # ---- penalty weight (provable, never uses the optimum) -------------------------------
    only_cardinality = [t[4] for t in terms] == ["cardinality"]
    if not terms:
        A, rule = 0.0, "no penalty terms: cardinality enforced exactly by the Dicke state + XY mixer"
    elif penalty is not None:
        A, rule = float(penalty), "user-specified"
    elif only_cardinality:
        # moving k bits away from |x| = K changes the objective by at most k*max|v| but adds A*k^2,
        # so A > max|v| already makes every wrong-weight state worse than the best weight-K state
        A, rule = safety * float(np.max(np.abs(prob.v))), f"A = {safety} * max|v_i| (cardinality-only bound)"
    else:
        # integer residuals => any violation costs >= A; A > sum|v| >= sum(v+) - V* suffices
        A, rule = safety * float(np.sum(np.abs(prob.v))), f"A = {safety} * sum|v_i| (general bound)"
    terms = [(a, w, rhs, A, name) for a, w, rhs, _, name in terms]
    n_vars = N + sum(len(t[1]) for t in terms)
    Q = np.zeros((n_vars, n_vars)); lin = np.zeros(n_vars); const = 0.0
    lin[:N] -= prob.v
    slack, start = {}, N
    for a, w, rhs, lam, name in terms:
        coef = np.zeros(n_vars); coef[:N] = a
        if w:
            coef[start:start + len(w)] = w
            slack[name] = (start, w)
            start += len(w)
        # lam * (coef.x - rhs)^2 = lam * [sum_i coef_i^2 x_i + 2 sum_{i<j} coef_i coef_j x_i x_j - 2 rhs coef.x + rhs^2]
        nz = np.nonzero(coef)[0]
        for i in nz:
            lin[i] += lam * (coef[i] ** 2 - 2 * rhs * coef[i])
        for i, j in itertools.combinations(nz, 2):
            Q[i, j] += 2 * lam * coef[i] * coef[j]
        const += lam * rhs ** 2
    return QUBO(prob, Q, lin, const, n_vars, slack,
                {"A": A, "rule": rule,
                 "terms": [t[4] for t in terms],
                 "presolve_bounds": {"max_cost_K": float(max_cost), "min_cost_K": float(min_cost),
                                     "max_capacity_K": float(max_cap), "min_capacity_K": float(min_cap)}},
                redundant, include_k_penalty)


def verify_ground_state(qubo: QUBO, feasible_sorted):
    """Exhaustively check that the QUBO minimum decodes to the constrained optimum."""
    E = qubo.energy_vector()
    order = np.argsort(E)
    best = order[0]
    bits = format(best, f"0{qubo.n_vars}b")
    x = qubo.decode(bits)
    opt = feasible_sorted[0]
    ok = tuple(x) == tuple(opt["x"])
    # energy gap between the ground state and the lowest-energy state that is NOT a feasible optimum
    first_bad = next(i for i in order if tuple(qubo.decode(format(i, f"0{qubo.n_vars}b"))) != tuple(opt["x"])
                     or not qubo.problem.evaluate(qubo.decode(format(i, f"0{qubo.n_vars}b")))["feasible"])
    return {"ground_state_index": int(best), "ground_state_bitstring": bits, "ground_energy": float(E[best]),
            "ground_state_is_constrained_optimum": bool(ok), "spectral_gap_to_next_distinct_solution":
            float(E[first_bad] - E[best]), "n_states_checked": int(len(E))}


# ============================================================================ Dicke state
def _scs(n: int, k: int) -> QuantumCircuit:
    """Split-and-cyclic-shift unitary SCS_{n,k} (Bartschi & Eidenbenz, 2019) on qubits 0..n-1."""
    qc = QuantumCircuit(n)
    qc.cx(n - 2, n - 1)
    qc.cry(2 * np.arccos(np.sqrt(1 / n)), n - 1, n - 2)
    qc.cx(n - 2, n - 1)
    for l in range(2, k + 1):
        qc.cx(n - l - 1, n - 1)
        qc.append(_ccry(2 * np.arccos(np.sqrt(l / n))), [n - 1, n - l, n - l - 1])
        qc.cx(n - l - 1, n - 1)
    return qc


def _ccry(theta):
    sub = QuantumCircuit(1)
    sub.ry(theta, 0)
    return sub.to_gate(label="ry").control(2)


def dicke_state_circuit(n: int, k: int) -> QuantumCircuit:
    """Deterministic preparation of |D^n_k> = C(n,k)^{-1/2} sum_{|x|=k} |x>  with O(nk) gates."""
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    qc.x(range(n - k, n))
    for m in range(n, k, -1):
        qc.compose(_scs(m, k), qubits=list(range(m)), inplace=True)
    for m in range(k, 1, -1):
        qc.compose(_scs(m, m - 1), qubits=list(range(m)), inplace=True)
    return qc


def dicke_vector(n: int, k: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    w = np.array([bin(i).count("1") for i in idx])
    v = (w == k).astype(complex)
    return v / np.sqrt(comb(n, k))


# ============================================================================ fast exact simulation
class QAOASimulator:
    """Exact statevector QAOA for a diagonal cost (energy vector E) with an X or XY mixer.

    Used for the classical outer-loop optimisation (thousands of evaluations). The final
    parameters are executed as genuine Qiskit circuits on Aer (see ``qaoa_circuit``); the two
    are checked to agree to numerical precision in tests/test_quantum_equivalence.py.
    """

    def __init__(self, E: np.ndarray, n_qubits: int, mixer: str = "x", n_decision: int | None = None,
                 K: int | None = None):
        self.E, self.n, self.mixer = E, n_qubits, mixer
        self.nd = n_decision if n_decision is not None else n_qubits
        if mixer == "x":
            self.psi0 = np.full(2 ** n_qubits, 2 ** (-n_qubits / 2), complex)
        else:
            d = dicke_vector(self.nd, K)
            ns = n_qubits - self.nd
            plus = np.full(2 ** ns, 2 ** (-ns / 2), complex)
            self.psi0 = np.kron(plus, d)  # slack qubits are the high-order qubits
        self.pairs = xy_ring_pairs(self.nd)

    def _rx_all(self, psi, beta, qubits):
        c, s = np.cos(beta), -1j * np.sin(beta)  # RX(2 beta)
        for q in qubits:
            t = psi.reshape(2 ** (self.n - 1 - q), 2, 2 ** q)
            a, b = t[:, 0, :], t[:, 1, :]
            psi = np.stack([c * a + s * b, s * a + c * b], axis=1).reshape(-1)
        return psi

    def _xy(self, psi, beta):
        # exp(-i beta (XX + YY)/2) on each pair: rotation inside the {|01>, |10>} subspace
        c, s = np.cos(beta), -1j * np.sin(beta)
        for (i, j) in self.pairs:
            lo, hi = min(i, j), max(i, j)
            t = psi.reshape(2 ** (self.n - 1 - hi), 2, 2 ** (hi - lo - 1), 2, 2 ** lo).copy()
            a, b = t[:, 0, :, 1, :].copy(), t[:, 1, :, 0, :].copy()   # (hi=0, lo=1) and (hi=1, lo=0)
            t[:, 0, :, 1, :] = c * a + s * b
            t[:, 1, :, 0, :] = s * a + c * b
            psi = t.reshape(-1)
        return psi

    def state(self, gammas, betas):
        psi = self.psi0.copy()
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.E)
            if self.mixer == "x":
                psi = self._rx_all(psi, b, range(self.n))
            else:
                psi = self._xy(psi, b)
                if self.n > self.nd:
                    psi = self._rx_all(psi, b, range(self.nd, self.n))
        return psi

    def probs(self, gammas, betas):
        return np.abs(self.state(gammas, betas)) ** 2


def xy_ring_pairs(n: int):
    """Parity-ordered ring: even pairs, odd pairs, then the closing pair (n-1, 0)."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    close = [(n - 1, 0)] if n > 2 else []
    return even + odd + close


def cvar(E, p, alpha, order=None):
    """Conditional value-at-risk of the energy distribution (mean of the lowest alpha mass).
    ``order`` = cached argsort(E) (E is fixed during an optimisation)."""
    if alpha >= 1:
        return float(E @ p)
    o = np.argsort(E) if order is None else order
    cp = np.cumsum(p[o])
    k = np.searchsorted(cp, alpha)
    mass_before = cp[k - 1] if k > 0 else 0.0
    val = (E[o[:k]] @ p[o[:k]] + E[o[k]] * (alpha - mass_before)) / alpha
    return float(val)


# ============================================================================ Qiskit circuits
def qaoa_circuit(qubo: QUBO, gammas, betas, mixer: str = "x", measure: bool = True) -> QuantumCircuit:
    """Build the QAOA circuit with Ising cost layer exp(-i gamma H_C) (global phase dropped).

    H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j  ->  RZ(2 gamma h_i), RZZ(2 gamma J_ij).
    """
    n, nd, K = qubo.n_vars, qubo.problem.n, qubo.problem.K
    h, J, _ = qubo.ising()
    qc = QuantumCircuit(n, name=f"QAOA_{mixer}_p{len(gammas)}")
    if mixer == "x":
        qc.h(range(n))
    else:
        qc.compose(dicke_state_circuit(nd, K), qubits=list(range(nd)), inplace=True)
        if n > nd:
            qc.h(range(nd, n))
    for g, b in zip(gammas, betas):
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * g * J[i, j], i, j)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        if mixer == "x":
            for i in range(n):
                qc.rx(2 * b, i)
        else:
            for (i, j) in xy_ring_pairs(nd):
                qc.append(XXPlusYYGate(2 * b, 0.0), [i, j])
            for i in range(nd, n):
                qc.rx(2 * b, i)
    if measure:
        qc.measure_all()
    return qc
