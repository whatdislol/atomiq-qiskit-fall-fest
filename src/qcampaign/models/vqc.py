"""Data re-uploading variational quantum classifier (hybrid QNN).

Circuit (n qubits, L layers), identical to :func:`build_qiskit_circuit`:

    for l in 1..L:
        for q in 0..n-1:  RY(w[l,q] * x[q] + b[l,q])  then  RZ(theta[l,q])
        CZ chain (q, q+1)
    measure <Z_q> for every qubit

Readout head (classical, 7 parameters for n=6):
    p(y=1 | x) = sigmoid( sum_q a_q <Z_q> + c )

Training minimises the mean binary cross-entropy (+ L2 on input weights, small L2 on head)
with L-BFGS. Exact gradients are obtained with adjoint differentiation on a
vectorised statevector simulator (batch of samples at once). The simulator uses
Qiskit's little-endian convention and is verified against
``qiskit.quantum_info.Statevector`` in ``tests/test_quantum_equivalence.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class _Sim:
    """Minimal batched statevector simulator for RY / RZ / CZ-chain circuits."""

    def __init__(self, n: int):
        self.n = n
        self.D = 2 ** n
        idx = np.arange(self.D)
        self.i0 = [idx[(idx >> q) & 1 == 0] for q in range(n)]
        self.i1 = [i | (1 << q) for q, i in enumerate(self.i0)]
        ph = np.ones(self.D)
        for q in range(n - 1):
            ph *= np.where(((idx >> q) & 1) & ((idx >> (q + 1)) & 1), -1.0, 1.0)
        self.cz = ph
        # z[q, k] = <k| Z_q |k>
        self.z = np.stack([1.0 - 2.0 * ((idx >> q) & 1) for q in range(n)])

    def ry(self, psi, q, t):  # t: (B,)
        c, s = np.cos(t / 2)[:, None], np.sin(t / 2)[:, None]
        a, b = psi[:, self.i0[q]], psi[:, self.i1[q]]
        out = psi.copy()
        out[:, self.i0[q]] = c * a - s * b
        out[:, self.i1[q]] = s * a + c * b
        return out

    def dry(self, psi, q, t):  # derivative of RY(t) applied to psi
        c, s = np.cos(t / 2)[:, None], np.sin(t / 2)[:, None]
        a, b = psi[:, self.i0[q]], psi[:, self.i1[q]]
        out = np.empty_like(psi)
        out[:, self.i0[q]] = 0.5 * (-s * a - c * b)
        out[:, self.i1[q]] = 0.5 * (c * a - s * b)
        return out

    def rz(self, psi, q, t):  # t scalar
        out = psi.copy()
        out[:, self.i0[q]] *= np.exp(-0.5j * t)
        out[:, self.i1[q]] *= np.exp(0.5j * t)
        return out

    def drz(self, psi, q, t):
        out = np.empty_like(psi)
        out[:, self.i0[q]] = -0.5j * np.exp(-0.5j * t) * psi[:, self.i0[q]]
        out[:, self.i1[q]] = 0.5j * np.exp(0.5j * t) * psi[:, self.i1[q]]
        return out


class DataReuploadingVQC:
    """Hybrid quantum-classical probabilistic classifier (sklearn-like API on arrays)."""

    def __init__(self, n_qubits: int = 6, layers: int = 2, l2: float = 1e-3,
                 input_scale: float = 0.5, maxiter: int = 300, seed: int = 42, l2_head: float = 1e-4):
        self.n_qubits, self.layers, self.l2, self.l2_head = n_qubits, layers, l2, l2_head
        self.input_scale, self.maxiter, self.seed = input_scale, maxiter, seed
        self.sim = _Sim(n_qubits)

    # ---------------- parameters ----------------
    def _split(self, P):
        n, L = self.n_qubits, self.layers
        W = P[: n * L].reshape(L, n)
        Bb = P[n * L: 2 * n * L].reshape(L, n)
        Th = P[2 * n * L: 3 * n * L].reshape(L, n)
        a = P[3 * n * L: 3 * n * L + n]
        c = P[-1]
        return W, Bb, Th, a, c

    def n_params(self):
        return 3 * self.n_qubits * self.layers + self.n_qubits + 1

    # ---------------- forward / adjoint ----------------
    def _forward(self, P, X, keep=False):
        sim, n, L = self.sim, self.n_qubits, self.layers
        W, Bb, Th, a, c = self._split(P)
        B = X.shape[0]
        psi = np.zeros((B, sim.D), complex)
        psi[:, 0] = 1.0
        ops = []
        for l in range(L):
            for q in range(n):
                t = W[l, q] * X[:, q] + Bb[l, q]
                psi = sim.ry(psi, q, t)
                ops.append(("ry", l, q, t))
                psi = sim.rz(psi, q, Th[l, q])
                ops.append(("rz", l, q, Th[l, q]))
            psi = psi * sim.cz
            ops.append(("cz", l, None, None))
        prob = np.abs(psi) ** 2
        Z = prob @ sim.z.T  # (B, n)
        s = Z @ a + c
        return (Z, s, psi, ops) if keep else (Z, s)

    def _loss_grad(self, P, X, y):
        sim, n, L = self.sim, self.n_qubits, self.layers
        W, Bb, Th, a, c = self._split(P)
        Z, s, psi, ops = self._forward(P, X, keep=True)
        B = len(y)
        p = expit(s)
        loss = np.mean(np.logaddexp(0.0, s) - y * s) + self.l2 * np.sum(W ** 2) + self.l2_head * np.sum(a ** 2)
        r = (p - y) / B  # dL/ds_b
        g = np.zeros_like(P)
        g[3 * n * L: 3 * n * L + n] = Z.T @ r + 2 * self.l2_head * a
        g[-1] = r.sum()
        # adjoint pass for d s_b / d angle, observable O = sum_q a_q Z_q (diagonal)
        o = a @ sim.z  # (D,)
        lam = psi * o[None, :]
        phi = psi
        gW = np.zeros((L, n)); gB = np.zeros((L, n)); gT = np.zeros((L, n))
        for kind, l, q, t in reversed(ops):
            if kind == "cz":
                phi = phi * sim.cz
                lam = lam * sim.cz
                continue
            if kind == "ry":
                phi = sim.ry(phi, q, -t)
                ds = 2.0 * np.real(np.sum(lam.conj() * sim.dry(phi, q, t), axis=1))  # (B,)
                gW[l, q] = np.sum(r * ds * X[:, q])
                gB[l, q] = np.sum(r * ds)
                lam = sim.ry(lam, q, -t)
            else:  # rz
                phi = sim.rz(phi, q, -t)
                ds = 2.0 * np.real(np.sum(lam.conj() * sim.drz(phi, q, t), axis=1))
                gT[l, q] = np.sum(r * ds)
                lam = sim.rz(lam, q, -t)
        g[: n * L] = gW.ravel() + 2 * self.l2 * W.ravel()
        g[n * L: 2 * n * L] = gB.ravel()
        g[2 * n * L: 3 * n * L] = gT.ravel()
        return loss, g

    # ---------------- public API ----------------
    def fit(self, X, y):
        X = np.asarray(X, float) * self.input_scale
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        n, L = self.n_qubits, self.layers
        prior = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        P0 = np.r_[
            1.0 + 0.1 * rng.standard_normal(n * L),          # input weights
            np.pi / 2 + 0.1 * rng.standard_normal(n * L),    # input biases (start near equator)
            0.1 * rng.standard_normal(n * L),                # RZ angles
            np.zeros(n),                                     # head weights
            np.log(prior / (1 - prior)),                     # head bias = prior log-odds
        ]
        res = minimize(self._loss_grad, P0, args=(X, y), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter})
        self.params_ = res.x
        self.opt_result_ = {"nit": int(res.nit), "fun": float(res.fun), "success": bool(res.success)}
        return self

    def z_expectations(self, X):
        return self._forward(self.params_, np.asarray(X, float) * self.input_scale)[0]

    def predict_proba(self, X):
        _, s = self._forward(self.params_, np.asarray(X, float) * self.input_scale)
        p = expit(s)
        return np.c_[1 - p, p]

    # ---------------- Qiskit circuit ----------------
    def build_qiskit_circuit(self, x=None, measure=False):
        """Return the Qiskit circuit. If ``x`` is given, parameters are bound for that sample."""
        n, L = self.n_qubits, self.layers
        xin = ParameterVector("x", n)
        qc = QuantumCircuit(n, name="reupload_vqc")
        if hasattr(self, "params_"):
            W, Bb, Th, _, _ = self._split(self.params_)
        else:
            W = ParameterVector("w", n * L); Bb = ParameterVector("b", n * L); Th = ParameterVector("t", n * L)
            W = np.array(list(W), dtype=object).reshape(L, n); Bb = np.array(list(Bb), dtype=object).reshape(L, n)
            Th = np.array(list(Th), dtype=object).reshape(L, n)
        for l in range(L):
            for q in range(n):
                qc.ry(W[l, q] * xin[q] + Bb[l, q], q)
                qc.rz(Th[l, q], q)
            for q in range(n - 1):
                qc.cz(q, q + 1)
        if measure:
            qc.measure_all()
        if x is not None:
            qc = qc.assign_parameters(dict(zip(xin, np.asarray(x, float) * self.input_scale)))
        return qc
