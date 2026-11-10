"""Quantum / hybrid Stage 1 classifiers.

* ``QuantumKernelKLR`` - fidelity quantum kernel (IQP-style feature map, 6 qubits,
  bandwidth-scaled inputs) evaluated with qiskit-machine-learning's
  ``FidelityStatevectorKernel``, followed by kernel logistic regression.
* ``QuantumVQC`` - data re-uploading variational circuit with a classical readout head
  (see ``models/vqc.py``).
* ``HybridEnsemble`` - equal-weight average of the classical LR baseline and the two
  quantum models' probabilities.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

from .. import config
from ..features import compact_preprocessor, selected_feature_names
from .base import grid_iter, inner_cv_splits, pooled_logloss
from .classical import _KernelKLR
from .vqc import DataReuploadingVQC


def iqp_feature_map(n: int = config.N_QUBITS_CLASSIFIER, reps: int = config.IQP_REPS) -> QuantumCircuit:
    """IQP-style map: [H^n . prod_i RZ(2 x_i) . prod_<i,i+1> RZZ(2 x_i x_{i+1})]^reps.

    Linear (nearest-neighbour) entanglement keeps the transpiled circuit SWAP-free on
    heavy-hex / square-lattice hardware. Inputs are standardised features times the
    bandwidth gamma (Shaydulin & Wild 2022), which counters exponential kernel concentration.
    """
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n, name="iqp_map")
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.rz(2 * x[i], i)
        for i in range(n - 1):
            qc.rzz(2 * x[i] * x[i + 1], i, i + 1)
    return qc


def kernel_estimation_circuit(x1, x2, feature_map: QuantumCircuit | None = None) -> QuantumCircuit:
    """Compute-uncompute circuit U(x2)^dagger U(x1): P(|0...0>) = k(x1, x2). For hardware runs."""
    fm = feature_map or iqp_feature_map(len(x1))
    params = list(fm.parameters)
    qc = fm.assign_parameters(dict(zip(params, x1))).compose(fm.assign_parameters(dict(zip(params, x2))).inverse())
    qc.measure_all()
    return qc


class QuantumKernelKLR(_KernelKLR):
    name = "qk_klr"
    grid = config.GRIDS["qk_klr"]

    def __init__(self):
        self.feature_map = iqp_feature_map()
        self.qkernel = FidelityStatevectorKernel(feature_map=self.feature_map, cache_size=None)

    def embed(self, Z):
        return np.asarray(Z, float)

    def kernel(self, A, B):
        return self.qkernel.evaluate(A, B)


class QuantumVQC:
    name = "vqc"
    grid = config.GRIDS["vqc"]

    def make(self, l2):
        return DataReuploadingVQC(n_qubits=config.N_QUBITS_CLASSIFIER, l2=l2, seed=config.SEEDS["vqc_init"],
                                  **config.VQC_FIXED)

    def fit(self, X, y):
        y = np.asarray(y)
        self.pre_ = compact_preprocessor(config.N_QUBITS_CLASSIFIER, clip=config.CLIP_Z).fit(X, y)
        Z = self.pre_.transform(X)
        splits = inner_cv_splits(y)
        scores = {}
        for params in grid_iter(self.grid):
            oof = np.zeros(len(y))
            for a, b in splits:
                oof[b] = self.make(**params).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
            scores[params["l2"]] = pooled_logloss(y, oof)
        l2 = min(scores, key=scores.get)
        self.best_params_ = {"l2": l2}
        self.inner_scores_ = {str(k): v for k, v in scores.items()}
        self.vqc_ = self.make(l2).fit(Z, y)
        return self

    def predict_proba(self, X):
        return self.vqc_.predict_proba(self.pre_.transform(X))

    def selected_features(self):
        return selected_feature_names(self.pre_)


class HybridEnsemble:
    """Equal-weight probability average of fitted component models (no extra fitting)."""
    name = "hybrid_ensemble"
    components = ("lr_compact", "qk_klr", "vqc")

    def __init__(self, fitted: dict):
        self.members = {k: fitted[k] for k in self.components}
        self.best_params_ = {"members": list(self.components), "weights": "equal"}

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members.values()], axis=0)
        return np.c_[1 - p, p]
