"""Classical probabilistic baselines (all self-tuning via inner CV on log loss)."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline

from .. import config
from ..features import compact_preprocessor, onehot_preprocessor, selected_feature_names
from .base import KernelLogisticRegression, grid_iter, inner_cv_splits, pooled_logloss


class PriorModel:
    name = "prior"
    def fit(self, X, y):
        self.p_ = float(np.mean(y)); self.best_params_ = {}; return self
    def predict_proba(self, X):
        p = np.full(len(X), self.p_); return np.c_[1 - p, p]


class _PipelineTuner:
    """Tune an sklearn pipeline factory over a grid with inner stratified CV (log loss)."""
    name = "base"
    grid: dict = {}

    def make(self, **params):
        raise NotImplementedError

    def fit(self, X, y):
        y = np.asarray(y)
        splits = inner_cv_splits(y)
        scores = {}
        for params in grid_iter(self.grid):
            oof = np.zeros(len(y))
            for a, b in splits:
                m = self.make(**params).fit(X.iloc[a], y[a])
                oof[b] = m.predict_proba(X.iloc[b])[:, 1]
            scores[tuple(params.items())] = pooled_logloss(y, oof)
        best = min(scores, key=scores.get)
        self.best_params_ = dict(best)
        self.inner_scores_ = {str(dict(k)): v for k, v in scores.items()}
        self.model_ = self.make(**self.best_params_).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class LRCompact(_PipelineTuner):
    """Classical baseline: engineered features -> top-k L1 selection -> L2 logistic regression."""
    name = "lr_compact"
    grid = config.GRIDS["lr_compact"]

    def make(self, k, C):
        return Pipeline([("pre", compact_preprocessor(k)), ("clf", LogisticRegression(C=C, max_iter=5000))])

    def selected_features(self):
        return selected_feature_names(self.model_.named_steps["pre"])

    def coefficients(self):
        pre = self.model_.named_steps["pre"]
        return dict(zip(selected_feature_names(pre), self.model_.named_steps["clf"].coef_[0]))


class LROneHot(_PipelineTuner):
    """Reference: plain one-hot logistic regression on all raw fields."""
    name = "lr_onehot"
    grid = config.GRIDS["lr_onehot"]

    def make(self, C):
        return Pipeline([("pre", onehot_preprocessor()), ("clf", LogisticRegression(C=C, max_iter=5000))])


class RandomForest(_PipelineTuner):
    """Non-linear classical baseline (fixed, conservative hyper-parameters)."""
    name = "rf"
    grid = {"min_samples_leaf": [5]}

    def make(self, min_samples_leaf):
        return Pipeline([("pre", compact_preprocessor(14)),
                         ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=min_samples_leaf,
                                                        max_features=0.3, random_state=config.SEEDS["global"],
                                                        n_jobs=1))])


class _KernelKLR:
    """Shared machinery for kernel logistic regression on the 6 selected features.

    Preprocessing (features, scaler, selector, clip) is fitted on the data passed to
    ``fit``; ``gamma`` rescales the standardised inputs before the kernel is evaluated.
    Hyper-parameters (gamma, C) are tuned by inner CV; the kernel for a given gamma is
    computed once and sliced across inner folds (kernel values do not depend on labels).
    """
    name = "kernel"
    grid: dict = {}

    def embed(self, Z):  # -> representation used by self.kernel
        raise NotImplementedError

    def kernel(self, A, B):
        raise NotImplementedError

    def fit(self, X, y):
        y = np.asarray(y)
        self.pre_ = compact_preprocessor(config.N_QUBITS_CLASSIFIER, clip=config.CLIP_Z).fit(X, y)
        Z = self.pre_.transform(X)
        splits = inner_cv_splits(y)
        scores, cache = {}, {}
        for g in self.grid["gamma"]:
            E = self.embed(Z * g)
            K = self.kernel(E, E)
            cache[g] = (E, K)
            for C in self.grid["C"]:
                oof = np.zeros(len(y))
                for a, b in splits:
                    m = KernelLogisticRegression(C).fit(K[np.ix_(a, a)], y[a])
                    oof[b] = m.predict_proba_from_kernel(K[np.ix_(b, a)])
                scores[(g, C)] = pooled_logloss(y, oof)
        g, C = min(scores, key=scores.get)
        self.best_params_ = {"gamma": g, "C": C}
        self.inner_scores_ = {str(k): v for k, v in scores.items()}
        self.E_train_, K = cache[g]
        self.klr_ = KernelLogisticRegression(C).fit(K, y)
        self.K_train_ = K
        return self

    def predict_proba(self, X):
        Z = self.pre_.transform(X) * self.best_params_["gamma"]
        Kx = self.kernel(self.embed(Z), self.E_train_)
        p = self.klr_.predict_proba_from_kernel(Kx)
        return np.c_[1 - p, p]

    def selected_features(self):
        return selected_feature_names(self.pre_)


class RBFKernelLR(_KernelKLR):
    """Classical control for the quantum kernel: same features, same KLR, Gaussian kernel."""
    name = "rbf_klr"
    grid = config.GRIDS["rbf_klr"]

    def embed(self, Z):
        return Z

    def kernel(self, A, B):
        return rbf_kernel(A, B, gamma=1.0)
