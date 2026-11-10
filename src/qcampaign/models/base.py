"""Shared helpers: kernel logistic regression and inner-CV hyper-parameter tuning."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .. import config


class KernelLogisticRegression:
    """Binary kernel logistic regression on a precomputed kernel.

    Minimises  sum_i [log(1 + e^{f_i}) - y_i f_i] + (1 / 2C) a^T K a,   f = K a + b
    (representer theorem). Optimising the log-likelihood directly yields probabilities
    without a separate Platt step, which matters because Stages 2-3 consume them.
    """

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, K, y):
        y = np.asarray(y, float)
        n = len(y)
        lam = 1.0 / self.C
        prior = np.clip(y.mean(), 1e-3, 1 - 1e-3)

        def obj(w):
            a, b = w[:-1], w[-1]
            Ka = K @ a
            s = Ka + b
            p = expit(s)
            loss = np.sum(np.logaddexp(0.0, s) - y * s) + 0.5 * lam * a @ Ka
            grad = np.r_[K @ (p - y) + lam * Ka, np.sum(p - y)]
            return loss, grad

        w0 = np.r_[np.zeros(n), np.log(prior / (1 - prior))]
        res = minimize(obj, w0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        self.alpha_, self.b_ = res.x[:-1], res.x[-1]
        return self

    def predict_proba_from_kernel(self, K_test_train):
        return expit(K_test_train @ self.alpha_ + self.b_)


def grid_iter(grid: dict):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def inner_cv_splits(y, seed=None):
    seed = config.SEEDS["cv_inner"] if seed is None else seed
    return list(StratifiedKFold(config.INNER_SPLITS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def best_by_logloss(scores: dict):
    """scores: {param_tuple: [oof predictions]} -> best params (lowest pooled inner log loss)."""
    return min(scores, key=lambda k: scores[k])


def pooled_logloss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
