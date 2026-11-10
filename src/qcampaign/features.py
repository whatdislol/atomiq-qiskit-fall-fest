"""Label-free feature engineering shared by every Stage 1 model.

All learned quantities (the modal month, scaler statistics, the selected feature subset)
are fitted on the training fold only, inside sklearn pipelines, so cross-validation is
leak-free and the test set is never used for fitting anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import FEATURES

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
ENGINEERED = ["age", "age_sq", "bal_slog", "day", "log_campaign", "prev_contacted", "log_pdays",
              "log_previous", "pout_success", "pout_failure", "contact_unknown", "housing_yes",
              "loan_yes", "month_is_mode"]


class CampaignFeatures(BaseEstimator, TransformerMixin):
    """14 compact numeric features.

    * heavy-tailed counts/amounts are log-transformed (signed log for balance)
    * ``pdays = -1`` (never contacted) becomes an explicit indicator
    * low-cardinality categoricals become indicators of their informative level
    * ``month_is_mode`` flags the most frequent campaign month in the *training fold*
      (chosen by frequency, not by target rate -> label-free)
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.month_mode_ = X["month"].mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=X.index)
        d["age"] = X["age"].astype(float)
        d["age_sq"] = (X["age"] - 40.0) ** 2 / 100.0
        d["bal_slog"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
        d["day"] = X["day"].astype(float)
        d["log_campaign"] = np.log(X["campaign"].clip(lower=1))
        d["prev_contacted"] = (X["pdays"] >= 0).astype(float)
        d["log_pdays"] = np.log1p(X["pdays"].clip(lower=0))
        d["log_previous"] = np.log1p(X["previous"])
        d["pout_success"] = (X["poutcome"] == "success").astype(float)
        d["pout_failure"] = (X["poutcome"] == "failure").astype(float)
        d["contact_unknown"] = (X["contact"] == "unknown").astype(float)
        d["housing_yes"] = (X["housing"] == "yes").astype(float)
        d["loan_yes"] = (X["loan"] == "yes").astype(float)
        d["month_is_mode"] = (X["month"] == self.month_mode_).astype(float)
        return d[ENGINEERED]

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED)


def _clip(Z, c):
    return np.clip(Z, -c, c)


def compact_preprocessor(k: int, clip: float | None = None) -> Pipeline:
    """engineered features -> standardise -> L1-logistic top-k selection (-> optional clip)."""
    steps = [
        ("features", CampaignFeatures()),
        ("scale", StandardScaler()),
        ("select", SelectFromModel(LogisticRegression(C=0.3, l1_ratio=1.0, solver="liblinear"),
                                   max_features=k, threshold=-np.inf)),
    ]
    if clip is not None:
        steps.append(("clip", FunctionTransformer(_clip, kw_args={"c": clip})))
    return Pipeline(steps)


def selected_feature_names(pre: Pipeline) -> list[str]:
    mask = pre.named_steps["select"].get_support()
    return [n for n, m in zip(ENGINEERED, mask) if m]


def onehot_preprocessor() -> Pipeline:
    num = ["age", "balance", "day", "campaign", "pdays", "previous"]
    return Pipeline([("ct", ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)]))])


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].copy()
