"""Loading the participant workbook and resolving customer IDs."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config

FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]


@dataclass
class CampaignData:
    train: pd.DataFrame            # customer_id, features, y (0/1)
    test: pd.DataFrame             # customer_id, features
    stage2: pd.DataFrame           # customer_id, contact_cost, expected_deposit_value, capacity_units, customer_segment
    constraints: dict              # N, K, C (budget), H (capacity)
    id_mapping_note: str = ""
    checks: dict = field(default_factory=dict)


def _parse_constraints(df: pd.DataFrame) -> dict:
    sym = dict(zip(df["symbol"].astype(str).str.strip(), df["value"]))
    return {"N": int(sym["N"]), "K": int(sym["K"]), "budget": float(sym["C"]), "capacity": float(sym["H"])}


def assign_test_ids(test: pd.DataFrame, train: pd.DataFrame, stage2: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Attach ``customer_id`` to Stage1_Test rows.

    The updated organiser release includes ``customer_id`` in Stage1_Test (the first release did
    not). Resolution order:
      1. If the sheet has ``customer_id`` -> validate and use it (current release).
      2. If ``data/test_customer_ids_override.csv`` exists (one column ``customer_id``,
         one row per Stage1_Test row, same order) -> use it.
      3. Otherwise ASSUME the test rows are the 100 IDs absent from Stage1_Train,
         in ascending order. This assumption is flagged in every output.
    No label information is used in any branch.
    """
    test = test.copy()
    remaining = sorted(set(stage2["customer_id"]) - set(train["customer_id"]))
    if "customer_id" in test.columns:
        test["customer_id"] = test["customer_id"].astype(str).str.strip()
        if not test["customer_id"].is_unique or set(test["customer_id"]) != set(remaining):
            raise ValueError("Stage1_Test customer_id must be the 100 IDs absent from Stage1_Train, each once")
        return test, "customer_id provided in Stage1_Test (updated organiser release); validated against Stage2_Data"
    if len(remaining) != len(test):
        raise ValueError(f"{len(remaining)} unassigned IDs but {len(test)} test rows")
    if config.TEST_ID_OVERRIDE_CSV.exists():
        ids = pd.read_csv(config.TEST_ID_OVERRIDE_CSV)["customer_id"].astype(str).tolist()
        if len(ids) != len(test) or set(ids) != set(remaining):
            raise ValueError("override IDs must be a permutation of the IDs missing from Stage1_Train")
        test.insert(0, "customer_id", ids)
        return test, f"customer_id taken from {config.TEST_ID_OVERRIDE_CSV.name}"
    test.insert(0, "customer_id", remaining)
    note = ("ASSUMPTION: Stage1_Test has no customer_id column; rows were mapped to the 100 IDs "
            "absent from Stage1_Train in ascending order (row 1 -> " + remaining[0] + ", row 100 -> "
            + remaining[-1] + "). Replace via data/test_customer_ids_override.csv if the organisers "
            "provide the true mapping; the whole pipeline re-runs unchanged.")
    return test, note


def load(path=config.DATA_XLSX) -> CampaignData:
    xl = pd.ExcelFile(path)
    train = pd.read_excel(xl, "Stage1_Train")
    test = pd.read_excel(xl, "Stage1_Test")
    stage2 = pd.read_excel(xl, "Stage2_Data")
    cons = _parse_constraints(pd.read_excel(xl, "Stage2_Constraints"))
    train["customer_id"] = train["customer_id"].astype(str)
    stage2["customer_id"] = stage2["customer_id"].astype(str)
    train["y"] = (train["y"].astype(str).str.strip().str.lower() == "yes").astype(int)
    test, note = assign_test_ids(test, train, stage2)

    checks = {
        "n_train": len(train), "n_test": len(test), "n_stage2": len(stage2),
        "train_positives": int(train["y"].sum()),
        "train_ids_unique": bool(train["customer_id"].is_unique),
        "test_ids_unique": bool(test["customer_id"].is_unique),
        "train_test_overlap": len(set(train.customer_id) & set(test.customer_id)),
        "all_test_ids_in_stage2": bool(set(test.customer_id) <= set(stage2.customer_id)),
        "missing_values_train": int(train[FEATURES].isna().sum().sum()),
        "missing_values_test": int(test[FEATURES].isna().sum().sum()),
    }
    assert checks["train_ids_unique"] and checks["test_ids_unique"] and checks["train_test_overlap"] == 0
    assert checks["all_test_ids_in_stage2"]
    return CampaignData(train, test, stage2, cons, note, checks)
