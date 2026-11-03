# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
from __future__ import annotations
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import sys
import logging
from collections import deque

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # EWS Model

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Parameters

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def coerce_model_numeric(series: pd.Series) -> pd.Series:
    """
    Convert model inputs to numeric values robustly.

    This handles:
      - regular numeric columns,
      - numeric values stored as strings,
      - common boolean / Y-N encodings returned by Dataiku.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        txt = series.astype("string").str.strip().str.upper()

        bool_map = {
            "Y": 1.0,
            "YES": 1.0,
            "TRUE": 1.0,
            "T": 1.0,
            "N": 0.0,
            "NO": 0.0,
            "FALSE": 0.0,
            "F": 0.0,
        }

        bool_values = txt.map(bool_map)
        numeric_values = pd.to_numeric(txt, errors="coerce")

        return bool_values.fillna(numeric_values).astype(float)

    return pd.to_numeric(series, errors="coerce")

TARGET_COL:     str = "TGT_ANY"
GCA_COL:        str = "nominal_amount"
COLLATERAL_COL: str = "adjusted_collateral_value"

# --- Business-facing rating labels ---
# Numerical master-scale codes remain unchanged for calculations.
# These labels are added only to make thresholds easier to review.
RATING_VALUE_LABELS: dict[int, str] = {
    1: "AAA+",
    2: "AAA",
    3: "AA+",
    4: "AA",
    5: "AA-",
    6: "A+",
    7: "A",
    8: "A-",
    9: "BBB+",
    10: "BBB",
    11: "BBB-",
    12: "BB+",
    13: "BB",
    14: "BB-",
    15: "B+",
    16: "B",
    17: "B-",
    18: "CCC",
    19: "D",
}

RATING_LEVEL_PREDICTORS = {
    "RATING_GRANULAR_SCALE",
    "GCC_RATING_GRANULAR_SCALE",
}

def rating_value_label(value) -> str | float:
    """Translate a numerical master-scale code to its business label."""
    if pd.isna(value):
        return np.nan
    try:
        return RATING_VALUE_LABELS.get(int(round(float(value))), np.nan)
    except (TypeError, ValueError):
        return np.nan

# --- Event merging ---
MAX_GAP_MONTHS: int = 3

# --- Indicators to evaluate ---
# NOTE: this list combines the indicators from the last known-working
# model run with the new columns produced by the updated training
# recipe (master-scale notch mapping, GCCPD, cash advances, neobank
# transfers). Anything referencing a column the training recipe doesn't
# actually produce will be skipped safely at runtime (see the
# "unavailable_columns" guard further down) rather than crashing, but
# everything below has been checked against the training recipe you
# shared, so nothing here is expected to be skipped.
INDICATORS: list = [
    # --- PD and rating scales ---
    "RATING_GRANULAR_SCALE",
    "GCC_RATING_GRANULAR_SCALE",
    "failed_debit_amount",
    "failed_debit_count_3m",
    "rating_granular_scale_downgrade_1m",
    "rating_granular_scale_downgrade_3m",
    "rating_granular_scale_downgrade_6m",
    "rating_granular_scale_downgrade_12m",
    "rating_granular_scale_downgrade_best_12m",
    "rating_granular_scale_downgrade_worst_12m",
    "gcc_rating_granular_scale_downgrade_1m",
    "gcc_rating_granular_scale_downgrade_3m",
    "gcc_rating_granular_scale_downgrade_6m",
    "gcc_rating_granular_scale_downgrade_12m",
    "gcc_rating_granular_scale_downgrade_best_12m",
    "gcc_rating_granular_scale_downgrade_worst_12m",
    "dpd",
    "PD_CRR",
    "pd_change_1m",
    "pd_change_3m",
    "pd_change_6m",
    "pd_change_12m",
    "pd_change_worst_12m",
    "pd_change_best_12m",
    "pd_abs_change_1m",
    "pd_abs_change_worst_12m",
    "pd_abs_change_best_12m",
    "ltv",

    # --- General limit utilisation: absolute and relative changes ---
    "limit_utilization_abs_change_1m",
    "limit_utilization_abs_change_3m",
    "limit_utilization_abs_change_6m",
    "limit_utilization_abs_change_12m",
    "limit_utilization_abs_change_3m_max",
    "limit_utilization_abs_change_6m_max",
    "limit_utilization_abs_change_12m_max",
    "limit_utilization_change_1m",
    "limit_utilization_change_3m",
    "limit_utilization_change_6m",
    "limit_utilization_change_12m",
    "limit_utilization_change_3m_max",
    "limit_utilization_change_6m_max",
    "limit_utilization_change_12m_max",

    # --- Credit-card utilisation (existing logic retained) ---
    "cc_limit_utilization",
    "cc_limit_utilization_abs_change_1m",
    "cc_limit_utilization_abs_change_3m",
    "cc_limit_utilization_abs_change_6m",
    "cc_limit_utilization_abs_change_12m",
    "cc_limit_utilization_abs_change_3m_max",
    "cc_limit_utilization_abs_change_6m_max",
    "cc_limit_utilization_abs_change_12m_max",
    "cc_limit_utilization_change_1m",
    "cc_limit_utilization_change_3m",
    "cc_limit_utilization_change_6m",
    "cc_limit_utilization_change_12m",
    "cc_limit_utilization_change_3m_max",
    "cc_limit_utilization_change_6m_max",
    "cc_limit_utilization_change_12m_max",

    # --- GCC PD ---
    "PD_CRR_GCC",
    "gccpd_abs_change_1m",
    "gccpd_abs_change_best_12m",
    "gccpd_pd_difference",

    # --- Cash advances / neobanks ---
    "cash_advance_amount",
    "cash_advance_count",
    "cash_advance_count_3m",
    "cash_advance_balance_ratio",
    "neobank_topup_amount",
    "neobank_topup_count",
    "neobank_topup_balance_ratio",
    "cash_neobank_amount",
    "cash_neobank_count",
    "cash_neobank_balance_ratio",

    # --- Cash deposits ---
    "cash_deposit_amount",
    "cash_deposit_count",

    # --- Transaction-level credit-limit utilisation, aggregated monthly ---
    "cash_advance_transaction_limit_utilization",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "neobank_transaction_limit_utilization",
    "neobank_limit_utilization_before_transaction",
    "neobank_limit_utilization_after_transaction",

    # --- Arrears persistence ---
    "arrears_days_above_250",
    "arrears_days_above_500",

    # --- Flags ---
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
    "forbearance",
    "unemployed",

    # --- Granular-rating combinations ---
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_1m"),
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_3m"),
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_6m"),
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_12m"),
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_best_12m"),
    ("RATING_GRANULAR_SCALE", "rating_granular_scale_downgrade_worst_12m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_1m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_3m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_6m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_12m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_best_12m"),
    ("GCC_RATING_GRANULAR_SCALE", "gcc_rating_granular_scale_downgrade_worst_12m"),
    ("RATING_GRANULAR_SCALE", "pd_abs_change_best_12m"),

    # --- New GCC / client combinations from review comments ---
    # GCC PD + client PD
    ("PD_CRR_GCC", "PD_CRR"),

    # GCC PD deterioration + client granular rating
    ("gccpd_abs_change_1m", "RATING_GRANULAR_SCALE"),
    ("gccpd_abs_change_best_12m", "RATING_GRANULAR_SCALE"),

    # GCC granular-rating deterioration + client granular rating
    ("gcc_rating_granular_scale_downgrade_1m", "RATING_GRANULAR_SCALE"),
    ("gcc_rating_granular_scale_downgrade_3m", "RATING_GRANULAR_SCALE"),
    ("gcc_rating_granular_scale_downgrade_6m", "RATING_GRANULAR_SCALE"),
    ("gcc_rating_granular_scale_downgrade_12m", "RATING_GRANULAR_SCALE"),
    ("gcc_rating_granular_scale_downgrade_best_12m", "RATING_GRANULAR_SCALE"),
    ("gcc_rating_granular_scale_downgrade_worst_12m", "RATING_GRANULAR_SCALE"),

    # GCC PD deterioration + client PD deterioration
    ("gccpd_abs_change_1m", "pd_abs_change_1m"),
    ("gccpd_abs_change_best_12m", "pd_abs_change_best_12m"),

    # GCC PD deterioration + client granular-rating deterioration
    ("gccpd_abs_change_1m", "rating_granular_scale_downgrade_1m"),
    ("gccpd_abs_change_best_12m", "rating_granular_scale_downgrade_12m"),
    ("gccpd_abs_change_best_12m", "rating_granular_scale_downgrade_best_12m"),
    ("gccpd_abs_change_best_12m", "rating_granular_scale_downgrade_worst_12m"),

    # GCC granular-rating deterioration + client PD deterioration
    ("gcc_rating_granular_scale_downgrade_1m", "pd_abs_change_1m"),
    ("gcc_rating_granular_scale_downgrade_12m", "pd_abs_change_best_12m"),
    ("gcc_rating_granular_scale_downgrade_best_12m", "pd_abs_change_best_12m"),
    ("gcc_rating_granular_scale_downgrade_worst_12m", "pd_abs_change_worst_12m"),

    # GCC granular-rating deterioration + client granular-rating deterioration
    ("gcc_rating_granular_scale_downgrade_1m", "rating_granular_scale_downgrade_1m"),
    ("gcc_rating_granular_scale_downgrade_3m", "rating_granular_scale_downgrade_3m"),
    ("gcc_rating_granular_scale_downgrade_6m", "rating_granular_scale_downgrade_6m"),
    ("gcc_rating_granular_scale_downgrade_12m", "rating_granular_scale_downgrade_12m"),
    ("gcc_rating_granular_scale_downgrade_best_12m", "rating_granular_scale_downgrade_best_12m"),
    ("gcc_rating_granular_scale_downgrade_worst_12m", "rating_granular_scale_downgrade_worst_12m"),

    # --- Existing PD combinations ---
    ("PD_CRR", "pd_change_1m"),
    ("PD_CRR", "pd_change_3m"),
    ("PD_CRR", "pd_change_6m"),
    ("PD_CRR", "pd_change_12m"),
    ("PD_CRR", "pd_change_worst_12m"),
    ("PD_CRR", "pd_change_best_12m"),
    ("PD_CRR", "pd_abs_change_1m"),
    ("PD_CRR", "pd_abs_change_worst_12m"),
    ("PD_CRR", "pd_abs_change_best_12m"),

    # --- Existing LTV combinations ---
    ("ltv", "ltv_change_1m"),
    ("ltv", "ltv_change_12m"),
    ("ltv", "ltv_change_best_12m"),
    ("ltv", "ltv_abs_change_1m"),
    ("ltv", "ltv_abs_change_12m"),
    ("ltv", "ltv_abs_change_best_12m"),
    ("ltv", "PD_CRR"),
    ("ltv", "ASSETS"),
    ("ltv", "balance_to_loan"),
    ("ltv", "pd_change_1m"),
    ("ltv", "pd_change_best_12m"),
    ("ltv", "pd_abs_change_1m"),
    ("ltv", "pd_abs_change_best_12m"),

    # --- LTV + granular rating combinations ---
    ("ltv", "RATING_GRANULAR_SCALE"),
    ("ltv", "rating_granular_scale_downgrade_1m"),
    ("ltv", "rating_granular_scale_downgrade_3m"),
    ("ltv", "rating_granular_scale_downgrade_6m"),
    ("ltv", "rating_granular_scale_downgrade_12m"),
    ("ltv", "rating_granular_scale_downgrade_best_12m"),
    ("ltv", "rating_granular_scale_downgrade_worst_12m"),

    # --- General limit utilisation combinations: absolute ---
    ("limit_utilization_abs_change_1m", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_3m", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_6m", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_12m", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_3m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_6m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_12m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_abs_change_1m", "cc_limit_utilization"),
    ("limit_utilization_abs_change_3m", "cc_limit_utilization"),
    ("limit_utilization_abs_change_6m", "cc_limit_utilization"),
    ("limit_utilization_abs_change_12m", "cc_limit_utilization"),
    ("limit_utilization_abs_change_3m_max", "cc_limit_utilization"),
    ("limit_utilization_abs_change_6m_max", "cc_limit_utilization"),
    ("limit_utilization_abs_change_12m_max", "cc_limit_utilization"),

    # --- General limit utilisation combinations: relative ---
    ("limit_utilization_change_1m", "non_mortgage_loan_amt"),
    ("limit_utilization_change_3m", "non_mortgage_loan_amt"),
    ("limit_utilization_change_6m", "non_mortgage_loan_amt"),
    ("limit_utilization_change_12m", "non_mortgage_loan_amt"),
    ("limit_utilization_change_3m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_change_6m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_change_12m_max", "non_mortgage_loan_amt"),
    ("limit_utilization_change_1m", "cc_limit_utilization"),
    ("limit_utilization_change_3m", "cc_limit_utilization"),
    ("limit_utilization_change_6m", "cc_limit_utilization"),
    ("limit_utilization_change_12m", "cc_limit_utilization"),
    ("limit_utilization_change_3m_max", "cc_limit_utilization"),
    ("limit_utilization_change_6m_max", "cc_limit_utilization"),
    ("limit_utilization_change_12m_max", "cc_limit_utilization"),

    # --- Existing credit-card utilisation combinations ---
    ("cc_limit_utilization_abs_change_1m", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_3m", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_6m", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_12m", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_3m_max", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_6m_max", "cc_limit_utilization"),
    ("cc_limit_utilization_abs_change_12m_max", "cc_limit_utilization"),

    # --- Credit-card utilisation combinations: relative ---
    ("cc_limit_utilization_change_1m", "cc_limit_utilization"),
    ("cc_limit_utilization_change_3m", "cc_limit_utilization"),
    ("cc_limit_utilization_change_6m", "cc_limit_utilization"),
    ("cc_limit_utilization_change_12m", "cc_limit_utilization"),
    ("cc_limit_utilization_change_3m_max", "cc_limit_utilization"),
    ("cc_limit_utilization_change_6m_max", "cc_limit_utilization"),
    ("cc_limit_utilization_change_12m_max", "cc_limit_utilization"),

    ("cc_limit_utilization_abs_change_1m", "ASSETS"),
    ("cc_limit_utilization", "ASSETS"),
]

PREDICTOR_DIRECTIONS: dict[str, str] = {
    "RATING_GRANULAR_SCALE": "above",
    "GCC_RATING_GRANULAR_SCALE": "above",
    "failed_debit_amount": "above",
    "failed_debit_count_3m": "above",
    "dpd": "above",
    "ltv": "above",

    "rating_granular_scale_downgrade_1m": "above",
    "rating_granular_scale_downgrade_3m": "above",
    "rating_granular_scale_downgrade_6m": "above",
    "rating_granular_scale_downgrade_12m": "above",
    "rating_granular_scale_downgrade_best_12m": "above",
    "rating_granular_scale_downgrade_worst_12m": "above",
    "gcc_rating_granular_scale_downgrade_1m": "above",
    "gcc_rating_granular_scale_downgrade_3m": "above",
    "gcc_rating_granular_scale_downgrade_6m": "above",
    "gcc_rating_granular_scale_downgrade_12m": "above",
    "gcc_rating_granular_scale_downgrade_best_12m": "above",
    "gcc_rating_granular_scale_downgrade_worst_12m": "above",

    "PD_CRR": "above",
    "pd_change_1m": "above",
    "pd_change_3m": "above",
    "pd_change_6m": "above",
    "pd_change_12m": "above",
    "pd_change_worst_12m": "above",
    "pd_change_best_12m": "above",
    "pd_abs_change_1m": "above",
    "pd_abs_change_worst_12m": "above",
    "pd_abs_change_best_12m": "above",

    "PD_CRR_GCC": "above",
    "gccpd_abs_change_1m": "above",
    "gccpd_abs_change_best_12m": "above",
    "gccpd_pd_difference": "above",

    "ltv_change_1m": "above",
    "ltv_change_12m": "above",
    "ltv_change_best_12m": "above",
    "ltv_abs_change_1m": "above",
    "ltv_abs_change_12m": "above",
    "ltv_abs_change_best_12m": "above",
    "ASSETS": "below",
    "balance_to_loan": "below",

    "limit_utilization": "above",
    "limit_utilization_abs_change_1m": "above",
    "limit_utilization_abs_change_3m": "above",
    "limit_utilization_abs_change_6m": "above",
    "limit_utilization_abs_change_12m": "above",
    "limit_utilization_abs_change_3m_max": "above",
    "limit_utilization_abs_change_6m_max": "above",
    "limit_utilization_abs_change_12m_max": "above",
    "limit_utilization_change_1m": "above",
    "limit_utilization_change_3m": "above",
    "limit_utilization_change_6m": "above",
    "limit_utilization_change_12m": "above",
    "limit_utilization_change_3m_max": "above",
    "limit_utilization_change_6m_max": "above",
    "limit_utilization_change_12m_max": "above",

    "cc_limit_utilization": "above",
    "cc_limit_utilization_abs_change_1m": "above",
    "cc_limit_utilization_abs_change_3m": "above",
    "cc_limit_utilization_abs_change_6m": "above",
    "cc_limit_utilization_abs_change_12m": "above",
    "cc_limit_utilization_abs_change_3m_max": "above",
    "cc_limit_utilization_abs_change_6m_max": "above",
    "cc_limit_utilization_abs_change_12m_max": "above",
    "cc_limit_utilization_change_1m": "above",
    "cc_limit_utilization_change_3m": "above",
    "cc_limit_utilization_change_6m": "above",
    "cc_limit_utilization_change_12m": "above",
    "cc_limit_utilization_change_3m_max": "above",
    "cc_limit_utilization_change_6m_max": "above",
    "cc_limit_utilization_change_12m_max": "above",

    "non_mortgage_loan_amt": "above",
    "cash_advance_amount": "above",
    "cash_advance_count": "above",
    "cash_advance_count_3m": "above",
    "cash_advance_balance_ratio": "above",
    "neobank_topup_amount": "above",
    "neobank_topup_count": "above",
    "neobank_topup_balance_ratio": "above",
    "cash_neobank_amount": "above",
    "cash_neobank_count": "above",
    "cash_neobank_balance_ratio": "above",
    "cash_deposit_amount": "above",
    "cash_deposit_count": "above",
    "cash_advance_transaction_limit_utilization": "above",
    "cash_advance_limit_utilization_before_transaction": "above",
    "cash_advance_limit_utilization_after_transaction": "above",
    "neobank_transaction_limit_utilization": "above",
    "neobank_limit_utilization_before_transaction": "above",
    "neobank_limit_utilization_after_transaction": "above",
    "arrears_days_above_250": "above",
    "arrears_days_above_500": "above",

    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE": "above",
    "forbearance": "above",
    "unemployed": "above",
}

# --- Signal logic per predictor ---
# "crossing": fires the month the value moves from below to above its own
#             threshold (needs a valid prior-month value).
# "active":   fires whenever the value is beyond its threshold, regardless of
#             the prior-month value. Used for predictors that already express
#             a change (a "second crossing" on top of them would compare a
#             change to its own prior change, which is not meaningful).
# Any predictor not listed here defaults to "crossing".
SIGNAL_LOGIC: dict[str, str] = {
    "rating_granular_scale_downgrade_1m": "active",
    "rating_granular_scale_downgrade_3m": "active",
    "rating_granular_scale_downgrade_6m": "active",
    "rating_granular_scale_downgrade_12m": "active",
    "rating_granular_scale_downgrade_best_12m": "active",
    "rating_granular_scale_downgrade_worst_12m": "active",
    "gcc_rating_granular_scale_downgrade_1m": "active",
    "gcc_rating_granular_scale_downgrade_3m": "active",
    "gcc_rating_granular_scale_downgrade_6m": "active",
    "gcc_rating_granular_scale_downgrade_12m": "active",
    "gcc_rating_granular_scale_downgrade_best_12m": "active",
    "gcc_rating_granular_scale_downgrade_worst_12m": "active",

    "pd_change_1m": "active",
    "pd_change_3m": "active",
    "pd_change_6m": "active",
    "pd_change_12m": "active",
    "pd_change_worst_12m": "active",
    "pd_change_best_12m": "active",
    "pd_abs_change_1m": "active",
    "pd_abs_change_worst_12m": "active",
    "pd_abs_change_best_12m": "active",

    "gccpd_abs_change_1m": "active",
    "gccpd_abs_change_best_12m": "active",
    "gccpd_pd_difference": "active",

    "ltv_change_1m": "active",
    "ltv_change_12m": "active",
    "ltv_change_best_12m": "active",
    "ltv_abs_change_1m": "active",
    "ltv_abs_change_12m": "active",
    "ltv_abs_change_best_12m": "active",

    "limit_utilization_abs_change_1m": "active",
    "limit_utilization_abs_change_3m": "active",
    "limit_utilization_abs_change_6m": "active",
    "limit_utilization_abs_change_12m": "active",
    "limit_utilization_abs_change_3m_max": "active",
    "limit_utilization_abs_change_6m_max": "active",
    "limit_utilization_abs_change_12m_max": "active",
    "limit_utilization_change_1m": "active",
    "limit_utilization_change_3m": "active",
    "limit_utilization_change_6m": "active",
    "limit_utilization_change_12m": "active",
    "limit_utilization_change_3m_max": "active",
    "limit_utilization_change_6m_max": "active",
    "limit_utilization_change_12m_max": "active",

    "cc_limit_utilization_abs_change_1m": "active",
    "cc_limit_utilization_abs_change_3m": "active",
    "cc_limit_utilization_abs_change_6m": "active",
    "cc_limit_utilization_abs_change_12m": "active",
    "cc_limit_utilization_abs_change_3m_max": "active",
    "cc_limit_utilization_abs_change_6m_max": "active",
    "cc_limit_utilization_abs_change_12m_max": "active",
    "cc_limit_utilization_change_1m": "active",
    "cc_limit_utilization_change_3m": "active",
    "cc_limit_utilization_change_6m": "active",
    "cc_limit_utilization_change_12m": "active",
    "cc_limit_utilization_change_3m_max": "active",
    "cc_limit_utilization_change_6m_max": "active",
    "cc_limit_utilization_change_12m_max": "active",

    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE": "active",
    "forbearance": "active",
    "unemployed": "active",
}

# --- Threshold grids ---
THRESHOLD_GRIDS = {
    "ltv": np.linspace(0.70, 1.29, 60),
    "PD_CRR": np.linspace(0.01, 0.50, 50),
    "PD_CRR_GCC": np.linspace(0.01, 0.50, 50),
    "limit_utilization": np.linspace(0.80, 1.29, 50),
    "cc_limit_utilization": np.linspace(0.80, 1.29, 50),

    # Granular rating scale: 1 = AAA+, 19 = D.
    "RATING_GRANULAR_SCALE": np.arange(1, 20, dtype=float),
    "GCC_RATING_GRANULAR_SCALE": np.arange(1, 20, dtype=float),

    # Number of notches of deterioration.
    "rating_granular_scale_downgrade_1m": np.arange(0, 19, dtype=float),
    "rating_granular_scale_downgrade_3m": np.arange(0, 19, dtype=float),
    "rating_granular_scale_downgrade_6m": np.arange(0, 19, dtype=float),
    "rating_granular_scale_downgrade_12m": np.arange(0, 19, dtype=float),
    "rating_granular_scale_downgrade_best_12m": np.arange(0, 19, dtype=float),
    "rating_granular_scale_downgrade_worst_12m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_1m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_3m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_6m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_12m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_best_12m": np.arange(0, 19, dtype=float),
    "gcc_rating_granular_scale_downgrade_worst_12m": np.arange(0, 19, dtype=float),

    "dpd": np.linspace(0, 18, 19),

    "pd_change_1m": np.linspace(0.02, 2.00, 100),
    "pd_change_3m": np.linspace(0.02, 2.00, 100),
    "pd_change_6m": np.linspace(0.02, 2.00, 100),
    "pd_change_12m": np.linspace(0.02, 2.00, 100),
    "pd_change_worst_12m": np.linspace(0.02, 2.00, 100),
    "pd_change_best_12m": np.linspace(0.02, 2.00, 100),

    "pd_abs_change_1m": np.linspace(0.01, 0.2, 20),
    "pd_abs_change_worst_12m": np.linspace(0.01, 0.2, 20),
    "pd_abs_change_best_12m": np.linspace(0.01, 0.2, 20),

    "gccpd_abs_change_1m": np.linspace(0.01, 0.2, 20),
    "gccpd_abs_change_best_12m": np.linspace(0.01, 0.2, 20),
    "gccpd_pd_difference": np.linspace(0.01, 0.2, 20),

    "ltv_change_1m": np.linspace(0.01, 0.2, 20),
    "ltv_change_12m": np.linspace(0.01, 0.2, 20),
    "ltv_change_best_12m": np.linspace(0.01, 0.2, 20),
    "ltv_abs_change_1m": np.linspace(0.01, 0.2, 20),
    "ltv_abs_change_12m": np.linspace(0.01, 0.2, 20),
    "ltv_abs_change_best_12m": np.linspace(0.01, 0.2, 20),

    "ASSETS": np.linspace(1000, 10000, 10),
    "unemployed": np.array([0.5]),

    # General limit utilisation - absolute change ("abs") and relative change.
    "limit_utilization_abs_change_1m": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_3m": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_6m": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_12m": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_3m_max": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_6m_max": np.linspace(0.05, 1.50, 30),
    "limit_utilization_abs_change_12m_max": np.linspace(0.05, 1.50, 30),

    # Relative changes are not capped at 100%; 1.0 corresponds to +100%.
    "limit_utilization_change_1m": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_3m": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_6m": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_12m": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_3m_max": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_6m_max": np.linspace(0.10, 5.00, 50),
    "limit_utilization_change_12m_max": np.linspace(0.10, 5.00, 50),

    # Credit-card utilisation: explicit absolute and relative changes.
    "cc_limit_utilization_abs_change_1m": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_3m": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_6m": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_12m": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_3m_max": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_6m_max": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_abs_change_12m_max": np.linspace(0.1, 1, 10),
    "cc_limit_utilization_change_1m": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_3m": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_6m": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_12m": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_3m_max": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_6m_max": np.linspace(0.10, 5.00, 50),
    "cc_limit_utilization_change_12m_max": np.linspace(0.10, 5.00, 50),

    "non_mortgage_loan_amt": np.linspace(500, 20000, 40),

    # Failed direct debit count accumulated over the last 3 months.
    "failed_debit_count_3m": np.arange(0, 31, dtype=float),

    # Cash advance / neobank amount thresholds requested as clean steps.
    "cash_advance_amount": np.r_[100.0, np.arange(250.0, 5000.1, 250.0)],
    "cash_advance_count": np.linspace(1, 10, 10),
    "cash_advance_count_3m": np.arange(0, 31, dtype=float),
    "neobank_topup_amount": np.r_[100.0, np.arange(250.0, 5000.1, 250.0)],
    "neobank_topup_count": np.linspace(1, 10, 10),
    "cash_neobank_amount": np.r_[100.0, np.arange(500.0, 8000.1, 500.0)],
    "cash_neobank_count": np.linspace(1, 15, 15),

    # Cash / Neobank activity relative to positive balances.
    "cash_advance_balance_ratio": np.arange(0.5, 5.01, 0.5),
    "neobank_topup_balance_ratio": np.arange(0.5, 5.01, 0.5),
    "cash_neobank_balance_ratio": np.arange(0.5, 5.01, 0.5),

    # Transaction-level credit-limit utilisation: same interpretable range
    # as the regular credit-card utilisation indicator.
    "cash_advance_transaction_limit_utilization": np.linspace(0.80, 1.29, 50),
    "cash_advance_limit_utilization_before_transaction": np.linspace(0.80, 1.29, 50),
    "cash_advance_limit_utilization_after_transaction": np.linspace(0.80, 1.29, 50),
    "neobank_transaction_limit_utilization": np.linspace(0.80, 1.29, 50),
    "neobank_limit_utilization_before_transaction": np.linspace(0.80, 1.29, 50),
    "neobank_limit_utilization_after_transaction": np.linspace(0.80, 1.29, 50),

    # Number of days in a month for which arrears remain above the amount threshold.
    "arrears_days_above_250": np.arange(0, 31, dtype=float),
    "arrears_days_above_500": np.arange(0, 31, dtype=float),

    # Binary flags.
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE": np.array([0.5]),
    "forbearance": np.array([0.5]),
}


PERCENTILE_GRIDS: dict[str, np.ndarray] = {
    "failed_debit_amount": np.arange(1, 100),
    "balance_to_loan": np.arange(1, 100),
}

# Cash-deposit activity is assessed against a compact set of thresholds
# derived from the strictly positive activity distribution.
POSITIVE_PERCENTILE_GRIDS: dict[str, np.ndarray] = {
    "cash_deposit_amount": np.array([10, 25, 50, 75, 90, 95, 99], dtype=float),
    "cash_deposit_count": np.array([10, 25, 50, 75, 90, 95, 99], dtype=float),
}


def build_positive_percentile_grid(
    series: pd.Series,
    percentiles: np.ndarray,
) -> np.ndarray:
    """Build unique thresholds from the strictly positive distribution."""
    values = coerce_model_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()
    positive = values[values > 0]

    if positive.empty:
        return np.array([0.0], dtype=float)

    thresholds = np.percentile(
        positive.to_numpy(dtype=float),
        np.asarray(percentiles, dtype=float),
    )
    return np.unique(np.r_[0.0, thresholds.astype(float)])


# --- Signal merging ---
MAX_SIGNAL_GAP_MONTHS: int = 2

# --- Max lag allowed between predictor A and predictor B for pairs (months) ---
DEFAULT_PREDICTOR_GAP_MONTHS: int = 120
MAX_GAP_BETWEEN_PREDICTORS_MONTHS: dict = {
}

# --- Signal-event pairing ---
MIN_LEAD_MONTHS: int = 1
MAX_LEAD_MONTHS: int = 11

# --- Exposure threshold ---
MIN_GCA: float = 50000.0

# --- TP cost rate as a function of lead time ---
COST_TP_PCT_LATE:  float = 0.20
COST_TP_PCT_EARLY: float = 0.10

def cost_tp_pct_fn(lead_months: np.ndarray) -> np.ndarray:
    """Linear interpolation between COST_TP_PCT_LATE and COST_TP_PCT_EARLY."""
    return np.interp(lead_months,
                      [MIN_LEAD_MONTHS, MAX_LEAD_MONTHS],
                      [COST_TP_PCT_LATE, COST_TP_PCT_EARLY])

# --- Cost parameters ---
COST_TP_FIXED: float = 1000.0
COST_FP:       float = 200.0
COST_FN_PCT:   float = 0.20

# --- Second cost function (collateral-adjusted) ---
# The collateral input is already haircut-adjusted at loan level in the
# training dataset and aggregated to entity-month.

# False-negative cost:
# max(10% * Exposure,
#     ((1 - AdjustedCollateralValue / Exposure) + 1%) * Exposure + 1000)
COST2_FN_MIN_PCT:   float = 0.10
COST2_FN_EXTRA_PCT: float = 0.01
COST2_FN_FIXED:     float = 1000.0

# True-positive cost as a share of the corresponding FN cost.
COST2_TP_PCT_LATE:  float = 1.00
COST2_TP_PCT_EARLY: float = 0.50

def cost2_tp_pct_fn(lead_months: np.ndarray) -> np.ndarray:
    """50% of FN cost if early, 100% if late, linear in between."""
    return np.interp(
        lead_months,
        [MIN_LEAD_MONTHS, MAX_LEAD_MONTHS],
        [COST2_TP_PCT_LATE, COST2_TP_PCT_EARLY],
    )

def calculate_cost2_fn(
    exposure: pd.Series,
    adjusted_collateral_value: pd.Series,
) -> pd.Series:
    """Calculate the collateral-adjusted false-negative cost."""
    exposure = coerce_model_numeric(exposure)
    adjusted_collateral_value = coerce_model_numeric(
        adjusted_collateral_value
    )

    valid_exposure = exposure.notna() & (exposure > 0)
    valid_collateral = (
        adjusted_collateral_value.notna()
        & (adjusted_collateral_value >= 0)
    )
    valid = valid_exposure & valid_collateral

    minimum_cost = COST2_FN_MIN_PCT * exposure

    raw_cost = (
        (
            1.0
            - adjusted_collateral_value / exposure
            + COST2_FN_EXTRA_PCT
        )
        * exposure
        + COST2_FN_FIXED
    )

    fn_cost = pd.Series(np.nan, index=exposure.index, dtype=float)
    fn_cost.loc[valid] = np.maximum(
        minimum_cost.loc[valid],
        raw_cost.loc[valid],
    )

    fallback = valid_exposure & ~valid_collateral
    fn_cost.loc[fallback] = minimum_cost.loc[fallback]

    return fn_cost


#--- arrears amount threshold flags---
ARREARS_THRESHOLDS = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

for t in ARREARS_THRESHOLDS:
    col = f"arrears_amount_above_{t}"
    INDICATORS.append(col)
    PREDICTOR_DIRECTIONS[col] = "above"
    SIGNAL_LOGIC[col] = "active"
    THRESHOLD_GRIDS[col] = np.array([0.5])  # binary flag (0 o 1), optimal cutoff is 0.5

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Load data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(
    "RUN MODE: EWS model using the updated upstream training data."
)

EWS_training = dataiku.Dataset("EWS_training")
EWS_training_df = EWS_training.get_dataframe()

# Standardise join / time keys before any temporal transformation.
EWS_training_df["ENTITY_ID"] = EWS_training_df["ENTITY_ID"].astype(str)
EWS_training_df["MAP_DTE"] = pd.to_datetime(
    EWS_training_df["MAP_DTE"],
    errors="coerce",
)

EWS_training_df = (
    EWS_training_df
    .sort_values(["ENTITY_ID", "MAP_DTE"])
    .reset_index(drop=True)
)

training_duplicate_entity_month_rows = int(
    EWS_training_df.duplicated(
        ["ENTITY_ID", "MAP_DTE"],
        keep=False,
    ).sum()
)

print(
    "Duplicate training rows by ENTITY_ID + MAP_DTE: "
    f"{training_duplicate_entity_month_rows:,}"
)

# ---------------------------------------------------------------------
# Numeric type normalisation
# ---------------------------------------------------------------------
# Convert all model inputs that are expected to be
# numeric before threshold comparisons, differences, rolling windows,
# or percentile calculations are performed.
NUMERIC_INPUT_COLUMNS = {
    "RATING_SCALE",
    "PD_MASTER_NOTCH",
    "GCCPD_MASTER_NOTCH",
    "failed_debit_amount",
    "failed_debit_count",
    "failed_debit_amount_3m",
    "failed_debit_count_3m",
    "dpd",
    "arrears_amount",
    "PD_CRR",
    "PD_CRR_GCC",
    "ltv",
    "ASSETS",
    "gross_carrying_amount",
    "nominal_amount",
    "allocatedcollateralvalue",
    "adjusted_collateral_value",
    "limit_utilization",
    "cc_limit_utilization",
    "non_mortgage_loan_amt",
    "rating_current",
    "rating_origin",
    "pd_loan",
    "inception_pd_loan",
    "cash_advance_amount",
    "cash_advance_count",
    "cash_advance_balance_ratio",
    "neobank_topup_amount",
    "neobank_topup_count",
    "neobank_topup_balance_ratio",
    "cash_neobank_amount",
    "cash_neobank_count",
    "cash_neobank_balance_ratio",

    # Monthly behavioural and arrears inputs.
    "cash_deposit_amount",
    "cash_deposit_count",
    "cash_advance_transaction_limit_utilization",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "neobank_transaction_limit_utilization",
    "neobank_limit_utilization_before_transaction",
    "neobank_limit_utilization_after_transaction",
    "arrears_days_above_250",
    "arrears_days_above_500",
    "arrears_above_250",
    "arrears_above_500",

    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
    "forbearance",
    "defaulted",
    "dim1",
}

for col in NUMERIC_INPUT_COLUMNS:
    if col in EWS_training_df.columns:
        EWS_training_df[col] = coerce_model_numeric(
            EWS_training_df[col]
        )

# ---------------------------------------------------------------------
# Missing-value treatment used by the model
# ---------------------------------------------------------------------
if "ltv" in EWS_training_df.columns:
    EWS_training_df["ltv"] = (
        EWS_training_df
        .groupby("ENTITY_ID")["ltv"]
        .ffill()
    )

for col in [
    "dpd",
    "arrears_amount",
    "ASSETS",
    "failed_debit_amount",
    "failed_debit_count",
    "failed_debit_amount_3m",
    "failed_debit_count_3m",
    "cash_advance_amount",
    "cash_advance_count",
    "cash_advance_balance_ratio",
    "neobank_topup_amount",
    "neobank_topup_count",
    "neobank_topup_balance_ratio",
    "cash_neobank_amount",
    "cash_neobank_count",
    "cash_neobank_balance_ratio",
    "cash_deposit_amount",
    "cash_deposit_count",
    "arrears_days_above_250",
    "arrears_days_above_500",
    "arrears_above_250",
    "arrears_above_500",
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
]:
    if col in EWS_training_df.columns:
        EWS_training_df[col] = (
            coerce_model_numeric(EWS_training_df[col])
            .fillna(0.0)
        )



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Daily panel

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# EWS_training is already the daily ENTITY_ID + MAP_DTE panel.
# Do not rebuild entities x dates here: on daily data that would create a
# very large Cartesian product and is unnecessary.
df = (
    EWS_training_df
    .sort_values(["ENTITY_ID", "MAP_DTE"])
    .reset_index(drop=True)
    .copy()
)

if training_duplicate_entity_month_rows > 0:
    raise ValueError(
        "EWS_training contains duplicated ENTITY_ID + MAP_DTE rows. "
        "The daily model requires one observation per entity-day."
    )

for col in [
    "arrears_above_250",
    "arrears_above_500",
]:
    if col not in df.columns:
        df[col] = 0.0
    else:
        df[col] = coerce_model_numeric(df[col]).fillna(0.0)

THRESHOLD_CHUNK_SIZE: int = 10
DAYS_PER_MONTH_FOR_COST: float = 365.25 / 12.0


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Calendar helpers for daily observations

_BASE_INDEX = pd.MultiIndex.from_arrays(
    [df["ENTITY_ID"], df["MAP_DTE"]],
    names=["ENTITY_ID", "MAP_DTE"],
)
_LAG_CACHE: dict[tuple[str, int], pd.Series] = {}


def calendar_lag_value(
    frame: pd.DataFrame,
    value_col: str,
    lag_months: int,
) -> pd.Series:
    """Return the value on the corresponding calendar date N months earlier."""
    key = (value_col, lag_months)
    if key in _LAG_CACHE:
        return _LAG_CACHE[key]

    values = coerce_model_numeric(frame[value_col])
    lookup = pd.Series(
        values.to_numpy(),
        index=_BASE_INDEX,
        dtype=float,
    )

    target_dates = (
        frame["MAP_DTE"]
        - pd.DateOffset(months=lag_months)
    )
    target_index = pd.MultiIndex.from_arrays(
        [frame["ENTITY_ID"], target_dates],
        names=["ENTITY_ID", "MAP_DTE"],
    )

    result = pd.Series(
        lookup.reindex(target_index).to_numpy(),
        index=frame.index,
        dtype=float,
    )
    _LAG_CACHE[key] = result
    return result


def calendar_rolling_stat(
    frame: pd.DataFrame,
    value_col: str,
    months: int,
    stat: str,
    include_current: bool = False,
) -> pd.Series:
    """Calendar-month rolling statistic, optionally including the current day."""
    values_all = coerce_model_numeric(frame[value_col]).to_numpy(dtype=float)
    out = np.full(len(frame), np.nan, dtype=float)

    for positions in frame.groupby("ENTITY_ID", sort=False).indices.values():
        pos = np.asarray(positions, dtype=int)
        dates_idx = pd.DatetimeIndex(frame.loc[pos, "MAP_DTE"])
        dates = dates_idx.to_numpy(dtype="datetime64[ns]")
        values = values_all[pos]
        n = len(pos)

        starts = (
            dates_idx - pd.DateOffset(months=months)
        ).to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(dates, starts, side="left")

        if stat in {"mean", "sum"}:
            valid = ~np.isnan(values)
            csum = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
            ccount = np.r_[0, np.cumsum(valid.astype(int))]

            right = np.arange(n) + (1 if include_current else 0)
            sums = csum[right] - csum[left]
            counts = ccount[right] - ccount[left]

            if stat == "mean":
                local = np.where(counts > 0, sums / counts, np.nan)
            else:
                local = np.where(counts > 0, sums, np.nan)

            out[pos] = local
            continue

        if stat not in {"min", "max"}:
            raise ValueError(f"Unsupported rolling stat: {stat}")

        dq = deque()

        for i in range(n):
            while dq and dq[0] < left[i]:
                dq.popleft()

            if include_current and not np.isnan(values[i]):
                if stat == "min":
                    while dq and values[dq[-1]] >= values[i]:
                        dq.pop()
                else:
                    while dq and values[dq[-1]] <= values[i]:
                        dq.pop()
                dq.append(i)

            if dq:
                out[pos[i]] = values[dq[0]]

            if not include_current and not np.isnan(values[i]):
                if stat == "min":
                    while dq and values[dq[-1]] >= values[i]:
                        dq.pop()
                else:
                    while dq and values[dq[-1]] <= values[i]:
                        dq.pop()
                dq.append(i)

    return pd.Series(out, index=frame.index, dtype=float)


def rolling_day_count(
    frame: pd.DataFrame,
    flag_col: str,
    days: int = 30,
) -> pd.Series:
    """Count active daily flags over the trailing N calendar days including today."""
    tmp = frame[["ENTITY_ID", "MAP_DTE", flag_col]].copy()
    tmp[flag_col] = coerce_model_numeric(tmp[flag_col]).fillna(0.0)

    rolled = (
        tmp
        .groupby("ENTITY_ID", sort=False)
        .rolling(
            f"{days}D",
            on="MAP_DTE",
            closed="right",
            min_periods=1,
        )[flag_col]
        .sum()
        .reset_index(drop=True)
    )

    return pd.Series(
        rolled.to_numpy(),
        index=frame.index,
        dtype=float,
    )


# Cash advance count accumulated over the trailing three calendar months.
if "cash_advance_count" in df.columns:
    df["cash_advance_count"] = (
        coerce_model_numeric(df["cash_advance_count"])
        .fillna(0.0)
    )
    df["cash_advance_count_3m"] = calendar_rolling_stat(
        df,
        "cash_advance_count",
        months=3,
        stat="sum",
        include_current=True,
    ).fillna(0.0)

# Daily arrears persistence: number of active days in the trailing 30 days.
df["arrears_days_above_250"] = rolling_day_count(
    df,
    "arrears_above_250",
    days=30,
)
df["arrears_days_above_500"] = rolling_day_count(
    df,
    "arrears_above_500",
    days=30,
)

# ---------------------------------------------------------------------
# Cash / Neobank activity relative to positive balances
# ---------------------------------------------------------------------
if "ASSETS" in df.columns:
    _positive_balances = coerce_model_numeric(df["ASSETS"])

    for _amount_col, _ratio_col in [
        ("cash_advance_amount", "cash_advance_balance_ratio"),
        ("neobank_topup_amount", "neobank_topup_balance_ratio"),
        ("cash_neobank_amount", "cash_neobank_balance_ratio"),
    ]:
        if _amount_col in df.columns:
            _amount = coerce_model_numeric(df[_amount_col]).fillna(0.0)
            df[_ratio_col] = np.where(
                _positive_balances > 0,
                _amount / _positive_balances,
                np.nan,
            )

df.info()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Variable transformations

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# =====================================================================
# CREATION OF ARREARS THRESHOLD COLUMNS
# =====================================================================
df["arrears_amount"] = coerce_model_numeric(
    df["arrears_amount"]
).fillna(0.0)

for t in ARREARS_THRESHOLDS:
    df[f"arrears_amount_above_{t}"] = (
        df["arrears_amount"] > t
    ).astype(int)
# =====================================================================

# ---------------------------------------------------------------------
# Rating scales and notch downgrades
# ---------------------------------------------------------------------
df["RATING_GRANULAR_SCALE"] = coerce_model_numeric(
    df["PD_MASTER_NOTCH"]
)
df["GCC_RATING_GRANULAR_SCALE"] = coerce_model_numeric(
    df["GCCPD_MASTER_NOTCH"]
)
df["RATING_AGGREGATED_SCALE"] = coerce_model_numeric(
    df["RATING_SCALE"]
)

df["RATING_GRANULAR_LABEL"] = df[
    "RATING_GRANULAR_SCALE"
].map(rating_value_label)
df["GCC_RATING_GRANULAR_LABEL"] = df[
    "GCC_RATING_GRANULAR_SCALE"
].map(rating_value_label)

for lag in [1, 3, 6, 12]:
    previous_rating = calendar_lag_value(
        df,
        "RATING_GRANULAR_SCALE",
        lag,
    )
    previous_gcc_rating = calendar_lag_value(
        df,
        "GCC_RATING_GRANULAR_SCALE",
        lag,
    )

    df[f"rating_granular_scale_downgrade_{lag}m"] = (
        df["RATING_GRANULAR_SCALE"] - previous_rating
    )
    df[f"gcc_rating_granular_scale_downgrade_{lag}m"] = (
        df["GCC_RATING_GRANULAR_SCALE"] - previous_gcc_rating
    )

# Best / worst rating over the previous 12 calendar months.
df["min_rating_granular_scale_12m"] = calendar_rolling_stat(
    df,
    "RATING_GRANULAR_SCALE",
    months=12,
    stat="min",
)
df["max_rating_granular_scale_12m"] = calendar_rolling_stat(
    df,
    "RATING_GRANULAR_SCALE",
    months=12,
    stat="max",
)

df["rating_granular_scale_downgrade_best_12m"] = (
    df["RATING_GRANULAR_SCALE"]
    - df["min_rating_granular_scale_12m"]
)
df["rating_granular_scale_downgrade_worst_12m"] = (
    df["RATING_GRANULAR_SCALE"]
    - df["max_rating_granular_scale_12m"]
)

df["min_gcc_rating_granular_scale_12m"] = calendar_rolling_stat(
    df,
    "GCC_RATING_GRANULAR_SCALE",
    months=12,
    stat="min",
)
df["max_gcc_rating_granular_scale_12m"] = calendar_rolling_stat(
    df,
    "GCC_RATING_GRANULAR_SCALE",
    months=12,
    stat="max",
)

df["gcc_rating_granular_scale_downgrade_best_12m"] = (
    df["GCC_RATING_GRANULAR_SCALE"]
    - df["min_gcc_rating_granular_scale_12m"]
)
df["gcc_rating_granular_scale_downgrade_worst_12m"] = (
    df["GCC_RATING_GRANULAR_SCALE"]
    - df["max_gcc_rating_granular_scale_12m"]
)

# ---------------------------------------------------------------------
# PD changes using calendar lags
# ---------------------------------------------------------------------
for lag in [1, 3, 6, 12]:
    previous_pd = calendar_lag_value(
        df,
        "PD_CRR",
        lag,
    )
    df[f"pd_change_{lag}m"] = np.where(
        previous_pd.notna() & (previous_pd != 0),
        (df["PD_CRR"] - previous_pd) / previous_pd,
        np.nan,
    )

previous_pd_1m = calendar_lag_value(
    df,
    "PD_CRR",
    1,
)
df["pd_abs_change_1m"] = (
    df["PD_CRR"] - previous_pd_1m
)

df["max_PD_12m"] = calendar_rolling_stat(
    df,
    "PD_CRR",
    months=12,
    stat="max",
)
df["min_PD_12m"] = calendar_rolling_stat(
    df,
    "PD_CRR",
    months=12,
    stat="min",
)

df["pd_change_worst_12m"] = np.where(
    df["max_PD_12m"].notna()
    & (df["max_PD_12m"] != 0),
    (
        df["PD_CRR"] - df["max_PD_12m"]
    ) / df["max_PD_12m"],
    np.nan,
)
df["pd_change_best_12m"] = np.where(
    df["min_PD_12m"].notna()
    & (df["min_PD_12m"] != 0),
    (
        df["PD_CRR"] - df["min_PD_12m"]
    ) / df["min_PD_12m"],
    np.nan,
)

df["pd_abs_change_worst_12m"] = (
    df["PD_CRR"] - df["max_PD_12m"]
)
df["pd_abs_change_best_12m"] = (
    df["PD_CRR"] - df["min_PD_12m"]
)

# Origination-PD reference columns retained as in the previous model.
df["pd_change_origination"] = np.where(
    df["inception_pd_loan"].notna()
    & (df["inception_pd_loan"] != 0),
    (
        df["pd_loan"] - df["inception_pd_loan"]
    ) / df["inception_pd_loan"],
    np.nan,
)
df["pd_abs_change_origination"] = (
    df["pd_loan"] - df["inception_pd_loan"]
)

# ---------------------------------------------------------------------
# GCC PD
# ---------------------------------------------------------------------
df["min_GCCPD_12m"] = calendar_rolling_stat(
    df,
    "PD_CRR_GCC",
    months=12,
    stat="min",
)

previous_gccpd_1m = calendar_lag_value(
    df,
    "PD_CRR_GCC",
    1,
)
df["gccpd_abs_change_1m"] = (
    df["PD_CRR_GCC"] - previous_gccpd_1m
)
df["gccpd_abs_change_best_12m"] = (
    df["PD_CRR_GCC"] - df["min_GCCPD_12m"]
)
df["gccpd_pd_difference"] = (
    df["PD_CRR_GCC"] - df["PD_CRR"]
)

# ---------------------------------------------------------------------
# LTV changes
# ---------------------------------------------------------------------
previous_ltv_1m = calendar_lag_value(
    df,
    "ltv",
    1,
)
df["ltv_change_1m"] = np.where(
    previous_ltv_1m.notna() & (previous_ltv_1m != 0),
    (df["ltv"] - previous_ltv_1m) / previous_ltv_1m,
    np.nan,
)
df["ltv_abs_change_1m"] = (
    df["ltv"] - previous_ltv_1m
)

# Keep the previous 12 -> 11 -> ... -> 6 month fallback, but use
# true calendar dates rather than row positions.
previous_ltv_12m = calendar_lag_value(
    df,
    "ltv",
    12,
).copy()
for lag in [11, 10, 9, 8, 7, 6]:
    previous_ltv_12m = previous_ltv_12m.fillna(
        calendar_lag_value(df, "ltv", lag)
    )

df["ltv_change_12m"] = np.where(
    previous_ltv_12m.notna() & (previous_ltv_12m != 0),
    (df["ltv"] - previous_ltv_12m) / previous_ltv_12m,
    np.nan,
)
df["ltv_abs_change_12m"] = (
    df["ltv"] - previous_ltv_12m
)

df["min_ltv_12m"] = calendar_rolling_stat(
    df,
    "ltv",
    months=12,
    stat="min",
)
df["ltv_change_best_12m"] = np.where(
    df["min_ltv_12m"].notna()
    & (df["min_ltv_12m"] != 0),
    (
        df["ltv"] - df["min_ltv_12m"]
    ) / df["min_ltv_12m"],
    np.nan,
)
df["ltv_abs_change_best_12m"] = (
    df["ltv"] - df["min_ltv_12m"]
)

# ---------------------------------------------------------------------
# Other indicators
# ---------------------------------------------------------------------
df["balance_to_loan"] = np.where(
    coerce_model_numeric(df["gross_carrying_amount"]) != 0,
    (
        coerce_model_numeric(df["ASSETS"])
        / coerce_model_numeric(df["gross_carrying_amount"])
    ),
    np.nan,
)

employment_status = df["EMPLOYMENT_STATUS"].astype("string")
df["unemployed"] = (
    employment_status
    .str.startswith("7")
    .map({True: 1.0, False: 0.0})
)

# ---------------------------------------------------------------------
# General limit utilisation: calendar-window absolute and relative changes
# ---------------------------------------------------------------------
previous_limit_1m = calendar_lag_value(
    df,
    "limit_utilization",
    1,
)
df["limit_utilization_abs_change_1m"] = (
    df["limit_utilization"] - previous_limit_1m
)
df["limit_utilization_change_1m"] = np.where(
    previous_limit_1m.notna() & (previous_limit_1m != 0),
    (
        df["limit_utilization"] - previous_limit_1m
    ) / previous_limit_1m,
    np.nan,
)

for months in [3, 6, 12]:
    average_col = f"average_utilization_{months}m"
    max_col = f"max_utilization_{months}m"

    df[average_col] = calendar_rolling_stat(
        df,
        "limit_utilization",
        months=months,
        stat="mean",
    )
    df[max_col] = calendar_rolling_stat(
        df,
        "limit_utilization",
        months=months,
        stat="max",
    )

    df[f"limit_utilization_abs_change_{months}m"] = (
        df["limit_utilization"] - df[average_col]
    )
    df[f"limit_utilization_change_{months}m"] = np.where(
        df[average_col].notna() & (df[average_col] != 0),
        (
            df["limit_utilization"] - df[average_col]
        ) / df[average_col],
        np.nan,
    )

    df[f"limit_utilization_abs_change_{months}m_max"] = (
        df["limit_utilization"] - df[max_col]
    )
    df[f"limit_utilization_change_{months}m_max"] = np.where(
        df[max_col].notna() & (df[max_col] != 0),
        (
            df["limit_utilization"] - df[max_col]
        ) / df[max_col],
        np.nan,
    )

# ---------------------------------------------------------------------
# Credit-card utilisation: calendar-window absolute and relative changes
# ---------------------------------------------------------------------
previous_cc_1m = calendar_lag_value(
    df,
    "cc_limit_utilization",
    1,
)
df["cc_limit_utilization_abs_change_1m"] = (
    df["cc_limit_utilization"] - previous_cc_1m
)
df["cc_limit_utilization_change_1m"] = np.where(
    previous_cc_1m.notna() & (previous_cc_1m != 0),
    (
        df["cc_limit_utilization"] - previous_cc_1m
    ) / previous_cc_1m,
    np.nan,
)

for months in [3, 6, 12]:
    average_col = f"cc_average_utilization_{months}m"
    max_col = f"cc_max_utilization_{months}m"

    df[average_col] = calendar_rolling_stat(
        df,
        "cc_limit_utilization",
        months=months,
        stat="mean",
    )
    df[max_col] = calendar_rolling_stat(
        df,
        "cc_limit_utilization",
        months=months,
        stat="max",
    )

    df[f"cc_limit_utilization_abs_change_{months}m"] = (
        df["cc_limit_utilization"] - df[average_col]
    )
    df[f"cc_limit_utilization_change_{months}m"] = np.where(
        df[average_col].notna() & (df[average_col] != 0),
        (
            df["cc_limit_utilization"] - df[average_col]
        ) / df[average_col],
        np.nan,
    )

    df[f"cc_limit_utilization_abs_change_{months}m_max"] = (
        df["cc_limit_utilization"] - df[max_col]
    )
    df[f"cc_limit_utilization_change_{months}m_max"] = np.where(
        df[max_col].notna() & (df[max_col] != 0),
        (
            df["cc_limit_utilization"] - df[max_col]
        ) / df[max_col],
        np.nan,
    )

# Daily calendar logic replaces the previous row-gap patch.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =====================================================================
# GRID / DISTRIBUTION REVIEW
# =====================================================================
GRID_REVIEW_COLUMNS = [
    "limit_utilization",
    "limit_utilization_abs_change_1m",
    "limit_utilization_abs_change_3m",
    "limit_utilization_abs_change_6m",
    "limit_utilization_abs_change_12m",
    "limit_utilization_change_1m",
    "limit_utilization_change_3m",
    "limit_utilization_change_6m",
    "limit_utilization_change_12m",
    "cc_limit_utilization",
    "cc_limit_utilization_abs_change_1m",
    "cc_limit_utilization_abs_change_3m",
    "cc_limit_utilization_abs_change_6m",
    "cc_limit_utilization_abs_change_12m",
    "cc_limit_utilization_change_1m",
    "cc_limit_utilization_change_3m",
    "cc_limit_utilization_change_6m",
    "cc_limit_utilization_change_12m",
    "cash_advance_balance_ratio",
    "neobank_topup_balance_ratio",
    "cash_neobank_balance_ratio",
    "cash_deposit_amount",
    "cash_deposit_count",
    "cash_advance_transaction_limit_utilization",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "neobank_transaction_limit_utilization",
    "neobank_limit_utilization_before_transaction",
    "neobank_limit_utilization_after_transaction",
    "arrears_days_above_250",
    "arrears_days_above_500",
]

for col in GRID_REVIEW_COLUMNS:
    if col not in df.columns:
        continue

    values = coerce_model_numeric(df[col]).dropna()
    if values.empty:
        continue

    positive = values[values > 0]

    print(f"\nGrid review - {col}")
    print(f"  observations          : {len(values):,}")
    print(f"  positive observations : {len(positive):,}")
    print(f"  min / max             : {values.min():.6g} / {values.max():.6g}")

    if len(positive) > 0:
        q = positive.quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
        )
        print(q.to_string())


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Create targets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
DPD_THRESHOLD = 20

df["TGT_PD_CRR"]       = df["PD_CRR"] == 1
df["TGT_DEFAULT"]      = df["defaulted"] == 1
df["TGT_DPD"]          = df["dpd"] >= DPD_THRESHOLD
df["TGT_FORBEARANCE"]  = df["forbearance"] == 1
df["TGT_DIM1"]         = df["dim1"] == 1

TARGET_COLS = [
    "TGT_PD_CRR",
    "TGT_DEFAULT",
    "TGT_DPD",
    "TGT_FORBEARANCE",
    "TGT_DIM1",
]

df["TGT_ANY"] = df[TARGET_COLS].any(axis=1)

prevalence = pd.DataFrame({
    "active_rows": df[TARGET_COLS + ["TGT_ANY"]].sum(),
    "prevalence_%": (
        df[TARGET_COLS + ["TGT_ANY"]].mean() * 100
    ).round(2),
})
print(prevalence)
print(f"Entities  : {df['ENTITY_ID'].nunique()}")
print(f"Rows      : {len(df):,}")
print(f"Date range: {df['MAP_DTE'].min().date()} -> {df['MAP_DTE'].max().date()}")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Define daily signal / event functions

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
logger = logging.getLogger(__name__)


def _lead_months_float(
    event_date: pd.Series,
    signal_date: pd.Series,
) -> np.ndarray:
    return (
        (event_date - signal_date).dt.days.to_numpy(dtype=float)
        / DAYS_PER_MONTH_FOR_COST
    )


def _valid_lead_mask(
    event_date: pd.Series,
    signal_date: pd.Series,
    min_lead_months: int,
    max_lead_months: int,
) -> pd.Series:
    earliest_signal = (
        event_date - pd.DateOffset(months=max_lead_months)
    )
    latest_signal = (
        event_date - pd.DateOffset(months=min_lead_months)
    )
    return (
        signal_date.ge(earliest_signal)
        & signal_date.le(latest_signal)
    )


def build_events(
    df: pd.DataFrame,
    target_col: str = "TGT_ANY",
    max_gap_months: int = 3,
    gca_col: str = "TOTAL_GCA",
) -> pd.DataFrame:
    """Group target observations into entity events using calendar dates."""
    active = (
        df.loc[
            df[target_col].astype(bool),
            ["ENTITY_ID", "MAP_DTE"],
        ]
        .sort_values(["ENTITY_ID", "MAP_DTE"])
        .copy()
    )

    if active.empty:
        return pd.DataFrame(
            columns=[
                "EVENT_ID",
                "ENTITY_ID",
                "EVENT_START",
                "EVENT_END",
                gca_col,
                "COST2_FN_COST",
            ]
        )

    prev_active = active.groupby(
        "ENTITY_ID",
        sort=False,
    )["MAP_DTE"].shift(1)

    active["_new_event"] = (
        prev_active.isna()
        | (
            active["MAP_DTE"]
            > prev_active + pd.DateOffset(months=max_gap_months)
        )
    ).astype(int)

    active["_event_group"] = (
        active.groupby("ENTITY_ID", sort=False)["_new_event"]
        .cumsum()
    )

    events_df = (
        active
        .groupby(
            ["ENTITY_ID", "_event_group"],
            sort=False,
        )["MAP_DTE"]
        .agg(
            EVENT_START="min",
            EVENT_END="max",
        )
        .reset_index()
        .drop(columns="_event_group")
    )

    events_df.insert(
        0,
        "EVENT_ID",
        [f"EVT{i + 1:05d}" for i in range(len(events_df))],
    )

    event_values = df[
        ["ENTITY_ID", "MAP_DTE", gca_col, "COST2_FN_COST"]
    ].rename(
        columns={"MAP_DTE": "EVENT_START"}
    )

    events_df = events_df.merge(
        event_values,
        on=["ENTITY_ID", "EVENT_START"],
        how="left",
        validate="1:1",
    )

    logger.info(
        "build_events: %d events across %d entities (max_gap=%d calendar months)",
        len(events_df),
        events_df["ENTITY_ID"].nunique(),
        max_gap_months,
    )
    return events_df


def events_with_available_history(
    events_df: pd.DataFrame,
    observations: pd.DataFrame,
    min_lead_months: int,
    max_lead_months: int,
) -> pd.DataFrame:
    """Keep events with at least one available observation in the lead window."""
    if events_df.empty or observations.empty:
        return events_df.iloc[0:0].copy()

    obs_dates = {
        entity_id: np.sort(
            grp["MAP_DTE"].dropna().to_numpy(dtype="datetime64[ns]")
        )
        for entity_id, grp in observations.groupby("ENTITY_ID", sort=False)
    }

    keep = []
    for row in events_df.itertuples(index=False):
        dates = obs_dates.get(row.ENTITY_ID)
        if dates is None or len(dates) == 0:
            keep.append(False)
            continue

        low = np.datetime64(
            row.EVENT_START - pd.DateOffset(months=max_lead_months),
            "ns",
        )
        high = np.datetime64(
            row.EVENT_START - pd.DateOffset(months=min_lead_months),
            "ns",
        )

        idx = np.searchsorted(dates, low, side="left")
        keep.append(
            idx < len(dates) and dates[idx] <= high
        )

    return events_df.loc[
        np.asarray(keep, dtype=bool)
    ].reset_index(drop=True)


def _comparison_matrix(values, thresholds, direction):
    valid = ~np.isnan(values[:, None])
    if direction == "above":
        return valid & (values[:, None] > thresholds[None, :])
    return valid & (values[:, None] < thresholds[None, :])


def _signal_starts_long(
    work: pd.DataFrame,
    predictor: str,
    thresholds: np.ndarray,
    group_values: np.ndarray,
    group_col: str,
    direction: str,
) -> pd.DataFrame:
    """Create only daily signal-start rows, avoiding one signal per active day."""
    values = coerce_model_numeric(work[predictor]).to_numpy(dtype=float)
    entities = work["ENTITY_ID"].astype(str).to_numpy()
    dates = work["MAP_DTE"].to_numpy(dtype="datetime64[ns]")
    target = work[TARGET_COL].astype(bool).to_numpy()

    n = len(work)
    same_prev_day = np.zeros(n, dtype=bool)
    if n > 1:
        day_gap = (
            dates[1:].astype("datetime64[D]")
            - dates[:-1].astype("datetime64[D]")
        ).astype(int)
        same_prev_day[1:] = (
            (entities[1:] == entities[:-1])
            & (day_gap == 1)
        )

    previous_values = np.full(n, np.nan, dtype=float)
    previous_values[1:] = values[:-1]

    previous_target = np.zeros(n, dtype=bool)
    previous_target[1:] = target[:-1]

    logic = SIGNAL_LOGIC.get(predictor, "crossing")
    records = []

    for start in range(0, len(thresholds), THRESHOLD_CHUNK_SIZE):
        stop = min(start + THRESHOLD_CHUNK_SIZE, len(thresholds))
        th = thresholds[start:stop]
        gv = group_values[start:stop]

        if logic == "active":
            raw_active = _comparison_matrix(values, th, direction)
            prev_active = np.zeros_like(raw_active, dtype=bool)
            prev_active[1:] = raw_active[:-1]
            prev_active[~same_prev_day, :] = False
            starts = raw_active & ~prev_active & ~target[:, None]
        else:
            valid = (
                same_prev_day[:, None]
                & ~np.isnan(previous_values[:, None])
                & ~np.isnan(values[:, None])
                & ~target[:, None]
                & ~previous_target[:, None]
            )

            if direction == "above":
                starts = (
                    valid
                    & (previous_values[:, None] <= th[None, :])
                    & (values[:, None] > th[None, :])
                )
            else:
                starts = (
                    valid
                    & (previous_values[:, None] >= th[None, :])
                    & (values[:, None] < th[None, :])
                )

        row_idx, threshold_idx = np.where(starts)
        if len(row_idx) == 0:
            continue

        records.append(
            pd.DataFrame({
                "ENTITY_ID": entities[row_idx],
                "MAP_DTE": pd.to_datetime(dates[row_idx]),
                group_col: gv[threshold_idx],
            })
        )

    if not records:
        return pd.DataFrame(
            columns=["ENTITY_ID", "MAP_DTE", group_col]
        )

    return pd.concat(records, ignore_index=True)


def _active_runs_long(
    work: pd.DataFrame,
    predictor: str,
    thresholds: np.ndarray,
    direction: str,
) -> pd.DataFrame:
    """Build sparse daily active-run intervals for pair logic."""
    values = coerce_model_numeric(work[predictor]).to_numpy(dtype=float)
    entities = work["ENTITY_ID"].astype(str).to_numpy()
    dates = work["MAP_DTE"].to_numpy(dtype="datetime64[ns]")
    target = work[TARGET_COL].astype(bool).to_numpy()

    n = len(work)
    same_prev_day = np.zeros(n, dtype=bool)
    same_next_day = np.zeros(n, dtype=bool)

    if n > 1:
        day_gap = (
            dates[1:].astype("datetime64[D]")
            - dates[:-1].astype("datetime64[D]")
        ).astype(int)
        contiguous = (
            (entities[1:] == entities[:-1])
            & (day_gap == 1)
        )
        same_prev_day[1:] = contiguous
        same_next_day[:-1] = contiguous

    run_frames = []

    for start in range(0, len(thresholds), THRESHOLD_CHUNK_SIZE):
        stop = min(start + THRESHOLD_CHUNK_SIZE, len(thresholds))
        th = thresholds[start:stop]

        state = (
            _comparison_matrix(values, th, direction)
            & ~target[:, None]
        )

        prev_state = np.zeros_like(state, dtype=bool)
        next_state = np.zeros_like(state, dtype=bool)
        prev_state[1:] = state[:-1]
        next_state[:-1] = state[1:]
        prev_state[~same_prev_day, :] = False
        next_state[~same_next_day, :] = False

        run_start = state & ~prev_state
        run_end = state & ~next_state

        s_row, s_th = np.where(run_start)
        e_row, e_th = np.where(run_end)

        if len(s_row) == 0:
            continue

        starts_df = pd.DataFrame({
            "ENTITY_ID": entities[s_row],
            "threshold": th[s_th],
            "RUN_START": pd.to_datetime(dates[s_row]),
            "_row": s_row,
        }).sort_values(
            ["ENTITY_ID", "threshold", "_row"]
        )

        ends_df = pd.DataFrame({
            "ENTITY_ID": entities[e_row],
            "threshold": th[e_th],
            "RUN_END": pd.to_datetime(dates[e_row]),
            "_row": e_row,
        }).sort_values(
            ["ENTITY_ID", "threshold", "_row"]
        )

        starts_df["_run_no"] = starts_df.groupby(
            ["ENTITY_ID", "threshold"],
            sort=False,
        ).cumcount()
        ends_df["_run_no"] = ends_df.groupby(
            ["ENTITY_ID", "threshold"],
            sort=False,
        ).cumcount()

        runs = starts_df.merge(
            ends_df,
            on=["ENTITY_ID", "threshold", "_run_no"],
            how="inner",
            validate="1:1",
        )

        run_frames.append(
            runs[
                ["ENTITY_ID", "threshold", "RUN_START", "RUN_END"]
            ]
        )

    if not run_frames:
        return pd.DataFrame(
            columns=["ENTITY_ID", "threshold", "RUN_START", "RUN_END"]
        )

    return pd.concat(run_frames, ignore_index=True)


def _merge_signal_starts(
    starts: pd.DataFrame,
    group_cols: list[str],
    max_signal_gap_months: int,
) -> pd.DataFrame:
    if starts.empty:
        return pd.DataFrame(
            columns=["ENTITY_ID"] + group_cols + ["SIGNAL_START", "signal_id"]
        )

    cx = starts.sort_values(
        ["ENTITY_ID"] + group_cols + ["MAP_DTE"]
    ).copy()

    prev_dte = cx.groupby(
        ["ENTITY_ID"] + group_cols,
        sort=False,
    )["MAP_DTE"].shift(1)

    cx["_new_sig"] = (
        prev_dte.isna()
        | (
            cx["MAP_DTE"]
            > prev_dte + pd.DateOffset(months=max_signal_gap_months)
        )
    ).astype(int)

    cx["_sig_grp"] = cx.groupby(
        ["ENTITY_ID"] + group_cols,
        sort=False,
    )["_new_sig"].cumsum()

    signals = (
        cx
        .groupby(
            ["ENTITY_ID"] + group_cols + ["_sig_grp"],
            sort=False,
        )["MAP_DTE"]
        .min()
        .rename("SIGNAL_START")
        .reset_index()
        .drop(columns="_sig_grp")
    )

    signals["signal_id"] = (
        "S" + (signals.index + 1).astype(str)
    )
    return signals


def _evaluate_signals(
    signals: pd.DataFrame,
    base: pd.DataFrame,
    grp_cols: list[str],
    events_df: pd.DataFrame,
    min_lead_months: int,
    max_lead_months: int,
    gca_col: str,
    cost_tp_pct_fn,
) -> pd.DataFrame:
    n_events = len(events_df)
    total_gca_sum = events_df[gca_col].fillna(0).sum()
    total_cost2_fn = events_df["COST2_FN_COST"].fillna(0).sum()

    if signals.empty:
        results = base.copy()
        results["n_tp"] = 0
        results["n_fp"] = 0
        results["n_fn"] = n_events
        results["sum_gca_tp"] = 0.0
        results["sum_gca_tp_wtd"] = 0.0
        results["sum_gca_fn"] = total_gca_sum
        results["sum_cost2_tp"] = 0.0
        results["sum_cost2_fn"] = total_cost2_fn
        results["max_monthly_signals"] = 0.0
        results["n_events_detectable"] = 0
        return results

    paired = signals.merge(
        events_df[
            ["EVENT_ID", "ENTITY_ID", "EVENT_START", gca_col, "COST2_FN_COST"]
        ],
        on="ENTITY_ID",
        how="left",
    )

    valid_lead = _valid_lead_mask(
        paired["EVENT_START"],
        paired["SIGNAL_START"],
        min_lead_months,
        max_lead_months,
    )

    valid_pairs = paired[
        paired["EVENT_ID"].notna()
        & valid_lead
    ].copy()

    keys = grp_cols + ["EVENT_ID"]
    first_signal = (
        valid_pairs
        .sort_values("SIGNAL_START")
        .drop_duplicates(keys, keep="first")
        .copy()
    )

    if not first_signal.empty:
        lead_months = _lead_months_float(
            first_signal["EVENT_START"],
            first_signal["SIGNAL_START"],
        )
        first_signal["_gca_wtd"] = (
            first_signal[gca_col].fillna(0).to_numpy()
            * cost_tp_pct_fn(lead_months)
        )
        first_signal["_cost2_tp"] = (
            first_signal["COST2_FN_COST"].fillna(0).to_numpy()
            * cost2_tp_pct_fn(lead_months)
        )
    else:
        first_signal["_gca_wtd"] = pd.Series(dtype=float)
        first_signal["_cost2_tp"] = pd.Series(dtype=float)

    n_events_detectable = first_signal["EVENT_ID"].nunique()

    tp_agg = (
        first_signal
        .groupby(grp_cols)
        .agg(
            n_tp=("EVENT_ID", "count"),
            sum_gca_tp=(gca_col, "sum"),
            sum_gca_tp_wtd=("_gca_wtd", "sum"),
            sum_cost2_fn_tp=("COST2_FN_COST", "sum"),
            sum_cost2_tp=("_cost2_tp", "sum"),
        )
        .reset_index()
        if not first_signal.empty
        else pd.DataFrame(columns=grp_cols + [
            "n_tp",
            "sum_gca_tp",
            "sum_gca_tp_wtd",
            "sum_cost2_fn_tp",
            "sum_cost2_tp",
        ])
    )

    matched_signals = (
        valid_pairs[
            ["ENTITY_ID", "SIGNAL_START"] + grp_cols
        ]
        .drop_duplicates()
        .assign(_m=True)
    )

    fp_counts = (
        signals
        .merge(
            matched_signals,
            on=["ENTITY_ID", "SIGNAL_START"] + grp_cols,
            how="left",
        )
        .pipe(lambda d: d[d["_m"].isna()])
        .groupby(grp_cols)
        .size()
        .rename("n_fp")
        .reset_index()
    )

    signal_month = signals.copy()
    signal_month["SIGNAL_MONTH"] = signal_month[
        "SIGNAL_START"
    ].dt.to_period("M")

    max_month_signals = (
        signal_month
        .groupby(
            grp_cols + ["SIGNAL_MONTH"]
        )
        .size()
        .rename("SIGNAL_COUNT")
        .reset_index()
        .groupby(grp_cols)["SIGNAL_COUNT"]
        .max()
        .rename("max_monthly_signals")
        .reset_index()
    )

    results = (
        base
        .merge(tp_agg, on=grp_cols, how="left")
        .merge(fp_counts, on=grp_cols, how="left")
        .merge(max_month_signals, on=grp_cols, how="left")
    )

    results["n_tp"] = results["n_tp"].fillna(0).astype(int)
    results["n_fp"] = results["n_fp"].fillna(0).astype(int)
    results["sum_gca_tp"] = results["sum_gca_tp"].fillna(0.0)
    results["sum_gca_tp_wtd"] = results["sum_gca_tp_wtd"].fillna(0.0)
    results["sum_cost2_fn_tp"] = results["sum_cost2_fn_tp"].fillna(0.0)
    results["sum_cost2_tp"] = results["sum_cost2_tp"].fillna(0.0)
    results["n_fn"] = n_events - results["n_tp"]
    results["sum_gca_fn"] = total_gca_sum - results["sum_gca_tp"]
    results["sum_cost2_fn"] = total_cost2_fn - results["sum_cost2_fn_tp"]
    results["max_monthly_signals"] = (
        results["max_monthly_signals"].fillna(0.0)
    )
    results["n_events_detectable"] = n_events_detectable

    return results


def sweep_thresholds(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    predictor: str,
    predictor_direction: str = "above",
    percentile_grid: np.ndarray | None = None,
    threshold_grid: np.ndarray | None = None,
    min_lead_months: int = 1,
    max_lead_months: int = 11,
    max_signal_gap_months: int = 2,
    gca_col: str = "TOTAL_GCA",
    target_col: str = "TGT_ANY",
    cost_tp_pct_fn=None,
) -> pd.DataFrame:
    """Daily threshold sweep with sparse signal-start generation."""
    if threshold_grid is not None and percentile_grid is not None:
        raise ValueError(
            "Pass either percentile_grid or threshold_grid, not both."
        )

    if threshold_grid is not None:
        thresholds = np.asarray(threshold_grid, dtype=float)
        grp_col = "threshold"
        group_values = thresholds.copy()
        base = pd.DataFrame({"threshold": thresholds})
    else:
        pct_grid = (
            np.arange(1, 100, 1)
            if percentile_grid is None
            else np.asarray(percentile_grid, dtype=float)
        )
        thresholds = np.percentile(
            coerce_model_numeric(df[predictor]).dropna().to_numpy(),
            pct_grid,
        )
        grp_col = "percentile"
        group_values = pct_grid
        base = pd.DataFrame({
            "percentile": pct_grid,
            "threshold": thresholds,
        })

    work = (
        df[["ENTITY_ID", "MAP_DTE", predictor, target_col]]
        .sort_values(["ENTITY_ID", "MAP_DTE"])
        .copy()
    )

    starts = _signal_starts_long(
        work,
        predictor,
        thresholds,
        group_values,
        grp_col,
        predictor_direction,
    )

    signals = _merge_signal_starts(
        starts,
        [grp_col],
        max_signal_gap_months,
    )

    results = _evaluate_signals(
        signals,
        base,
        [grp_col],
        events_df,
        min_lead_months,
        max_lead_months,
        gca_col,
        cost_tp_pct_fn,
    )

    logger.info(
        "sweep_thresholds daily: %d thresholds predictor=%s direction=%s",
        len(results),
        predictor,
        predictor_direction,
    )

    out_cols = (
        ["percentile", "threshold"]
        if grp_col == "percentile"
        else ["threshold"]
    )
    return results[out_cols + [
        "n_tp",
        "n_fp",
        "n_fn",
        "sum_gca_tp",
        "sum_gca_tp_wtd",
        "sum_gca_fn",
        "sum_cost2_tp",
        "sum_cost2_fn",
        "max_monthly_signals",
        "n_events_detectable",
    ]]


def _crossing_confirms_within_run(
    crossings_x: pd.DataFrame,
    runs_y: pd.DataFrame,
    x_thresh_col: str,
    y_thresh_col: str,
    max_predictor_gap_months: int,
) -> pd.DataFrame:
    if crossings_x.empty or runs_y.empty:
        return pd.DataFrame(
            columns=["ENTITY_ID", "MAP_DTE", x_thresh_col, y_thresh_col]
        )

    m = crossings_x.merge(
        runs_y,
        on="ENTITY_ID",
        how="inner",
    )

    latest_confirmation = (
        m["RUN_START"]
        + pd.DateOffset(months=max_predictor_gap_months)
    )

    keep = (
        m["MAP_DTE"].ge(m["RUN_START"])
        & m["MAP_DTE"].le(m["RUN_END"])
        & m["MAP_DTE"].le(latest_confirmation)
    )

    return (
        m.loc[
            keep,
            ["ENTITY_ID", "MAP_DTE", "threshold_x", "threshold_y"],
        ]
        .rename(
            columns={
                "threshold_x": x_thresh_col,
                "threshold_y": y_thresh_col,
            }
        )
    )


def sweep_thresholds_pair(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    predictor_a: str,
    predictor_b: str,
    predictor_direction_a: str = "above",
    predictor_direction_b: str = "above",
    threshold_grid_a: np.ndarray | None = None,
    threshold_grid_b: np.ndarray | None = None,
    min_lead_months: int = 1,
    max_lead_months: int = 11,
    max_signal_gap_months: int = 2,
    max_predictor_gap_months: int = 3,
    gca_col: str = "TOTAL_GCA",
    target_col: str = "TGT_ANY",
    cost_tp_pct_fn=None,
) -> pd.DataFrame:
    """Daily pair sweep: the second crossing can confirm while the first remains active."""
    col_a = f"threshold_{predictor_a}"
    col_b = f"threshold_{predictor_b}"
    grp_cols = [col_a, col_b]

    thresholds_a = np.asarray(threshold_grid_a, dtype=float)
    thresholds_b = np.asarray(threshold_grid_b, dtype=float)

    ai, bi = np.meshgrid(
        np.arange(len(thresholds_a)),
        np.arange(len(thresholds_b)),
        indexing="ij",
    )
    base = pd.DataFrame({
        col_a: thresholds_a[ai.ravel()],
        col_b: thresholds_b[bi.ravel()],
    })

    work = (
        df[["ENTITY_ID", "MAP_DTE", predictor_a, predictor_b, target_col]]
        .sort_values(["ENTITY_ID", "MAP_DTE"])
        .copy()
    )

    crossings_a = _signal_starts_long(
        work,
        predictor_a,
        thresholds_a,
        thresholds_a,
        "threshold",
        predictor_direction_a,
    )
    crossings_b = _signal_starts_long(
        work,
        predictor_b,
        thresholds_b,
        thresholds_b,
        "threshold",
        predictor_direction_b,
    )

    runs_a = _active_runs_long(
        work,
        predictor_a,
        thresholds_a,
        predictor_direction_a,
    )
    runs_b = _active_runs_long(
        work,
        predictor_b,
        thresholds_b,
        predictor_direction_b,
    )

    crossings_a_m = crossings_a.rename(
        columns={"threshold": "threshold_x"}
    )
    crossings_b_m = crossings_b.rename(
        columns={"threshold": "threshold_x"}
    )
    runs_a_m = runs_a.rename(
        columns={"threshold": "threshold_y"}
    )
    runs_b_m = runs_b.rename(
        columns={"threshold": "threshold_y"}
    )

    b_confirms_a = _crossing_confirms_within_run(
        crossings_a_m,
        runs_b_m,
        col_a,
        col_b,
        max_predictor_gap_months,
    )
    a_confirms_b = _crossing_confirms_within_run(
        crossings_b_m,
        runs_a_m,
        col_b,
        col_a,
        max_predictor_gap_months,
    )

    joint = (
        pd.concat(
            [b_confirms_a, a_confirms_b],
            ignore_index=True,
        )
        .drop_duplicates()
    )

    signals = _merge_signal_starts(
        joint,
        grp_cols,
        max_signal_gap_months,
    )

    results = _evaluate_signals(
        signals,
        base,
        grp_cols,
        events_df,
        min_lead_months,
        max_lead_months,
        gca_col,
        cost_tp_pct_fn,
    )

    logger.info(
        "sweep_thresholds_pair daily: %d combinations (%s x %s)",
        len(results),
        predictor_a,
        predictor_b,
    )

    return results[[
        col_a,
        col_b,
        "n_tp",
        "n_fp",
        "n_fn",
        "sum_gca_tp",
        "sum_gca_tp_wtd",
        "sum_gca_fn",
        "sum_cost2_tp",
        "sum_cost2_fn",
        "max_monthly_signals",
        "n_events_detectable",
    ]]


def sweep_thresholds_any(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    predictor: "str | tuple[str, str]",
    predictor_direction: "str | tuple[str, str]" = "above",
    threshold_grid: "np.ndarray | tuple[np.ndarray, np.ndarray] | None" = None,
    percentile_grid: "np.ndarray | None" = None,
    min_lead_months: int = 1,
    max_lead_months: int = 11,
    max_signal_gap_months: int = 2,
    max_predictor_gap_months: int = 3,
    gca_col: str = "TOTAL_GCA",
    target_col: str = "TGT_ANY",
    cost_tp_pct_fn=None,
) -> pd.DataFrame:
    if isinstance(predictor, tuple):
        pred_a, pred_b = predictor
        if isinstance(predictor_direction, tuple):
            dir_a, dir_b = predictor_direction
        else:
            dir_a = dir_b = predictor_direction

        grid_a, grid_b = threshold_grid

        return sweep_thresholds_pair(
            df,
            events_df,
            predictor_a=pred_a,
            predictor_b=pred_b,
            predictor_direction_a=dir_a,
            predictor_direction_b=dir_b,
            threshold_grid_a=grid_a,
            threshold_grid_b=grid_b,
            min_lead_months=min_lead_months,
            max_lead_months=max_lead_months,
            max_signal_gap_months=max_signal_gap_months,
            max_predictor_gap_months=max_predictor_gap_months,
            gca_col=gca_col,
            target_col=target_col,
            cost_tp_pct_fn=cost_tp_pct_fn,
        )

    direction = (
        predictor_direction[0]
        if isinstance(predictor_direction, tuple)
        else predictor_direction
    )

    return sweep_thresholds(
        df,
        events_df,
        predictor=predictor,
        predictor_direction=direction,
        threshold_grid=threshold_grid,
        percentile_grid=percentile_grid,
        min_lead_months=min_lead_months,
        max_lead_months=max_lead_months,
        max_signal_gap_months=max_signal_gap_months,
        gca_col=gca_col,
        target_col=target_col,
        cost_tp_pct_fn=cost_tp_pct_fn,
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Build events

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if COLLATERAL_COL not in df.columns:
    raise KeyError(
        f"Required collateral column '{COLLATERAL_COL}' is missing from EWS_training."
    )

df["COST2_FN_COST"] = calculate_cost2_fn(
    df[GCA_COL],
    df[COLLATERAL_COL],
)

_cost2_missing_collateral = (
    coerce_model_numeric(df[GCA_COL]).gt(0)
    & (
        coerce_model_numeric(df[COLLATERAL_COL]).isna()
        | coerce_model_numeric(df[COLLATERAL_COL]).lt(0)
    )
)
print(
    "Second cost function - rows using 10% floor because collateral is unavailable: "
    f"{int(_cost2_missing_collateral.sum()):,}"
)

events_df = build_events(
    df,
    target_col=TARGET_COL,
    max_gap_months=MAX_GAP_MONTHS,
    gca_col=GCA_COL,
)

print()
print("Event diagnostics:")
print(f"  total events              : {len(events_df):,}")
print(f"  entities with events      : {events_df['ENTITY_ID'].nunique():,}")
print(
    "  duplicate ENTITY_ID+DATE : "
    f"{training_duplicate_entity_month_rows:,}"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Trim panel boundaries using calendar dates

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
panel_end = df["MAP_DTE"].max()

entity_first_obs = (
    df.groupby("ENTITY_ID")["MAP_DTE"]
    .min()
    .rename("ENTITY_FIRST_OBS")
    .reset_index()
)

events_with_entry = events_df.merge(
    entity_first_obs,
    on="ENTITY_ID",
    how="left",
)

events_trimmed = (
    events_with_entry[
        events_with_entry["EVENT_START"]
        >= (
            events_with_entry["ENTITY_FIRST_OBS"]
            + pd.DateOffset(months=MAX_LEAD_MONTHS)
        )
    ]
    .drop(columns="ENTITY_FIRST_OBS")
    .reset_index(drop=True)
)

latest_signal_date = (
    panel_end - pd.DateOffset(months=MIN_LEAD_MONTHS)
)
df_trimmed = df[
    df["MAP_DTE"] <= latest_signal_date
].reset_index(drop=True)

print(f"Panel end   : {panel_end.date()}")
print()
print(
    f"Events : {len(events_df):,} -> {len(events_trimmed):,} "
    f"({len(events_df) - len(events_trimmed)} dropped)"
)
print(
    f"Rows   : {len(df):,} -> {len(df_trimmed):,} "
    f"({len(df) - len(df_trimmed)} dropped)"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Event rate by GCA bucket

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
GCA_BINS: list = [0, 200_000, 600_000, np.inf]
GCA_LABELS: list = [
    "small (<200k)",
    "medium (200k-600k)",
    "large (>600k)",
]

entity_gca = (
    df.sort_values("MAP_DTE")
    .groupby("ENTITY_ID", sort=False)
    .first()[[GCA_COL]]
    .reset_index()
)
entity_gca["GCA_BUCKET"] = pd.cut(
    entity_gca[GCA_COL],
    bins=GCA_BINS,
    labels=GCA_LABELS,
)

entities_with_event = set(events_trimmed["ENTITY_ID"])
events_per_entity = (
    events_trimmed.groupby("ENTITY_ID").size().rename("n_events")
)
entity_gca["has_event"] = entity_gca["ENTITY_ID"].isin(
    entities_with_event
)
entity_gca = entity_gca.merge(
    events_per_entity,
    on="ENTITY_ID",
    how="left",
)
entity_gca["n_events"] = entity_gca["n_events"].fillna(0).astype(int)

bucket_stats = (
    entity_gca
    .groupby("GCA_BUCKET", observed=True)
    .agg(
        n_entities=("ENTITY_ID", "count"),
        n_with_event=("has_event", "sum"),
        n_events=("n_events", "sum"),
    )
    .assign(
        event_rate=lambda d: (
            d["n_with_event"] / d["n_entities"]
        ).map("{:.1%}".format)
    )
)

print("Event rate per GCA bucket (based on trimmed events):\n")
print(bucket_stats.to_string())


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Exposure filter

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_inscope = df_trimmed[
    df_trimmed[GCA_COL] >= MIN_GCA
].reset_index(drop=True)

events_inscope = events_with_available_history(
    events_trimmed,
    df_inscope[["ENTITY_ID", "MAP_DTE"]],
    MIN_LEAD_MONTHS,
    MAX_LEAD_MONTHS,
)

print(f"MIN_GCA = {MIN_GCA:,.0f}")
print(f"Rows   : {len(df_trimmed):>5} -> {len(df_inscope):>5}")
print(f"Events : {len(events_trimmed):>5} -> {len(events_inscope):>5}")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Threshold sweep - all indicators

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
all_records: list[dict] = []
skipped_indicators: list[dict] = []

n_entities = df_inscope["ENTITY_ID"].nunique()

for indicator in INDICATORS:
    is_pair = isinstance(indicator, tuple)

    if is_pair:
        pred_a, pred_b = indicator
        indicator_name = f"{pred_a} x {pred_b}"
        indicator_cols = [pred_a, pred_b]
        col_a = f"threshold_{pred_a}"
        col_b = f"threshold_{pred_b}"
        direction = (
            PREDICTOR_DIRECTIONS[pred_a],
            PREDICTOR_DIRECTIONS[pred_b],
        )
    else:
        pred_a, pred_b = indicator, None
        indicator_name = indicator
        indicator_cols = [pred_a]
        col_a = "threshold"
        col_b = None
        direction = PREDICTOR_DIRECTIONS[indicator]

    unavailable_columns = [
        col
        for col in indicator_cols
        if (
            col not in df_inscope.columns
            or coerce_model_numeric(df_inscope[col]).dropna().empty
        )
    ]

    if unavailable_columns:
        skipped_indicators.append({
            "indicator": indicator_name,
            "predictor_a": pred_a,
            "predictor_b": pred_b,
            "unavailable_columns": ", ".join(unavailable_columns),
        })
        logger.warning(
            "Indicator skipped because no usable observations are available: %s (missing/empty: %s)",
            indicator_name,
            ", ".join(unavailable_columns),
        )
        continue

    def _threshold_grid_for_predictor(predictor_name: str) -> np.ndarray:
        if predictor_name in POSITIVE_PERCENTILE_GRIDS:
            return build_positive_percentile_grid(
                df_inscope[predictor_name],
                POSITIVE_PERCENTILE_GRIDS[predictor_name],
            )

        if predictor_name in PERCENTILE_GRIDS:
            values = coerce_model_numeric(
                df_inscope[predictor_name]
            ).dropna()
            return np.unique(
                np.percentile(
                    values,
                    PERCENTILE_GRIDS[predictor_name],
                )
            )

        return THRESHOLD_GRIDS[predictor_name]

    if is_pair:
        grid_a = _threshold_grid_for_predictor(pred_a)
        grid_b = _threshold_grid_for_predictor(pred_b)
        call_tgrid = (grid_a, grid_b)
        call_pctgrid = None
        n_combos = len(grid_a) * len(grid_b)
    else:
        if indicator in POSITIVE_PERCENTILE_GRIDS:
            call_tgrid = _threshold_grid_for_predictor(indicator)
            call_pctgrid = None
            n_combos = len(call_tgrid)
        elif indicator in PERCENTILE_GRIDS:
            call_tgrid = None
            call_pctgrid = PERCENTILE_GRIDS[indicator]
            n_combos = len(call_pctgrid)
        else:
            call_tgrid = THRESHOLD_GRIDS[indicator]
            call_pctgrid = None
            n_combos = len(call_tgrid)

    print(
        f"\n{indicator_name} "
        f"({n_combos:,} threshold combinations)"
    )

    for col in indicator_cols:
        numeric_count = int(
            coerce_model_numeric(df_inscope[col]).notna().sum()
        )
        print(
            f"  {col}: dtype={df_inscope[col].dtype}, "
            f"numeric observations={numeric_count:,}"
        )

    available_observations = (
        df_inscope
        .dropna(subset=indicator_cols)[["ENTITY_ID", "MAP_DTE"]]
    )
    events_indicator = events_with_available_history(
        events_trimmed,
        available_observations,
        MIN_LEAD_MONTHS,
        MAX_LEAD_MONTHS,
    )

    predictor_gap_months = (
        MAX_GAP_BETWEEN_PREDICTORS_MONTHS.get(
            indicator,
            DEFAULT_PREDICTOR_GAP_MONTHS,
        )
        if is_pair
        else DEFAULT_PREDICTOR_GAP_MONTHS
    )

    sweep_b = sweep_thresholds_any(
        df_inscope,
        events_indicator,
        predictor=indicator,
        predictor_direction=direction,
        threshold_grid=call_tgrid,
        percentile_grid=call_pctgrid,
        min_lead_months=MIN_LEAD_MONTHS,
        max_lead_months=MAX_LEAD_MONTHS,
        max_signal_gap_months=MAX_SIGNAL_GAP_MONTHS,
        max_predictor_gap_months=predictor_gap_months,
        gca_col=GCA_COL,
        target_col=TARGET_COL,
        cost_tp_pct_fn=cost_tp_pct_fn,
    )

    sweep_b["total_cost"] = (
        sweep_b["sum_gca_tp_wtd"]
        + COST_TP_FIXED * sweep_b["n_tp"]
        + COST_FP * sweep_b["n_fp"]
        + COST_FN_PCT * sweep_b["sum_gca_fn"]
    )

    sweep_b["total_cost_2"] = (
        sweep_b["sum_cost2_tp"]
        + COST_FP * sweep_b["n_fp"]
        + sweep_b["sum_cost2_fn"]
    )

    opt_idx = sweep_b["total_cost"].idxmin()
    opt_idx_2 = sweep_b["total_cost_2"].idxmin()
    best_b = sweep_b.loc[opt_idx]
    best_b_2 = sweep_b.loc[opt_idx_2]

    if is_pair:
        print(
            f"  original optimal: {col_a}={best_b[col_a]:.4g} "
            f"{col_b}={best_b[col_b]:.4g} cost=EUR{best_b['total_cost']:,.0f}"
        )
        print(
            f"  cost 2 optimal  : {col_a}={best_b_2[col_a]:.4g} "
            f"{col_b}={best_b_2[col_b]:.4g} cost=EUR{best_b_2['total_cost_2']:,.0f}"
        )
    else:
        print(
            f"  original optimal: threshold={best_b['threshold']:.4g} "
            f"cost=EUR{best_b['total_cost']:,.0f}"
        )
        print(
            f"  cost 2 optimal  : threshold={best_b_2['threshold']:.4g} "
            f"cost=EUR{best_b_2['total_cost_2']:,.0f}"
        )

    n_events_detectable = (
        int(sweep_b["n_events_detectable"].iloc[0])
        if len(sweep_b)
        else 0
    )
    n_events_reachable = int(sweep_b["n_tp"].max())

    for idx, row in sweep_b.iterrows():
        n_alerts = int(row["n_tp"]) + int(row["n_fp"])
        n_events_row = int(row["n_tp"]) + int(row["n_fn"])
        accuracy = (
            row["n_tp"] / n_alerts
            if n_alerts > 0
            else float("nan")
        )
        coverage = (
            row["n_tp"] / n_events_row
            if n_events_row > 0
            else float("nan")
        )
        coverage_2 = (
            row["n_tp"] / n_events_detectable
            if n_events_detectable > 0
            else float("nan")
        )
        coverage_3 = (
            row["n_tp"] / n_events_reachable
            if n_events_reachable > 0
            else float("nan")
        )

        all_records.append({
            "indicator": indicator_name,
            "predictor_a": pred_a,
            "predictor_b": pred_b,
            "n_entities": n_entities,
            "threshold_a": row[col_a],
            "threshold_b": row[col_b] if col_b is not None else None,
            "n_tp": int(row["n_tp"]),
            "n_fp": int(row["n_fp"]),
            "n_fn": int(row["n_fn"]),
            "sum_gca_tp": float(row["sum_gca_tp"]),
            "sum_gca_tp_wtd": float(row["sum_gca_tp_wtd"]),
            "sum_gca_fn": float(row["sum_gca_fn"]),
            "sum_cost2_tp": float(row["sum_cost2_tp"]),
            "sum_cost2_fn": float(row["sum_cost2_fn"]),
            "n_alerts": n_alerts,
            "max_monthly_alerts": int(row["max_monthly_signals"]),
            "n_events": n_events_row,
            "n_events_detectable": n_events_detectable,
            "accuracy": round(accuracy, 3),
            "coverage": round(coverage, 3),
            "coverage_2": round(coverage_2, 3),
            "coverage_3": round(coverage_3, 3),
            "total_cost": row["total_cost"],
            "optimal": idx == opt_idx,
            "total_cost_2": row["total_cost_2"],
            "optimal_cost_2": idx == opt_idx_2,
        })

if skipped_indicators:
    print(
        f"\n{len(skipped_indicators)} indicator(s) skipped (missing/empty columns):"
    )
    for s in skipped_indicators:
        print(
            f"  - {s['indicator']}: {s['unavailable_columns']}"
        )

if not all_records:
    raise ValueError(
        "No model results were generated from the in-scope observations. "
        "Every indicator was skipped - check skipped_indicators above."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Results

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
results_df = pd.DataFrame(all_records)

results_df["threshold_a_rating_label"] = np.where(
    results_df["predictor_a"].isin(RATING_LEVEL_PREDICTORS),
    results_df["threshold_a"].map(rating_value_label),
    np.nan,
)
results_df["threshold_b_rating_label"] = np.where(
    results_df["predictor_b"].isin(RATING_LEVEL_PREDICTORS),
    results_df["threshold_b"].map(rating_value_label),
    np.nan,
)

n_calendar_months = max(
    int(df_inscope["MAP_DTE"].dt.to_period("M").nunique()),
    1,
)

results_df["av_nr_mon_alerts"] = round(
    results_df["n_alerts"] / n_calendar_months,
    0,
)
results_df["av_exp_nr_mon_alerts"] = (
    results_df["av_nr_mon_alerts"] / 0.7
)
results_df["max_exp_nr_mon_alerts"] = (
    results_df["max_monthly_alerts"] / 0.7
)

_cost_component_cols = [
    "n_tp",
    "n_fp",
    "n_fn",
    "sum_gca_tp_wtd",
    "sum_gca_fn",
]

_cost_diag = results_df[
    ["indicator", "total_cost"] + _cost_component_cols
].copy()

_cost_diag["_signature"] = (
    _cost_diag[_cost_component_cols]
    .round(6)
    .astype(str)
    .agg("|".join, axis=1)
)

repeated_cost_summary = (
    _cost_diag
    .groupby(["indicator", "total_cost"], dropna=False)
    .agg(
        n_threshold_rows=("_signature", "size"),
        n_distinct_cost_inputs=("_signature", "nunique"),
    )
    .reset_index()
)

repeated_cost_summary = repeated_cost_summary[
    repeated_cost_summary["n_threshold_rows"] > 1
]

print()
print("Repeated-cost diagnostic:")
print(
    "  repeated cost groups                         : "
    f"{len(repeated_cost_summary):,}"
)
print(
    "  repeated groups with identical cost inputs   : "
    f"{int((repeated_cost_summary['n_distinct_cost_inputs'] == 1).sum()):,}"
)
print(
    "  repeated groups with different cost inputs   : "
    f"{int((repeated_cost_summary['n_distinct_cost_inputs'] > 1).sum()):,}"
)

n_single = sum(isinstance(i, str) for i in INDICATORS)
n_pairs = sum(isinstance(i, tuple) for i in INDICATORS)

print(
    f"Total rows : {len(results_df):,} "
    f"({n_single} single(s), {n_pairs} pair(s) x threshold combinations)"
)
print()

n_over_100 = (results_df["coverage_3"] > 1.0).sum()
print(
    f"coverage_3 rows above 100%: {n_over_100} "
    "(should always be 0)"
)

results_df


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Result QA

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
QA_COLUMNS = [
    "qa_type",
    "check_name",
    "indicator",
    "predictor_a",
    "predictor_b",
    "unavailable_columns",
    "entity_id",
    "map_dte",
    "signal_logic",
    "direction",
    "threshold_a",
    "threshold_b",
    "predictor_value",
    "previous_value",
    "alert_generated",
    "check_passed",
    "check_value",
    "expected_value",
    "n_alerts",
    "av_nr_mon_alerts",
    "av_exp_nr_mon_alerts",
    "max_exp_nr_mon_alerts",
    "accuracy",
    "coverage",
    "coverage_2",
    "coverage_3",
    "total_cost",
]


def _qa_record(**kwargs) -> dict:
    record = {col: np.nan for col in QA_COLUMNS}
    record.update(kwargs)
    return record


qa_records: list[dict] = []

for item in skipped_indicators:
    qa_records.append(
        _qa_record(
            qa_type="INDICATOR_AVAILABILITY",
            check_name="Usable observations available",
            indicator=item["indicator"],
            predictor_a=item["predictor_a"],
            predictor_b=item["predictor_b"],
            unavailable_columns=item["unavailable_columns"],
            check_passed=False,
            check_value=0,
            expected_value=1,
        )
    )

optimal_results = (
    results_df[results_df["optimal"]]
    .copy()
    .sort_values("indicator")
)

# Active indicators are masked in target days before daily signal starts are built.
active_target_month_signal_violations = 0
for _, _opt in optimal_results[
    optimal_results["predictor_b"].isna()
].iterrows():
    _predictor = _opt["predictor_a"]
    if SIGNAL_LOGIC.get(_predictor, "crossing") != "active":
        continue
    if _predictor not in df_inscope.columns:
        continue

    _threshold = float(_opt["threshold_a"])
    _direction = PREDICTOR_DIRECTIONS[_predictor]
    _values = coerce_model_numeric(df_inscope[_predictor])
    _target = df_inscope[TARGET_COL].astype(bool)

    if _direction == "above":
        _raw_active = _values > _threshold
    else:
        _raw_active = _values < _threshold

    _masked_active = _raw_active & ~_target
    active_target_month_signal_violations += int(
        (_masked_active & _target).sum()
    )

for _, row in optimal_results.iterrows():
    qa_records.append(
        _qa_record(
            qa_type="OPTIMAL_THRESHOLD",
            indicator=row["indicator"],
            predictor_a=row["predictor_a"],
            predictor_b=row["predictor_b"],
            threshold_a=row["threshold_a"],
            threshold_b=row["threshold_b"],
            n_alerts=row["n_alerts"],
            av_nr_mon_alerts=row["av_nr_mon_alerts"],
            av_exp_nr_mon_alerts=row["av_exp_nr_mon_alerts"],
            max_exp_nr_mon_alerts=row["max_exp_nr_mon_alerts"],
            accuracy=row["accuracy"],
            coverage=row["coverage"],
            coverage_2=row["coverage_2"],
            coverage_3=row["coverage_3"],
            total_cost=row["total_cost"],
        )
    )

# Initial high-risk level must not create a crossing signal without a previous day.
optimal_single = optimal_results[
    optimal_results["predictor_b"].isna()
].copy()

level_indicators = {
    indicator
    for indicator in INDICATORS
    if isinstance(indicator, str)
    and SIGNAL_LOGIC.get(indicator, "crossing") == "crossing"
}

edge_case_alert_violations = 0

for _, opt_row in optimal_single.iterrows():
    predictor = opt_row["predictor_a"]

    if predictor not in level_indicators or predictor not in df_inscope.columns:
        continue

    threshold = float(opt_row["threshold_a"])
    direction = PREDICTOR_DIRECTIONS[predictor]

    work = (
        df_inscope[
            ["ENTITY_ID", "MAP_DTE", predictor, TARGET_COL]
        ]
        .sort_values(["ENTITY_ID", "MAP_DTE"])
        .copy()
    )

    work[predictor] = coerce_model_numeric(work[predictor])
    prev_col = f"PREV_{predictor}"
    work[prev_col] = work.groupby(
        "ENTITY_ID",
        sort=False,
    )[predictor].shift(1)

    prev_dte = work.groupby(
        "ENTITY_ID",
        sort=False,
    )["MAP_DTE"].shift(1)

    day_gap = (
        work["MAP_DTE"] - prev_dte
    ).dt.days

    work.loc[day_gap != 1, prev_col] = np.nan

    first_valid = (
        work[work[predictor].notna()]
        .groupby("ENTITY_ID", sort=False)
        .head(1)
        .copy()
    )

    if direction == "above":
        first_valid["beyond_threshold"] = (
            first_valid[predictor] > threshold
        )
        first_valid["alert_generated"] = (
            first_valid[prev_col].notna()
            & (first_valid[prev_col] <= threshold)
            & (first_valid[predictor] > threshold)
        )
    else:
        first_valid["beyond_threshold"] = (
            first_valid[predictor] < threshold
        )
        first_valid["alert_generated"] = (
            first_valid[prev_col].notna()
            & (first_valid[prev_col] >= threshold)
            & (first_valid[predictor] < threshold)
        )

    high_start = first_valid[
        first_valid["beyond_threshold"]
    ].copy()

    n_initial_alerts = (
        int(high_start["alert_generated"].sum())
        if len(high_start)
        else 0
    )

    edge_case_alert_violations += n_initial_alerts

    qa_records.append(
        _qa_record(
            qa_type="EDGE_CASE_SUMMARY",
            check_name="Initial high-risk level alerts",
            indicator=predictor,
            predictor_a=predictor,
            signal_logic="crossing",
            direction=direction,
            threshold_a=threshold,
            check_passed=(n_initial_alerts == 0),
            check_value=n_initial_alerts,
            expected_value=0,
            n_alerts=len(high_start),
        )
    )

    for _, row in high_start.iterrows():
        qa_records.append(
            _qa_record(
                qa_type="EDGE_CASE_START_LEVEL",
                check_name=(
                    "Initial high-risk level does not create a crossing alert"
                ),
                indicator=predictor,
                predictor_a=predictor,
                entity_id=row["ENTITY_ID"],
                map_dte=row["MAP_DTE"],
                signal_logic="crossing",
                direction=direction,
                threshold_a=threshold,
                predictor_value=row[predictor],
                previous_value=row[prev_col],
                alert_generated=bool(row["alert_generated"]),
                check_passed=not bool(row["alert_generated"]),
                expected_value=0,
            )
        )

coverage_range_violations = int(
    (
        (results_df["coverage"] < 0)
        | (results_df["coverage"] > 1)
        | (results_df["coverage_2"] < 0)
        | (results_df["coverage_2"] > 1)
        | (results_df["coverage_3"] < 0)
        | (results_df["coverage_3"] > 1)
    )
    .fillna(False)
    .sum()
)

coverage_order_violations = int(
    (
        (
            results_df["coverage_2"].notna()
            & results_df["coverage"].notna()
            & (results_df["coverage_2"] < results_df["coverage"])
        )
        | (
            results_df["coverage_3"].notna()
            & results_df["coverage_2"].notna()
            & (results_df["coverage_3"] < results_df["coverage_2"])
        )
    ).sum()
)

accuracy_range_violations = int(
    (
        (results_df["accuracy"] < 0)
        | (results_df["accuracy"] > 1)
    )
    .fillna(False)
    .sum()
)

alert_identity_violations = int(
    (
        results_df["n_alerts"]
        != results_df["n_tp"] + results_df["n_fp"]
    ).sum()
)

event_identity_violations = int(
    (
        results_df["n_events"]
        != results_df["n_tp"] + results_df["n_fn"]
    ).sum()
)

optimal_counts = results_df.groupby("indicator")["optimal"].sum()
optimal_count_violations = int((optimal_counts != 1).sum())

lead_window_configuration_violations = int(
    not (MIN_LEAD_MONTHS == 1 and MAX_LEAD_MONTHS == 11)
)

target_month_signal_mask_violations = (
    active_target_month_signal_violations
)

sanity_checks = [
    (
        "Duplicate ENTITY_ID + MAP_DTE rows in daily training",
        training_duplicate_entity_month_rows,
    ),
    ("Coverage values within [0, 1]", coverage_range_violations),
    ("Coverage ordering", coverage_order_violations),
    ("Accuracy values within [0, 1]", accuracy_range_violations),
    ("Alert count identity", alert_identity_violations),
    ("Event count identity", event_identity_violations),
    ("One optimal threshold per indicator", optimal_count_violations),
    ("Initial high-risk level crossing behaviour", edge_case_alert_violations),
    ("Standalone signals excluded from target days", target_month_signal_mask_violations),
    (
        "Signal-event lead window configured as 1-11 calendar months inclusive",
        lead_window_configuration_violations,
    ),
]

for check_name, violations in sanity_checks:
    qa_records.append(
        _qa_record(
            qa_type="MODEL_CHECK",
            check_name=check_name,
            check_passed=(violations == 0),
            check_value=violations,
            expected_value=0,
        )
    )

test_results_df = pd.DataFrame(
    qa_records,
    columns=QA_COLUMNS,
)

print()
print("QA summary:")
print(
    test_results_df[
        test_results_df["qa_type"] == "MODEL_CHECK"
    ][
        ["check_name", "check_passed", "check_value"]
    ].to_string(index=False)
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Write outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
results3 = dataiku.Dataset("results")
results3.write_with_schema(results_df)

test_results3 = dataiku.Dataset("test_results")
test_results3.write_with_schema(test_results_df)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Plots (single variable -> histogram, two variables -> scatter)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt

for indicator in INDICATORS:
    print("\n" + "=" * 80)

    if isinstance(indicator, str):
        col = indicator

        if col not in df.columns:
            print(f"Column '{col}' not found.")
            continue

        series = (
            coerce_model_numeric(df[col])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if series.empty:
            print(f"No numeric observations available for '{col}'.")
            continue

        print(f"Distribution summary for: {col}")
        print(f"Count    : {series.count():,.0f}")
        print(f"Mean     : {series.mean():.4f}")
        print(f"Median   : {series.median():.4f}")
        print(f"Std Dev  : {series.std():.4f}")
        print(f"Min      : {series.min():.4f}")
        print(f"25%      : {series.quantile(0.25):.4f}")
        print(f"75%      : {series.quantile(0.75):.4f}")
        print(f"Max      : {series.max():.4f}")

        plt.figure(figsize=(8, 5))
        plt.hist(series, bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.show()
        plt.close()

    elif isinstance(indicator, tuple) and len(indicator) == 2:
        x_col, y_col = indicator
        missing_cols = [
            c for c in (x_col, y_col) if c not in df.columns
        ]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            continue

        temp = df[[x_col, y_col]].copy()
        temp[x_col] = (
            coerce_model_numeric(temp[x_col])
            .replace([np.inf, -np.inf], np.nan)
        )
        temp[y_col] = (
            coerce_model_numeric(temp[y_col])
            .replace([np.inf, -np.inf], np.nan)
        )
        temp = temp.dropna()

        if temp.empty:
            print(
                f"No numeric observations available for {x_col} vs {y_col}."
            )
            continue

        print(f"Scatter analysis: {x_col} vs {y_col}")
        print("\nSummary statistics:")
        print(temp.describe().T)

        corr = temp[x_col].corr(temp[y_col])
        print(f"\nCorrelation: {corr:.4f}")

        plot_data = (
            temp
            if len(temp) <= 5000
            else temp.sample(5000, random_state=0)
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(
            plot_data[x_col],
            plot_data[y_col],
            alpha=0.6,
        )
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(alpha=0.3)
        plt.show()
        plt.close()

    else:
        print(f"Invalid indicator format: {indicator}")