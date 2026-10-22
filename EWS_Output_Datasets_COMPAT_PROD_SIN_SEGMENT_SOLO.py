# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-


import dataiku
import numpy as np
import pandas as pd

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration

ALERT_DATASET = "EWS_Daily_Alerts"
CURRENT_DATASET = "Available_Indicators_Current"
REFERENCE_DATASET = "Implementation_Reference_Date"

ENTITY_DATASET = "Entity_Information"
RM_DATASET = "RM_Data_Daily_Run"
BALANCE_DATASET = "Customer_Balances"
LOAN_DATASET = "vars_loan"
LOAN_LEVEL_DATASET = "vars_loan_level"
ACCOUNT_DATASET = "ACCOUNT_VW_1"

BREACH_OUTPUT = "ews_ewi_breach_information"
ENTITY_OUTPUT = "ews_entity_static_data"
EXPOSURE_OUTPUT = "ews_exposure_static_data"

PURE_GEC_HEAD_DOWNGRADE_CODES = {
    "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
}


DEPRECATED_INDICATOR_CODES = {
    "RRE_DPD_GT_30_ARREARS_GT_5000_OR_EXPOSURE_GT_1M",
    "RRE_DPD_GT_60_ARREARS_GT_100",
}


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers

def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        })
    )


def normalize_date(series):
    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


def concat_unique(series):
    values = sorted({
        str(value).strip()
        for value in series
        if pd.notna(value)
        and str(value).strip()
        not in {"", "<NA>", "nan", "None"}
    })
    return ", ".join(values) if values else np.nan


def first_non_null(series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def combine_errors(*errors):
    values = [
        str(error).strip()
        for error in errors
        if pd.notna(error)
        and str(error).strip() != ""
    ]
    return "; ".join(dict.fromkeys(values))


def ensure_columns(frame, dataset_name, columns):
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            "{} is missing required column(s): {}".format(
                dataset_name,
                ", ".join(missing),
            )
        )


def optional_column(frame, column, default=np.nan):
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def numeric_value(row, column):
    if column not in row.index:
        return np.nan
    return pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce",
    ).iloc[0]


def text_value(row, column):
    if column not in row.index:
        return np.nan

    value = row[column]

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "" or value.upper() in {"NAN", "NONE", "<NA>", "MISS"}:
        return np.nan

    return value


def select_latest_snapshot(frame, date_column, target_date):
    if date_column not in frame.columns:
        return frame.copy()

    result = frame.copy()
    result[date_column] = normalize_date(result[date_column])

    available = (
        result.loc[
            result[date_column].notna()
            & result[date_column].le(target_date),
            date_column,
        ]
        .dropna()
    )

    if available.empty:
        return result.iloc[0:0].copy()

    snapshot_date = available.max()

    return result.loc[
        result[date_column].eq(snapshot_date)
    ].copy()


def pd_to_rating(pd_value):
    value = pd.to_numeric(
        pd.Series([pd_value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return np.nan

    scale = [
        ("AAA+", 0.00000, 0.00005),
        ("AAA",  0.00005, 0.00015),
        ("AA+",  0.00015, 0.00025),
        ("AA",   0.00025, 0.00035),
        ("AA-",  0.00035, 0.00045),
        ("A+",   0.00045, 0.00055),
        ("A",    0.00055, 0.00065),
        ("A-",   0.00065, 0.00125),
        ("BBB+", 0.00125, 0.00260),
        ("BBB",  0.00260, 0.00525),
        ("BBB-", 0.00525, 0.00795),
        ("BB+",  0.00795, 0.01015),
        ("BB",   0.01015, 0.01915),
        ("BB-",  0.01915, 0.03315),
        ("B+",   0.03315, 0.06510),
        ("B",    0.06510, 0.11455),
        ("B-",   0.11455, 0.22355),
        ("CCC",  0.22355, 0.99975),
        ("D",    0.99975, 1.00500),
    ]

    for rating, lower, upper in scale:
        if lower <= value < upper:
            return rating

    return np.nan


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Centralized dates

reference_df = dataiku.Dataset(
    REFERENCE_DATASET
).get_dataframe()

if len(reference_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(
            REFERENCE_DATASET
        )
    )

ensure_columns(
    reference_df,
    REFERENCE_DATASET,
    [
        "REFERENCE_DATE",
        "PREVIOUS_DATE",
    ],
)

REFERENCE_DATE = normalize_date(
    reference_df["REFERENCE_DATE"]
).iloc[0]

PREVIOUS_DATE = normalize_date(
    reference_df["PREVIOUS_DATE"]
).iloc[0]

if "RUN_DATE" in reference_df.columns:
    RUN_DATE = normalize_date(
        reference_df["RUN_DATE"]
    ).iloc[0]
else:
    RUN_DATE = REFERENCE_DATE

if (
    pd.isna(REFERENCE_DATE)
    or pd.isna(PREVIOUS_DATE)
    or pd.isna(RUN_DATE)
):
    raise ValueError(
        "RUN_DATE / REFERENCE_DATE / PREVIOUS_DATE could not be parsed."
    )

REFERENCE_DATE_STR = REFERENCE_DATE.strftime("%Y-%m-%d")
PREVIOUS_DATE_STR = PREVIOUS_DATE.strftime("%Y-%m-%d")
RUN_DATE_STR = RUN_DATE.strftime("%Y-%m-%d")

BREACH_ID_PREFIX = REFERENCE_DATE.strftime("%Y%m%d")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily alerts

alerts = dataiku.Dataset(
    ALERT_DATASET
).get_dataframe()

required_alert_columns = [
    "ENTITY_ID",
    "TRIGGERED_INDICATOR_CODES",
]

ensure_columns(
    alerts,
    ALERT_DATASET,
    required_alert_columns,
)

alerts["ENTITY_ID"] = normalize_id(
    alerts["ENTITY_ID"]
)

if "ALERT_DATE" in alerts.columns:
    alerts["_ALERT_DATE"] = normalize_date(
        alerts["ALERT_DATE"]
    )
elif "REFERENCE_DATE" in alerts.columns:
    alerts["_ALERT_DATE"] = normalize_date(
        alerts["REFERENCE_DATE"]
    )
else:
    alerts["_ALERT_DATE"] = REFERENCE_DATE

if not alerts.empty:
    alert_dates = (
        alerts["_ALERT_DATE"]
        .dropna()
        .drop_duplicates()
    )

    if len(alert_dates) != 1:
        raise ValueError(
            "EWS_Daily_Alerts must contain exactly one alert date."
        )

    ALERT_DATE = alert_dates.iloc[0]
else:
    ALERT_DATE = REFERENCE_DATE

ALERT_DATE_STR = ALERT_DATE.strftime("%Y-%m-%d")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Alert detail

breach_source_rows = []

for _, alert_row in alerts.iterrows():

    entity_id = alert_row["ENTITY_ID"]

    code_text = alert_row.get(
        "TRIGGERED_INDICATOR_CODES",
        np.nan,
    )

    if pd.isna(code_text):
        continue

    codes = [
        code.strip()
        for code in str(code_text).split("|")
        if code.strip()
    ]

    for code in codes:
        if code in DEPRECATED_INDICATOR_CODES:
            continue

        breach_source_rows.append({
            "ENTITY_ID": entity_id,
            "INDICATOR_CODE": code,
        })


alert_detail = pd.DataFrame(
    breach_source_rows,
    columns=[
        "ENTITY_ID",
        "INDICATOR_CODE",
    ],
)

if not alert_detail.empty:
    alert_detail = (
        alert_detail
        .drop_duplicates(
            subset=[
                "ENTITY_ID",
                "INDICATOR_CODE",
            ]
        )
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current indicator snapshot

current = dataiku.Dataset(
    CURRENT_DATASET
).get_dataframe()

ensure_columns(
    current,
    CURRENT_DATASET,
    [
        "ENTITY_ID",
        "MAP_DTE",
    ],
)

current["ENTITY_ID"] = normalize_id(
    current["ENTITY_ID"]
)

current["MAP_DTE"] = normalize_date(
    current["MAP_DTE"]
)

current = current.loc[
    current["MAP_DTE"].eq(REFERENCE_DATE)
].copy()

if current["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Available_Indicators_Current contains duplicate ENTITY_ID rows."
    )

current = current.set_index(
    "ENTITY_ID",
    drop=False,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Indicator metadata

INDICATOR_META = {
    'RRE_FORBEARANCE_PROBATION_DPD_GT_0': {
        'name': 'Forbearance / probation period and days past due > 0',
        'value_1': 'DPD',
        'threshold_1': 0.0,
        'explanation': 'Forbearance or probation is active and days past due are above 0.',
        'methodology': 'Forbearance or probation active and days past due > 0',
    },
    'RRE_DPD_15_EVENT': {
        'name': 'Days past due > 15, overdraft > EUR 500 and GEC exposure > EUR 1m',
        'value_1': 'DPD',
        'threshold_1': 15.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 1000000.0,
        'explanation': 'Days past due moved above 15 while overdraft is above EUR 500 and GEC exposure is above EUR 1m.',
        'methodology': 'Days past due > 15; overdraft > EUR 500; GEC exposure > EUR 1m',
    },
    'RRE_DPD_30_EVENT': {
        'name': 'Days past due > 30, overdraft > EUR 500 and GEC exposure > EUR 250k',
        'value_1': 'DPD',
        'threshold_1': 30.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 250000.0,
        'explanation': 'Days past due moved above 30 while overdraft is above EUR 500 and GEC exposure is above EUR 250k.',
        'methodology': 'Days past due > 30; overdraft > EUR 500; GEC exposure > EUR 250k.',
    },
    'RRE_DPD_60_EVENT': {
        'name': 'Days past due > 60, overdraft > EUR 500 and GEC exposure > EUR 50k',
        'value_1': 'DPD',
        'threshold_1': 60.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 50000.0,
        'explanation': 'Days past due moved above 60 while overdraft is above EUR 500 and GEC exposure is above EUR 50k.',
        'methodology': 'Days past due > 60; overdraft > EUR 500; GEC exposure > EUR 50k.',
    },
    'RRE_DOWNGRADE_TO_STAGE2': {
        'name': 'Downgrade from Stage 1 to Stage 2',
        'value_1': 'GEC_NET_EXPOSURE',
        'threshold_1': 150000.0,
        'explanation': 'The client moved from Stage 1 to Stage 2 and the GEC exposure exceeds the materiality threshold.',
        'methodology': 'GEC exposure > EUR 150,000',
    },
    'RRE_PERSONAL_CIRCUMSTANCES_EVENT': {
        'name': 'Personal circumstances deterioration',
        'explanation': 'Employment loss or a salary decrease of at least 20% year on year for three consecutive months.',
        'methodology': 'Employment loss OR salary decrease >= 20% YoY for 3 consecutive months',
    },
    'RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M': {
        'name': 'Client rating downgrade by at least 6 notches vs 12 months ago',
        'value_1': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating has deteriorated by at least 6 notches compared with 12 months ago.',
        'methodology': 'Rating downgrade >= 6 notches vs 12 months',
    },
    'RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M': {
        'name': 'Client rating worse than BBB- and downgrade by at least 5 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'threshold_1': 11.0,
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'threshold_2': 5.0,
        'explanation': 'The client rating is worse than BBB- and has deteriorated by at least 5 notches compared with 12 months ago.',
        'methodology': 'Rating worse than BBB- and downgrade >= 5 notches vs 12 months',
    },
    'RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M': {
        'name': 'Client rating worse than BB- and downgrade by at least 4 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'threshold_1': 14.0,
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'threshold_2': 4.0,
        'explanation': 'The client rating is worse than BB- and has deteriorated by at least 4 notches compared with 12 months ago.',
        'methodology': 'Rating worse than BB- and downgrade >= 4 notches vs 12 months',
    },
    'RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M': {
        'name': 'Client rating worse than B+ and downgrade by at least 2 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than B+ and has deteriorated by at least 2 notches compared with 12 months ago.',
        'methodology': 'Rating worse than B+ and downgrade >= 2 notches vs 12 months',
    },
    'RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M': {
        'name': 'Client rating worse than B and downgrade by at least 1 notch vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than B and has deteriorated by at least 1 notch compared with 12 months ago.',
        'methodology': 'Rating worse than B and downgrade >= 1 notch vs 12 months',
    },
    'RRE_CASH_NEOBANK_ABOVE_4X_POSITIVE_BALANCES': {
        'name': 'Cash advance / transfer to neobanks > 4x positive balances',
        'explanation': "Cash advances and transfers to neobanks exceed four times the client's positive balances.",
        'methodology': 'Cash advance / neobank transfers > 4x positive balances',
    },
    'RRE_CASH_ADVANCE_LIMIT_AFTER_ABOVE_94': {
        'name': 'Cash advance and same-day limit utilization crossing above 94%',
        'value_1': 'cash_advance_amount',
        'value_2': 'cash_advance_limit_utilization_after_transaction',
        'threshold_2': 0.94,
        'explanation': 'A cash advance occurs on the reference date and credit-card limit utilization crosses from at or below 94% on the previous business day to above 94% on the reference date.',
        'methodology': 'Cash advance on reference date and utilization crossing from <= 94% previous business day to > 94% reference date.',
    },
    'RRE_LTV_ABOVE_95_CLIENT_DOWNGRADE_MORE_THAN_4_VS_12M': {
        'name': 'Loan-to-value above 95% and client downgrade by more than 4 notches',
        'value_1': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The loan-to-value ratio is above 95% and the client rating has deteriorated by more than 4 notches compared with 12 months ago.',
        'methodology': 'LTV > 95% and client downgrade > 4 notches vs 12 months',
    },
    'RRE_LTV_ABOVE_95_CLIENT_RATING_WORSE_B': {
        'name': 'Loan-to-value above 95% and client rating worse than B',
        'explanation': 'The loan-to-value ratio is above 95% and the client rating is worse than B.',
        'methodology': 'LTV > 95% and client rating worse than B',
    },
    'RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M': {
        'name': 'GEC head rating worse than B and downgrade by at least 4 notches vs 12 months ago',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'explanation': 'The GEC head rating is worse than B and has deteriorated by at least 4 notches compared with 12 months ago.',
        'methodology': 'GEC head rating worse than B and downgrade >= 4 notches vs 12 months',
    },
    'RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M': {
        'name': 'GEC head rating worse than B- and downgrade by at least 1 notch vs 12 months ago',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'explanation': 'The GEC head rating is worse than B- and has deteriorated by at least 1 notch compared with 12 months ago.',
        'methodology': 'GEC head rating worse than B- and downgrade >= 1 notch vs 12 months',
    },
    'RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M': {
        'name': 'GEC head downgrade by more than 7 notches vs 12 months ago',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'explanation': 'The GEC head rating has deteriorated by more than 7 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade > 7 notches vs 12 months',
    },
    'RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS': {
        'name': 'GEC head downgrade by at least 3 notches and client rating worse than B+',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'value_2': 'CLIENT_RATING_NOTCH',
        'explanation': 'The GEC head rating has deteriorated by at least 3 notches and the client rating is worse than B+.',
        'methodology': 'GEC head downgrade >= 3 notches and client rating worse than B+',
    },
    'RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6': {
        'name': 'GEC head downgrade by at least 1 notch and client downgrade by at least 6 notches',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The GEC head rating has deteriorated by at least 1 notch and the client rating has deteriorated by at least 6 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 1 notch and client downgrade >= 6 notches',
    },
    'RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5': {
        'name': 'GEC head downgrade by at least 3 notches and client downgrade by at least 5 notches',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'threshold_1': 3.0,
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'threshold_2': 5.0,
        'explanation': 'The GEC head rating has deteriorated by at least 3 notches and the client rating has deteriorated by at least 5 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 3 notches and client downgrade >= 5 notches',
    },
    'RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3': {
        'name': 'GEC head downgrade by at least 6 notches and client downgrade by at least 3 notches',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'threshold_1': 6.0,
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'threshold_2': 3.0,
        'explanation': 'The GEC head rating has deteriorated by at least 6 notches and the client rating has deteriorated by at least 3 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 6 notches and client downgrade >= 3 notches.',
    },
    'RRE_GCC_STAGE2_CONNECTED': {
        'name': 'Connected client classified as Stage 2',
        'value_1': 'GEC_NET_EXPOSURE',
        'threshold_1': 150000.0,
        'explanation': 'At least one connected client is classified as Stage 2.',
        'methodology': 'GEC exposure > EUR 150,000',
    },
    'RRE_GCC_UTP_CONNECTED': {
        'name': 'Connected client Unlikely to Pay',
        'explanation': 'At least one connected client has an active Unlikely to Pay status.',
        'methodology': 'Active Unlikely to Pay status in a connected client',
    },
    'OTHER_FORBEARANCE_PROBATION_DPD_GT_0': {
        'name': 'Forbearance / probation period and days past due > 0',
        'value_1': 'DPD',
        'threshold_1': 0.0,
        'explanation': 'Forbearance or probation is active and days past due are above 0.',
        'methodology': 'Forbearance or probation active and days past due > 0',
    },
    'OTHER_DPD_15_EVENT': {
        'name': 'Days past due > 15, overdraft > EUR 500 and GEC exposure > EUR 1m',
        'value_1': 'DPD',
        'threshold_1': 15.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 1000000.0,
        'explanation': 'Days past due moved above 15 while overdraft is above EUR 500 and GEC exposure is above EUR 1m.',
        'methodology': 'Days past due > 15; overdraft > EUR 500; GEC exposure > EUR 1m',
    },
    'OTHER_DPD_30_EVENT': {
        'name': 'Days past due > 30, overdraft > EUR 500 and GEC exposure > EUR 250k',
        'value_1': 'DPD',
        'threshold_1': 30.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 250000.0,
        'explanation': 'Days past due moved above 30 while overdraft is above EUR 500 and GEC exposure is above EUR 250k.',
        'methodology': 'Days past due > 30; overdraft > EUR 500; GEC exposure > EUR 250k.',
    },
    'OTHER_DPD_60_EVENT': {
        'name': 'Days past due > 60, overdraft > EUR 500 and GEC exposure > EUR 50k',
        'value_1': 'DPD',
        'threshold_1': 60.0,
        'value_2': 'GEC_NET_EXPOSURE',
        'threshold_2': 50000.0,
        'explanation': 'Days past due moved above 60 while overdraft is above EUR 500 and GEC exposure is above EUR 50k.',
        'methodology': 'Days past due > 60; overdraft > EUR 500; GEC exposure > EUR 50k.',
    },
    'OTHER_DOWNGRADE_TO_STAGE2': {
        'name': 'Downgrade from Stage 1 to Stage 2',
        'value_1': 'GEC_NET_EXPOSURE',
        'threshold_1': 150000.0,
        'explanation': 'The client moved from Stage 1 to Stage 2 and the GEC exposure exceeds the materiality threshold.',
        'methodology': 'GEC exposure > EUR 150,000',
    },
    'OTHER_FAILED_DIRECT_DEBITS_MORE_THAN_6_MTD': {
        'name': 'More than 6 failed direct debits in the current month',
        'value_1': 'FAILED_DEBIT_COUNT_MTD',
        'threshold_1': 6.0,
        'explanation': 'The number of failed direct debits in the current month is greater than 6.',
        'methodology': 'Failed direct debit count MTD > 6',
    },
    'OTHER_PERSONAL_CIRCUMSTANCES_EVENT': {
        'name': 'Personal circumstances deterioration',
        'explanation': 'Employment loss or a salary decrease of at least 20% year on year for three consecutive months.',
        'methodology': 'Employment loss OR salary decrease >= 20% YoY for 3 consecutive months',
    },
    'OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M': {
        'name': 'Client rating downgrade by at least 6 notches vs 12 months ago',
        'value_1': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating has deteriorated by at least 6 notches compared with 12 months ago.',
        'methodology': 'Rating downgrade >= 6 notches vs 12 months',
    },
    'OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M': {
        'name': 'Client rating worse than BBB- and downgrade by at least 5 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than BBB- and has deteriorated by at least 5 notches compared with 12 months ago.',
        'methodology': 'Rating worse than BBB- and downgrade >= 5 notches vs 12 months',
    },
    'OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M': {
        'name': 'Client rating worse than BB- and downgrade by at least 4 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than BB- and has deteriorated by at least 4 notches compared with 12 months ago.',
        'methodology': 'Rating worse than BB- and downgrade >= 4 notches vs 12 months',
    },
    'OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M': {
        'name': 'Client rating worse than B- and downgrade by at least 1 notch vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than B- and has deteriorated by at least 1 notch compared with 12 months ago.',
        'methodology': 'Rating worse than B- and downgrade >= 1 notch vs 12 months',
    },
    'OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M': {
        'name': 'Client rating worse than B and downgrade by at least 3 notches vs 12 months ago',
        'value_1': 'CLIENT_RATING_NOTCH',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The client rating is worse than B and has deteriorated by at least 3 notches compared with 12 months ago.',
        'methodology': 'Rating worse than B and downgrade >= 3 notches vs 12 months',
    },
    'OTHER_CASH_NEOBANK_ABOVE_HALF_POSITIVE_BALANCES': {
        'name': 'Cash advance / transfer to neobanks > 0.5x positive balances',
        'explanation': "Cash advances and transfers to neobanks exceed half of the client's positive balances.",
        'methodology': 'Cash advance / neobank transfers > 0.5x positive balances',
    },
    'OTHER_CASH_ADVANCE_LIMIT_AFTER_ABOVE_97': {
        'name': 'Cash advance and same-day limit utilization crossing above 97%',
        'value_1': 'cash_advance_amount',
        'value_2': 'cash_advance_limit_utilization_after_transaction',
        'threshold_2': 0.97,
        'explanation': 'A cash advance occurs on the reference date and credit-card limit utilization crosses from at or below 97% on the previous business day to above 97% on the reference date.',
        'methodology': 'Cash advance on reference date and utilization crossing from <= 97% previous business day to > 97% reference date.',
    },
    'OTHER_CASH_ADVANCE_AMOUNT_ABOVE_5000': {
        'name': 'Cash advance amount above EUR 5,000',
        'value_1': 'cash_advance_amount',
        'threshold_1': 5000.0,
        'explanation': 'The cash advance amount exceeds EUR 5,000.',
        'methodology': 'Cash advance amount > EUR 5,000',
    },
    'OTHER_NEOBANK_TOPUP_AMOUNT_ABOVE_1000': {
        'name': 'Neobank top-up amount above EUR 1,000',
        'value_1': 'neobank_topup_amount',
        'threshold_1': 1000.0,
        'explanation': 'The neobank top-up amount exceeds EUR 1,000.',
        'methodology': 'Neobank top-up amount > EUR 1,000.',
    },
    'OTHER_CC_UTILIZATION_ABOVE_100': {
        'name': 'Credit card limit utilization above 100%',
        'value_1': 'cc_limit_utilization',
        'threshold_1': 1.0,
        'explanation': 'Credit card utilization exceeds the available credit limit.',
        'methodology': 'Credit card utilization > 100%',
    },
    'OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95': {
        'name': 'Credit card utilization absolute change > 95% vs average credit card utilization over the last 12 months and current utilization > 95%',
        'explanation': 'Credit card utilization has increased by more than 95% compared with the average credit card utilization over the last 12 months and current utilization is above 95%.',
        'methodology': 'Absolute change > 95% vs average credit card utilization over the last 12 months and current utilization > 95%',
    },
    'OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90': {
        'name': 'Credit card utilization relative change > 100% vs 3-month maximum and current utilization > 90%',
        'explanation': 'Credit card utilization has increased by more than 100% compared with the 3-month maximum and current utilization is above 90%.',
        'methodology': 'Relative change > 100% vs 3-month maximum and current utilization > 90%',
    },
    'OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M': {
        'name': 'GEC head rating worse than B and downgrade by at least 4 notches vs 12 months ago',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'explanation': 'The GEC head rating is worse than B and has deteriorated by at least 4 notches compared with 12 months ago.',
        'methodology': 'GEC head rating worse than B and downgrade >= 4 notches vs 12 months',
    },
    'OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M': {
        'name': 'GEC head rating worse than B- and downgrade by at least 1 notch vs 12 months ago',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'explanation': 'The GEC head rating is worse than B- and has deteriorated by at least 1 notch compared with 12 months ago.',
        'methodology': 'GEC head rating worse than B- and downgrade >= 1 notch vs 12 months',
    },
    'OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M': {
        'name': 'GEC head downgrade by at least 6 notches vs 12 months ago',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'explanation': 'The GEC head rating has deteriorated by at least 6 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 6 notches vs 12 months',
    },
    'OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B': {
        'name': 'GEC head downgrade by at least 3 notches and client rating worse than B',
        'value_1': 'GEC_HEAD_RATING_NOTCH',
        'value_2': 'CLIENT_RATING_NOTCH',
        'explanation': 'The GEC head rating has deteriorated by at least 3 notches and the client rating is worse than B.',
        'methodology': 'GEC head downgrade >= 3 notches and client rating worse than B',
    },
    'OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3': {
        'name': 'GEC head downgrade by at least 6 notches and client downgrade by at least 3 notches',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The GEC head rating has deteriorated by at least 6 notches and the client rating has deteriorated by at least 3 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 6 notches and client downgrade >= 3 notches',
    },
    'OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5': {
        'name': 'GEC head downgrade by at least 3 notches and client downgrade by at least 5 notches',
        'value_1': 'GEC_HEAD_DOWNGRADE_12M',
        'value_2': 'CLIENT_DOWNGRADE_12M',
        'explanation': 'The GEC head rating has deteriorated by at least 3 notches and the client rating has deteriorated by at least 5 notches compared with 12 months ago.',
        'methodology': 'GEC head downgrade >= 3 notches and client downgrade >= 5 notches',
    },
    'OTHER_GCC_STAGE2_CONNECTED': {
        'name': 'Connected client classified as Stage 2',
        'value_1': 'GEC_NET_EXPOSURE',
        'threshold_1': 150000.0,
        'explanation': 'At least one connected client is classified as Stage 2.',
        'methodology': 'GEC exposure > EUR 150,000',
    },
    'OTHER_GCC_UTP_CONNECTED': {
        'name': 'Connected client Unlikely to Pay',
        'explanation': 'At least one connected client has an active Unlikely to Pay status.',
        'methodology': 'Active Unlikely to Pay status in a connected client',
    },
}



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Rating output configuration
#
# Front-end rating values are exported as rating labels (e.g. CCC, B-).
# Deviations remain numeric and are expressed in notches beyond threshold.

RATING_OUTPUT_CONFIG = {
    "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M": {
        "value_1": "CLIENT_DOWNGRADE_12M",
        "threshold_1": 6.0,
    },
    "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "BBB-",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 11.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 5.0,
    },
    "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "BB-",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 14.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 4.0,
    },
    "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "B+",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 15.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 2.0,
    },
    "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "B",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 16.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 1.0,
    },
    "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M": {
        "value_1": "CLIENT_DOWNGRADE_12M",
        "threshold_1": 6.0,
    },
    "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "BBB-",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 11.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 5.0,
    },
    "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "BB-",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 14.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 4.0,
    },
    "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "B-",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 17.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 1.0,
    },
    "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M": {
        "value_1_text": "CLIENT_RATING",
        "threshold_1": "B",
        "deviation_1_value": "CLIENT_RATING_NOTCH",
        "deviation_1_threshold": 16.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 3.0,
    },
    "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M": {
        "value_1_text": "GEC_HEAD_RATING",
        "threshold_1": "B",
        "deviation_1_value": "GEC_HEAD_RATING_NOTCH",
        "deviation_1_threshold": 16.0,
        "value_2": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_2": 4.0,
    },
    "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": {
        "value_1_text": "GEC_HEAD_RATING",
        "threshold_1": "B-",
        "deviation_1_value": "GEC_HEAD_RATING_NOTCH",
        "deviation_1_threshold": 17.0,
        "value_2": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_2": 1.0,
    },
    "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 7.0,
    },
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 3.0,
        "value_2_text": "CLIENT_RATING",
        "threshold_2": "B+",
        "deviation_2_value": "CLIENT_RATING_NOTCH",
        "deviation_2_threshold": 15.0,
    },
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 1.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 6.0,
    },
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 3.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 5.0,
    },
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 6.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 3.0,
    },
    "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M": {
        "value_1_text": "GEC_HEAD_RATING",
        "threshold_1": "B",
        "deviation_1_value": "GEC_HEAD_RATING_NOTCH",
        "deviation_1_threshold": 16.0,
        "value_2": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_2": 4.0,
    },
    "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": {
        "value_1_text": "GEC_HEAD_RATING",
        "threshold_1": "B-",
        "deviation_1_value": "GEC_HEAD_RATING_NOTCH",
        "deviation_1_threshold": 17.0,
        "value_2": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_2": 1.0,
    },
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 6.0,
    },
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 3.0,
        "value_2_text": "CLIENT_RATING",
        "threshold_2": "B",
        "deviation_2_value": "CLIENT_RATING_NOTCH",
        "deviation_2_threshold": 16.0,
    },
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 6.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 3.0,
    },
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5": {
        "value_1": "GEC_HEAD_DOWNGRADE_12M",
        "threshold_1": 3.0,
        "value_2": "CLIENT_DOWNGRADE_12M",
        "threshold_2": 5.0,
    },
}

for indicator_code, output_config in RATING_OUTPUT_CONFIG.items():
    if indicator_code not in INDICATOR_META:
        raise KeyError(
            "Rating output configuration references unknown indicator: {}".format(
                indicator_code
            )
        )
    INDICATOR_META[indicator_code].update(output_config)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Indicator metadata validation

unknown_indicator_codes = sorted(
    set(
        alert_detail["INDICATOR_CODE"]
        .dropna()
        .astype(str)
        .tolist()
    )
    - set(INDICATOR_META.keys())
)

if unknown_indicator_codes:
    raise ValueError(
        "Missing INDICATOR_META mapping(s): {}".format(
            ", ".join(unknown_indicator_codes)
        )
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Breach information

breach_rows = []

for _, detail_row in alert_detail.iterrows():

    entity_id = detail_row["ENTITY_ID"]
    indicator_code = detail_row["INDICATOR_CODE"]

    metadata = INDICATOR_META[
        indicator_code
    ]

    if entity_id in current.index:
        current_row = current.loc[entity_id]
    else:
        current_row = pd.Series(dtype="object")

    value_1_col = metadata.get("value_1")
    value_2_col = metadata.get("value_2")
    value_1_text_col = metadata.get("value_1_text")
    value_2_text_col = metadata.get("value_2_text")

    if value_1_text_col:
        value_1 = text_value(current_row, value_1_text_col)
    elif value_1_col:
        value_1 = numeric_value(current_row, value_1_col)
    else:
        value_1 = np.nan

    if value_2_text_col:
        value_2 = text_value(current_row, value_2_text_col)
    elif value_2_col:
        value_2 = numeric_value(current_row, value_2_col)
    else:
        value_2 = np.nan

    threshold_1 = metadata.get("threshold_1", np.nan)
    threshold_2 = metadata.get("threshold_2", np.nan)

    deviation_1 = np.nan
    deviation_2 = np.nan

    if metadata.get("deviation_1_value"):
        deviation_1 = (
            numeric_value(
                current_row,
                metadata["deviation_1_value"],
            )
            - float(metadata["deviation_1_threshold"])
        )

    if metadata.get("deviation_2_value"):
        deviation_2 = (
            numeric_value(
                current_row,
                metadata["deviation_2_value"],
            )
            - float(metadata["deviation_2_threshold"])
        )

    breach_rows.append({
        "ENTITY_ID": entity_id,
        "INDICATOR_CODE": indicator_code,
        "EWI_NAME": metadata["name"],
        "EWI_VALUE_1": value_1,
        "THRESHOLD_1": threshold_1,
        "DEVIATION_1": deviation_1,
        "EWI_VALUE_2": value_2,
        "THRESHOLD2": threshold_2,
        "DEVIATION2": deviation_2,
        "MAIN_EXPLN": metadata["explanation"],
        "EWI_methodology": metadata["methodology"],
    })


breach_information_df = pd.DataFrame(
    breach_rows,
    columns=[
        "ENTITY_ID",
        "INDICATOR_CODE",
        "EWI_NAME",
        "EWI_VALUE_1",
        "THRESHOLD_1",
        "DEVIATION_1",
        "EWI_VALUE_2",
        "THRESHOLD2",
        "DEVIATION2",
        "MAIN_EXPLN",
        "EWI_methodology",
    ],
)

duplicate_display_rows = int(
    breach_information_df.duplicated(
        subset=["ENTITY_ID", "EWI_NAME"],
        keep=False,
    ).sum()
)

if duplicate_display_rows > 0:
    breach_information_df = (
        breach_information_df
        .sort_values(["ENTITY_ID", "EWI_NAME", "INDICATOR_CODE"])
        .drop_duplicates(
            subset=["ENTITY_ID", "EWI_NAME"],
            keep="first",
        )
        .reset_index(drop=True)
    )

print(
    "Duplicate displayed EWI rows removed:",
    duplicate_display_rows,
)

# Cash / limit-utilisation output QA.
cash_limit_indicator_codes = {
    "RRE_CASH_ADVANCE_LIMIT_AFTER_ABOVE_94",
    "OTHER_CASH_ADVANCE_LIMIT_AFTER_ABOVE_97",
}

cash_limit_breaches = breach_information_df[
    breach_information_df["INDICATOR_CODE"].isin(
        cash_limit_indicator_codes
    )
].copy()

if not cash_limit_breaches.empty:
    missing_cash_limit_value = int(
        pd.to_numeric(
            cash_limit_breaches["EWI_VALUE_2"],
            errors="coerce",
        )
        .isna()
        .sum()
    )

    print(
        "Cash-limit breaches with missing post-transaction utilisation:",
        missing_cash_limit_value,
    )

    if missing_cash_limit_value > 0:
        raise ValueError(
            "Cash-limit breach output is missing the post-transaction "
            "limit-utilisation value for one or more alerts."
        )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Entity / customer source

entity_information_df = dataiku.Dataset(
    ENTITY_DATASET
).get_dataframe()

ensure_columns(
    entity_information_df,
    ENTITY_DATASET,
    [
        "ENTITY_ID",
        "CUSTOMER_ID",
        "MAP_DTE",
        "DEFAULT_STATUS",
    ],
)

entity_information_df["ENTITY_ID"] = normalize_id(
    entity_information_df["ENTITY_ID"]
)
entity_information_df["CUSTOMER_ID"] = normalize_id(
    entity_information_df["CUSTOMER_ID"]
)

entity_information_df = select_latest_snapshot(
    entity_information_df,
    "MAP_DTE",
    REFERENCE_DATE,
)

entity_information_df["DEFAULT_STATUS"] = (
    entity_information_df["DEFAULT_STATUS"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

default_entities = set(
    entity_information_df.loc[
        entity_information_df["DEFAULT_STATUS"].isin(["18", "19", "20"]),
        "ENTITY_ID",
    ]
    .dropna()
    .astype("string")
)

customer_ids_by_entity = (
    entity_information_df
    .groupby("ENTITY_ID")["CUSTOMER_ID"]
    .apply(concat_unique)
    .to_dict()
)

primary_customer_by_entity = (
    entity_information_df
    .sort_values(
        [
            "ENTITY_ID",
            "CUSTOMER_ID",
        ]
    )
    .drop_duplicates(
        subset=["ENTITY_ID"],
        keep="first",
    )
    .set_index("ENTITY_ID")["CUSTOMER_ID"]
    .to_dict()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RM data

rm_df = dataiku.Dataset(
    RM_DATASET
).get_dataframe()

for column in [
    "ENTITY_ID",
    "ID_ACTEUR",
]:
    if column in rm_df.columns:
        rm_df[column] = normalize_id(
            rm_df[column]
        )

if "REFERENCE_DATE" in rm_df.columns:
    rm_dates = normalize_date(
        rm_df["REFERENCE_DATE"]
    )

    available_rm_dates = (
        rm_dates.loc[
            rm_dates.notna()
            & rm_dates.le(REFERENCE_DATE)
        ]
        .dropna()
    )

    if not available_rm_dates.empty:
        rm_date = available_rm_dates.max()
        rm_df = rm_df.loc[
            rm_dates.eq(rm_date)
        ].copy()


def effective_rm_record(entity_id, customer_id):

    entity_text = (
        str(entity_id).strip()
        if pd.notna(entity_id)
        else ""
    )

    if (
        entity_text.startswith("6")
        and "ENTITY_ID" in rm_df.columns
    ):
        subset = rm_df.loc[
            rm_df["ENTITY_ID"].eq(entity_id)
        ].copy()

        if not subset.empty:
            if "ID_RM" in subset.columns:
                numeric_rm = pd.to_numeric(
                    subset["ID_RM"],
                    errors="coerce",
                )
                if numeric_rm.notna().any():
                    return subset.loc[
                        numeric_rm.idxmax()
                    ]
            return subset.iloc[0]

    if (
        pd.notna(customer_id)
        and "ID_ACTEUR" in rm_df.columns
    ):
        subset = rm_df.loc[
            rm_df["ID_ACTEUR"].eq(customer_id)
        ]
        if not subset.empty:
            return subset.iloc[-1]

    if "ENTITY_ID" in rm_df.columns:
        subset = rm_df.loc[
            rm_df["ENTITY_ID"].eq(entity_id)
        ]
        if not subset.empty:
            return subset.iloc[-1]

    return None


rm_rows = []

for _, row in breach_information_df.iterrows():

    entity_id = row["ENTITY_ID"]
    customer_id = primary_customer_by_entity.get(
        entity_id,
        np.nan,
    )

    record = effective_rm_record(
        entity_id,
        customer_id,
    )

    if record is None:
        rm_rows.append({
            "RM": np.nan,
            "ID_RM": np.nan,
            "RM_N+1": np.nan,
            "ID_RM_N+1": np.nan,
            "BUSINESS_LINE": np.nan,
            "TEAM": np.nan,
            "error": "#1 - identifying RM",
        })
    else:
        id_rm = record.get(
            "ID_RM",
            np.nan,
        )

        rm_rows.append({
            "RM": record.get("RM", np.nan),
            "ID_RM": id_rm,
            "RM_N+1": record.get("RM_N+1", np.nan),
            "ID_RM_N+1": record.get("ID_RM_N+1", np.nan),
            "BUSINESS_LINE": record.get("BUSINESS_LINE", np.nan),
            "TEAM": record.get("TEAM", np.nan),
            "error": (
                "#2 - RM not available"
                if pd.isna(id_rm)
                else ""
            ),
        })


rm_breach_df = pd.DataFrame(
    rm_rows,
    columns=[
        "RM",
        "ID_RM",
        "RM_N+1",
        "ID_RM_N+1",
        "BUSINESS_LINE",
        "TEAM",
        "error",
    ],
)

breach_information_df = pd.concat(
    [
        breach_information_df.reset_index(drop=True),
        rm_breach_df.reset_index(drop=True),
    ],
    axis=1,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final exclusions

if not breach_information_df.empty:

    business_line_normalized = (
        breach_information_df["BUSINESS_LINE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )

    corporate_mask = (
        business_line_normalized
        .str.contains(
            r"^corporate\s*(?:(?:&|and)\s*)?institutional\s*banking$",
            regex=True,
            na=False,
        )
    )

    exclusion_text = (
        breach_information_df[["RM", "BUSINESS_LINE", "TEAM"]]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    remediation_exit_mask = (
        exclusion_text.str.contains(
            "KYC EXIT CIC",
            regex=False,
            na=False,
        )
        | exclusion_text.str.contains(
            "BIL REMEDIATION EXIT",
            regex=False,
            na=False,
        )
    )

    default_mask = (
        breach_information_df["ENTITY_ID"].isin(default_entities)
        & breach_information_df["INDICATOR_CODE"].isin(
            PURE_GEC_HEAD_DOWNGRADE_CODES
        )
    )

    excluded_mask = (
        remediation_exit_mask
        | default_mask
        | corporate_mask
    )

    print()
    print("FINAL EXPORT EXCLUSIONS")
    print("-----------------------")
    print(
        "KYC / remediation exit breach rows removed:",
        int(remediation_exit_mask.sum()),
    )
    print(
        "Default pure GEC head downgrade rows removed:",
        int(default_mask.sum()),
    )
    print(
        "Corporate and Institutional Banking breach rows removed:",
        int(corporate_mask.sum()),
    )
    print(
        "Total breach rows removed:",
        int(excluded_mask.sum()),
    )

    breach_information_df = (
        breach_information_df.loc[~excluded_mask]
        .copy()
        .reset_index(drop=True)
    )

    remaining_default_rows = int(
        (
            breach_information_df["ENTITY_ID"].isin(default_entities)
            & breach_information_df["INDICATOR_CODE"].isin(
                PURE_GEC_HEAD_DOWNGRADE_CODES
            )
        ).sum()
    )

    if remaining_default_rows > 0:
        raise ValueError(
            "Default pure GEC head downgrade indicators remain in the final breach output."
        )

    remaining_corporate_rows = int(
        breach_information_df["BUSINESS_LINE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
        .str.contains(
            r"^corporate\s*(?:(?:&|and)\s*)?institutional\s*banking$",
            regex=True,
            na=False,
        )
        .sum()
    )

    if remaining_corporate_rows > 0:
        raise ValueError(
            "Corporate and Institutional Banking rows remain in the Retail output."
        )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Breach output

default_deviation_1 = (
    pd.to_numeric(
        breach_information_df["EWI_VALUE_1"],
        errors="coerce",
    )
    - pd.to_numeric(
        breach_information_df["THRESHOLD_1"],
        errors="coerce",
    )
)

default_deviation_2 = (
    pd.to_numeric(
        breach_information_df["EWI_VALUE_2"],
        errors="coerce",
    )
    - pd.to_numeric(
        breach_information_df["THRESHOLD2"],
        errors="coerce",
    )
)

breach_information_df["DEVIATION_1"] = (
    pd.to_numeric(
        breach_information_df["DEVIATION_1"],
        errors="coerce",
    )
    .combine_first(default_deviation_1)
)

breach_information_df["DEVIATION2"] = (
    pd.to_numeric(
        breach_information_df["DEVIATION2"],
        errors="coerce",
    )
    .combine_first(default_deviation_2)
)

rating_indicator_codes = set(RATING_OUTPUT_CONFIG.keys())

rating_breaches = breach_information_df[
    breach_information_df["INDICATOR_CODE"].isin(
        rating_indicator_codes
    )
].copy()

if not rating_breaches.empty:
    rating_label_fields = int(
        rating_breaches["EWI_VALUE_1"]
        .astype("string")
        .str.contains(r"[A-Z]", regex=True, na=False)
        .sum()
        +
        rating_breaches["EWI_VALUE_2"]
        .astype("string")
        .str.contains(r"[A-Z]", regex=True, na=False)
        .sum()
    )

    print(
        "Rating breach fields containing rating labels:",
        rating_label_fields,
    )

breach_information_df["ALERT_DATE"] = ALERT_DATE_STR
breach_information_df["DAY"] = RUN_DATE_STR

breach_information_df["Entity_Breaches_12M"] = np.nan
breach_information_df["ENTITY_BREACHES_TOTAL"] = np.nan

breach_information_df = (
    breach_information_df
    .sort_values(
        [
            "ENTITY_ID",
            "INDICATOR_CODE",
        ]
    )
    .reset_index(drop=True)
)

breach_information_df["BREACH_ID"] = [
    "{}_{:06d}".format(
        BREACH_ID_PREFIX,
        index + 1,
    )
    for index in range(
        len(breach_information_df)
    )
]

breach_columns = [
    "BREACH_ID",
    "EWI_NAME",
    "EWI_VALUE_1",
    "THRESHOLD_1",
    "DEVIATION_1",
    "EWI_VALUE_2",
    "THRESHOLD2",
    "DEVIATION2",
    "ENTITY_ID",
    "RM",
    "ID_RM",
    "RM_N+1",
    "ID_RM_N+1",
    "BUSINESS_LINE",
    "MAIN_EXPLN",
    "TEAM",
    "ALERT_DATE",
    "DAY",
    "Entity_Breaches_12M",
    "ENTITY_BREACHES_TOTAL",
    "EWI_methodology",
    "error",
]

ews_ewi_breach_information_df = (
    breach_information_df[
        breach_columns
    ].copy()
)

breached_entities = (
    ews_ewi_breach_information_df[
        "ENTITY_ID"
    ]
    .dropna()
    .drop_duplicates()
    .tolist()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Counterparty static data

entity_static_columns = [
    "ENTITY_ID",
    "ENTITY_NAME",
    "CUSTOMER_IDs",
    "INTERNAL_RATING",
    "NACE2",
    "GCC_ID",
    "GCC_NAME",
    "TOTAL_DEBT",
    "TOTAL_ASSETS",
    "DAY",
    "error",
]


if not breached_entities:
    ews_entity_static_data_df = pd.DataFrame(
        columns=entity_static_columns
    )

else:
    entity_source = (
        entity_information_df.loc[
            entity_information_df[
                "ENTITY_ID"
            ].isin(breached_entities)
        ]
        .sort_values(
            [
                "ENTITY_ID",
                "CUSTOMER_ID",
            ]
        )
        .drop_duplicates(
            subset=["ENTITY_ID"],
            keep="first",
        )
        .copy()
    )

    for column in [
        "ENTITY_NAME",
        "ECONOMIC_ACTIVITY",
        "GCC_PARENT_ID",
        "GCC_PARENT_NAME",
        "PD_CRR",
    ]:
        if column not in entity_source.columns:
            entity_source[column] = np.nan

    balances_df = dataiku.Dataset(
        BALANCE_DATASET
    ).get_dataframe()

    ensure_columns(
        balances_df,
        BALANCE_DATASET,
        [
            "ENTITY_ID",
            "ASSETS",
        ],
    )

    balances_df["ENTITY_ID"] = normalize_id(
        balances_df["ENTITY_ID"]
    )

    if "MAP_DTE" in balances_df.columns:
        balances_df = select_latest_snapshot(
            balances_df,
            "MAP_DTE",
            REFERENCE_DATE,
        )

    balances_df["ASSETS"] = pd.to_numeric(
        balances_df["ASSETS"],
        errors="coerce",
    )

    assets_by_entity = (
        balances_df
        .drop_duplicates(
            subset=["ENTITY_ID"],
            keep="last",
        )
        .set_index("ENTITY_ID")["ASSETS"]
        .to_dict()
    )

    internal_rating = entity_source["PD_CRR"].map(
        pd_to_rating
    )

    pd_percent = (
        pd.to_numeric(
            entity_source["PD_CRR"],
            errors="coerce",
        )
        * 100.0
    )

    internal_rating_text = pd.Series(
        np.nan,
        index=entity_source.index,
        dtype="object",
    )

    valid_rating = (
        internal_rating.notna()
        & pd_percent.notna()
    )

    internal_rating_text.loc[
        valid_rating
    ] = (
        internal_rating.loc[valid_rating]
        + " - "
        + pd_percent.loc[valid_rating].map(
            lambda value: "{:.2f}%".format(
                value
            )
        )
    )

    ews_entity_static_data_df = pd.DataFrame({
        "ENTITY_ID":
            entity_source["ENTITY_ID"].to_numpy(),

        "ENTITY_NAME":
            entity_source["ENTITY_NAME"].to_numpy(),

        "CUSTOMER_IDs":
            entity_source["ENTITY_ID"]
            .map(customer_ids_by_entity)
            .to_numpy(),

        "INTERNAL_RATING":
            internal_rating_text.to_numpy(),

        "NACE2":
            entity_source["ECONOMIC_ACTIVITY"].to_numpy(),

        "GCC_ID":
            entity_source["GCC_PARENT_ID"].to_numpy(),

        "GCC_NAME":
            entity_source["GCC_PARENT_NAME"].to_numpy(),

        "TOTAL_DEBT":
            np.nan,

        "TOTAL_ASSETS":
            entity_source["ENTITY_ID"]
            .map(assets_by_entity)
            .to_numpy(),

        "DAY":
            RUN_DATE_STR,
    })

    ews_entity_static_data_df["error"] = (
        ews_entity_static_data_df
        .apply(
            lambda row: combine_errors(
                (
                    "#3 - identifying customer information"
                    if pd.isna(row["CUSTOMER_IDs"])
                    else ""
                ),
                (
                    "#4 - total assets not available"
                    if pd.isna(row["TOTAL_ASSETS"])
                    else ""
                ),
            ),
            axis=1,
        )
    )

    ews_entity_static_data_df = (
        ews_entity_static_data_df[
            entity_static_columns
        ]
        .copy()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Exposure static data

exposure_columns = [
    "EXPOSURE_ID",
    "ENTITY_ID",
    "PRODUCT_TYPE",
    "NET_EXPOSURE_AMOUNT_EUR",
    "CURRENCY",
    "IFRS9_STAGE",
    "DAYS_PAST_DUE",
    "PERFORMING_STATUS",
    "GROSS_EXPOSURE_AMOUNT_EUR",
    "COLLATERAL_AMOUNT",
    "COLLATERAL_LAST_APPRAISAL_DATE",
    "LIMIT_AMOUNT",
    "DAY",
    "error",
]

if not breached_entities:
    ews_exposure_static_data_df = pd.DataFrame(
        columns=exposure_columns
    )

else:
    loan_df = dataiku.Dataset(
        LOAN_DATASET
    ).get_dataframe()

    ensure_columns(
        loan_df,
        LOAN_DATASET,
        [
            "INSTRUMENT_ID",
            "ENTITY_ID",
            "MAP_DTE",
        ],
    )

    loan_df["ENTITY_ID"] = normalize_id(
        loan_df["ENTITY_ID"]
    )
    loan_df["INSTRUMENT_ID"] = normalize_id(
        loan_df["INSTRUMENT_ID"]
    )

    loan_df = select_latest_snapshot(
        loan_df,
        "MAP_DTE",
        REFERENCE_DATE,
    )

    loan_df = loan_df.loc[
        loan_df["ENTITY_ID"].isin(
            breached_entities
        )
    ].copy()

    # --------------------------------------------------------------------------------
    ### Product Type

    loan_level_df = dataiku.Dataset(
        LOAN_LEVEL_DATASET
    ).get_dataframe()

    ensure_columns(
        loan_level_df,
        LOAN_LEVEL_DATASET,
        [
            "INSTRUMENT_ID",
            "CONTRACT_ID_SHORT",
            "ENTITY_ID",
            "INSTRUMENT_TYPE",
        ],
    )

    loan_level_df["ENTITY_ID"] = normalize_id(
        loan_level_df["ENTITY_ID"]
    )
    loan_level_df["INSTRUMENT_ID"] = normalize_id(
        loan_level_df["INSTRUMENT_ID"]
    )
    loan_level_df["CONTRACT_ID_SHORT"] = normalize_id(
        loan_level_df["CONTRACT_ID_SHORT"]
    )

    if "MAP_DTE" in loan_level_df.columns:
        loan_level_df["MAP_DTE"] = normalize_date(
            loan_level_df["MAP_DTE"]
        )

        loan_level_df = (
            loan_level_df.loc[
                loan_level_df["MAP_DTE"].le(
                    REFERENCE_DATE
                )
            ]
            .sort_values(
                [
                    "INSTRUMENT_ID",
                    "ENTITY_ID",
                    "MAP_DTE",
                ]
            )
            .drop_duplicates(
                subset=[
                    "INSTRUMENT_ID",
                    "ENTITY_ID",
                ],
                keep="last",
            )
            .copy()
        )

    product_exact = (
        loan_level_df[
            [
                "INSTRUMENT_ID",
                "ENTITY_ID",
                "INSTRUMENT_TYPE",
            ]
        ]
        .drop_duplicates(
            subset=[
                "INSTRUMENT_ID",
                "ENTITY_ID",
            ],
            keep="last",
        )
        .rename(
            columns={
                "INSTRUMENT_TYPE": "PRODUCT_TYPE"
            }
        )
        .copy()
    )

    loan_export_df = loan_df.merge(
        product_exact,
        how="left",
        on=[
            "INSTRUMENT_ID",
            "ENTITY_ID",
        ],
        validate="m:1",
    )

    exact_product_match = int(
        loan_export_df["PRODUCT_TYPE"]
        .notna()
        .sum()
    )

    fallback_lookup = (
        loan_level_df[
            [
                "ENTITY_ID",
                "CONTRACT_ID_SHORT",
                "INSTRUMENT_TYPE",
            ]
        ]
        .dropna(
            subset=[
                "ENTITY_ID",
                "CONTRACT_ID_SHORT",
            ]
        )
        .copy()
    )

    fallback_lookup[
        "RFO_CONTRACT_REFERENCE"
    ] = (
        "211"
        + fallback_lookup[
            "CONTRACT_ID_SHORT"
        ].astype("string")
    )

    fallback_type_counts = (
        fallback_lookup
        .groupby(
            [
                "ENTITY_ID",
                "RFO_CONTRACT_REFERENCE",
            ]
        )["INSTRUMENT_TYPE"]
        .nunique(
            dropna=True
        )
        .rename(
            "PRODUCT_TYPE_COUNT"
        )
        .reset_index()
    )

    ambiguous_fallback_keys = int(
        fallback_type_counts[
            "PRODUCT_TYPE_COUNT"
        ].gt(1).sum()
    )

    unique_fallback_keys = (
        fallback_type_counts.loc[
            fallback_type_counts[
                "PRODUCT_TYPE_COUNT"
            ].eq(1),
            [
                "ENTITY_ID",
                "RFO_CONTRACT_REFERENCE",
            ],
        ]
    )

    fallback_lookup = (
        fallback_lookup.merge(
            unique_fallback_keys,
            how="inner",
            on=[
                "ENTITY_ID",
                "RFO_CONTRACT_REFERENCE",
            ],
            validate="m:1",
        )
        [
            [
                "ENTITY_ID",
                "RFO_CONTRACT_REFERENCE",
                "INSTRUMENT_TYPE",
            ]
        ]
        .drop_duplicates(
            subset=[
                "ENTITY_ID",
                "RFO_CONTRACT_REFERENCE",
            ],
            keep="last",
        )
        .rename(
            columns={
                "INSTRUMENT_TYPE":
                    "PRODUCT_TYPE_FALLBACK"
            }
        )
    )

    loan_export_df = loan_export_df.merge(
        fallback_lookup,
        how="left",
        left_on=[
            "ENTITY_ID",
            "INSTRUMENT_ID",
        ],
        right_on=[
            "ENTITY_ID",
            "RFO_CONTRACT_REFERENCE",
        ],
        validate="m:1",
    )

    fallback_product_mask = (
        loan_export_df["PRODUCT_TYPE"].isna()
        & loan_export_df[
            "PRODUCT_TYPE_FALLBACK"
        ].notna()
    )

    loan_export_df.loc[
        fallback_product_mask,
        "PRODUCT_TYPE",
    ] = loan_export_df.loc[
        fallback_product_mask,
        "PRODUCT_TYPE_FALLBACK",
    ]

    fallback_product_match = int(
        fallback_product_mask.sum()
    )

    instrument_type_description = {
        "20": "Overdrafts",
        "51": "Credit card debt",
        "71": "Trade receivables",
        "80": "Finance leases",
        "1000": "Deposits other than reverse repurchase agreements",
        "1001": "Revolving credit other than overdrafts and credit card debt",
        "1002": "Credit lines other than revolving credit",
        "1003": "Reverse repurchase agreements",
        "1004": (
            "Loans other than overdrafts, convenience credit, extended credit, "
            "credit card credit, revolving credit other than credit card credit, "
            "reverse repurchase agreements, trade receivables and financial leases"
        ),
        "1005": "Debt securities",
    }

    product_type_code = (
        loan_export_df["PRODUCT_TYPE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
            "MISS": pd.NA,
        })
    )

    mapped_product_type = product_type_code.map(
        instrument_type_description
    )

    loan_export_df["PRODUCT_TYPE"] = (
        mapped_product_type
        .where(
            mapped_product_type.notna(),
            product_type_code,
        )
    )

    account_desc_match = 0
    current_account_fallback = 0

    missing_product_mask = (
        loan_export_df["PRODUCT_TYPE"].isna()
    )

    if missing_product_mask.any():

        missing_exposure_ids = (
            loan_export_df.loc[
                missing_product_mask,
                "INSTRUMENT_ID",
            ]
            .dropna()
            .astype("string")
            .str.strip()
            .unique()
            .tolist()
        )

        if missing_exposure_ids:

            account_ids_sql = ", ".join(
                "'{}'".format(
                    value.replace(
                        "'",
                        "''",
                    )
                )
                for value in missing_exposure_ids
            )

            account_query = """
            SELECT
                TRIM(
                    CAST(
                        ACCOUNT_REFERENCE
                        AS VARCHAR2(200)
                    )
                ) AS ACCOUNT_REFERENCE,

                TRIM(
                    CAST(
                        ACCOUNT_DESC
                        AS VARCHAR2(1000)
                    )
                ) AS ACCOUNT_DESC

            FROM DTM_RFO_PLY.ACCOUNT_VW

            WHERE ACCOUNT_REFERENCE IS NOT NULL

              AND TRIM(
                  CAST(
                      ACCOUNT_REFERENCE
                      AS VARCHAR2(200)
                  )
              ) IN ({account_ids})
            """.format(
                account_ids=account_ids_sql
            )

            account_desc_df = SQLExecutor2(
                dataset=dataiku.Dataset(
                    ACCOUNT_DATASET
                )
            ).query_to_df(
                account_query
            )

            if not account_desc_df.empty:

                account_desc_df[
                    "ACCOUNT_REFERENCE"
                ] = normalize_id(
                    account_desc_df[
                        "ACCOUNT_REFERENCE"
                    ]
                )

                account_desc_df[
                    "ACCOUNT_DESC"
                ] = (
                    account_desc_df[
                        "ACCOUNT_DESC"
                    ]
                    .astype("string")
                    .str.strip()
                    .replace({
                        "": pd.NA,
                        "nan": pd.NA,
                        "None": pd.NA,
                        "<NA>": pd.NA,
                    })
                )

                account_desc_counts = (
                    account_desc_df
                    .groupby(
                        "ACCOUNT_REFERENCE"
                    )[
                        "ACCOUNT_DESC"
                    ]
                    .nunique(
                        dropna=True
                    )
                )

                unique_account_refs = (
                    account_desc_counts.loc[
                        account_desc_counts.eq(1)
                    ]
                    .index
                )

                account_desc_map = (
                    account_desc_df.loc[
                        account_desc_df[
                            "ACCOUNT_REFERENCE"
                        ].isin(
                            unique_account_refs
                        )
                    ]
                    .dropna(
                        subset=[
                            "ACCOUNT_DESC",
                        ]
                    )
                    .drop_duplicates(
                        subset=[
                            "ACCOUNT_REFERENCE",
                        ],
                        keep="last",
                    )
                    .set_index(
                        "ACCOUNT_REFERENCE"
                    )[
                        "ACCOUNT_DESC"
                    ]
                )

                account_desc_values = (
                    loan_export_df[
                        "INSTRUMENT_ID"
                    ]
                    .map(
                        account_desc_map
                    )
                )

                account_desc_mask = (
                    loan_export_df[
                        "PRODUCT_TYPE"
                    ].isna()
                    & account_desc_values.notna()
                )

                loan_export_df.loc[
                    account_desc_mask,
                    "PRODUCT_TYPE",
                ] = account_desc_values.loc[
                    account_desc_mask
                ]

                account_desc_match = int(
                    account_desc_mask.sum()
                )

    remaining_missing_product = (
        loan_export_df[
            "PRODUCT_TYPE"
        ].isna()
    )

    current_account_fallback = int(
        remaining_missing_product.sum()
    )

    loan_export_df.loc[
        remaining_missing_product,
        "PRODUCT_TYPE",
    ] = "Current Account"

    product_type_populated = int(
        loan_export_df["PRODUCT_TYPE"]
        .notna()
        .sum()
    )

    product_type_missing = int(
        loan_export_df["PRODUCT_TYPE"]
        .isna()
        .sum()
    )

    print()
    print("PRODUCT TYPE QA")
    print("---------------")
    print(
        "Exposure rows                    :",
        "{:,}".format(len(loan_export_df)),
    )
    print(
        "Exact instrument matches         :",
        "{:,}".format(exact_product_match),
    )
    print(
        "Contract reference matches       :",
        "{:,}".format(fallback_product_match),
    )
    print(
        "ACCOUNT_DESC exact matches       :",
        "{:,}".format(account_desc_match),
    )
    print(
        "Current Account fallback         :",
        "{:,}".format(current_account_fallback),
    )
    print(
        "Ambiguous contract reference keys:",
        "{:,}".format(ambiguous_fallback_keys),
    )
    print(
        "PRODUCT_TYPE populated           :",
        "{:,}".format(product_type_populated),
    )
    print(
        "PRODUCT_TYPE missing             :",
        "{:,}".format(product_type_missing),
    )

    for column in [
        "NET_EXPOSURE",
        "STAGE",
        "PAST_DUE_DAYS",
        "PERFORMING_STATUS",
        "GROSS_EXPOSURE",
        "COLLATERAL_ASSIGNED_AMOUNT",
    ]:
        if column not in loan_export_df.columns:
            loan_export_df[column] = np.nan

    # Performing status
    performing_status_normalized = (
        loan_export_df["PERFORMING_STATUS"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    performing_status_output = (
        performing_status_normalized
        .map(
            {
                "F": "Performing",
                "T": "Non-performing",
            }
        )
    )

    unknown_performing_status = (
        performing_status_normalized.notna()
        & ~performing_status_normalized.isin(["F", "T"])
    )

    if unknown_performing_status.any():
        print(
            "WARNING - unmapped PERFORMING_STATUS values:",
            sorted(
                performing_status_normalized.loc[
                    unknown_performing_status
                ].dropna().unique().tolist()
            ),
        )
        performing_status_output.loc[
            unknown_performing_status
        ] = loan_export_df.loc[
            unknown_performing_status,
            "PERFORMING_STATUS",
        ].astype("string")

    ews_exposure_static_data_df = pd.DataFrame({
        "EXPOSURE_ID":
            loan_export_df["INSTRUMENT_ID"].to_numpy(),

        "ENTITY_ID":
            loan_export_df["ENTITY_ID"].to_numpy(),

        "PRODUCT_TYPE":
            loan_export_df["PRODUCT_TYPE"].to_numpy(),

        "NET_EXPOSURE_AMOUNT_EUR":
            pd.to_numeric(
                loan_export_df["NET_EXPOSURE"],
                errors="coerce",
            ).to_numpy(),

        "CURRENCY":
            "EUR",

        "IFRS9_STAGE":
            loan_export_df["STAGE"].to_numpy(),

        "DAYS_PAST_DUE":
            pd.to_numeric(
                loan_export_df["PAST_DUE_DAYS"],
                errors="coerce",
            ).to_numpy(),

        "PERFORMING_STATUS":
            performing_status_output.to_numpy(),

        "GROSS_EXPOSURE_AMOUNT_EUR":
            pd.to_numeric(
                loan_export_df["GROSS_EXPOSURE"],
                errors="coerce",
            ).to_numpy(),

        "COLLATERAL_AMOUNT":
            pd.to_numeric(
                loan_export_df[
                    "COLLATERAL_ASSIGNED_AMOUNT"
                ],
                errors="coerce",
            ).to_numpy(),

        "COLLATERAL_LAST_APPRAISAL_DATE":
            np.nan,

        "LIMIT_AMOUNT":
            np.nan,

        "DAY":
            RUN_DATE_STR,

        "error":
            "",
    })

    missing_product = (
        ews_exposure_static_data_df[
            "PRODUCT_TYPE"
        ].isna()
    )

    ews_exposure_static_data_df.loc[
        missing_product,
        "error",
    ] = "#5 - product type not available"

    ews_exposure_static_data_df = (
        ews_exposure_static_data_df[
            exposure_columns
        ]
        .copy()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA

source_indicator_alerts = len(
    alert_detail
)

final_indicator_alerts = len(
    ews_ewi_breach_information_df
)

source_clients = (
    alert_detail["ENTITY_ID"].nunique()
    if not alert_detail.empty
    else 0
)

final_clients = (
    ews_ewi_breach_information_df[
        "ENTITY_ID"
    ].nunique()
    if not ews_ewi_breach_information_df.empty
    else 0
)

unknown_indicator_rows = 0

if not alert_detail.empty:
    unknown_indicator_rows = int(
        (
            ~alert_detail[
                "INDICATOR_CODE"
            ].isin(
                INDICATOR_META.keys()
            )
        ).sum()
    )

duplicate_breach_rows = int(
    breach_information_df
    .duplicated(
        subset=[
            "ENTITY_ID",
            "INDICATOR_CODE",
        ]
    )
    .sum()
)

duplicate_displayed_ewi_rows = int(
    ews_ewi_breach_information_df
    .duplicated(
        subset=["ENTITY_ID", "EWI_NAME"],
    )
    .sum()
)

print()
print("=" * 100)
print("EWS OUTPUT CONSTRUCTION QA")
print("=" * 100)
print("Run date                     :", RUN_DATE_STR)
print("Reference date               :", REFERENCE_DATE_STR)
print("Previous date                :", PREVIOUS_DATE_STR)
print("Alert date                   :", ALERT_DATE_STR)
print("Source indicator alerts      :", source_indicator_alerts)
print("Source unique clients        :", source_clients)
print("Final breach rows            :", final_indicator_alerts)
print("Final breached clients       :", final_clients)
print("Unknown metadata mappings    :", unknown_indicator_rows)
print("Duplicate entity/code rows    :", duplicate_breach_rows)
print("Duplicate entity/EWI rows     :", duplicate_displayed_ewi_rows)
print("Entity static rows           :", len(ews_entity_static_data_df))
print("Exposure static rows         :", len(ews_exposure_static_data_df))

if unknown_indicator_rows > 0:
    raise ValueError(
        "Unknown indicator metadata mappings detected in final breach output."
    )

if duplicate_breach_rows > 0:
    raise ValueError(
        "Duplicate ENTITY_ID + INDICATOR_CODE rows detected in final breach output."
    )

if duplicate_displayed_ewi_rows > 0:
    raise ValueError(
        "Duplicate ENTITY_ID + EWI_NAME rows detected in final breach output."
    )

if not ews_ewi_breach_information_df.empty:
    print()
    print("Breaches by EWI:")
    print(
        ews_ewi_breach_information_df[
            "EWI_NAME"
        ]
        .value_counts()
        .to_string()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write outputs

dataiku.Dataset(
    BREACH_OUTPUT
).write_with_schema(
    ews_ewi_breach_information_df
)

dataiku.Dataset(
    ENTITY_OUTPUT
).write_with_schema(
    ews_entity_static_data_df
)

dataiku.Dataset(
    EXPOSURE_OUTPUT
).write_with_schema(
    ews_exposure_static_data_df
)

print()
print("Output construction completed successfully.")