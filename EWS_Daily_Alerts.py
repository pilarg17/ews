# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd


# Configuration

REFERENCE_DATASET = "Implementation_Reference_Date"
CURRENT_DATASET = "Available_Indicators_Current"
PREVIOUS_DATASET = "Available_Indicators_Previous"
OUTPUT_DATASET = "EWS_Daily_Alerts"


# Indicator catalogue

INDICATORS = [
    ("RRE_FORBEARANCE_PROBATION_DPD_GT_0", "RRE", "STATE",
     "Forbearance / probation period and DPD > 0"),
    ("RRE_DPD_15_EVENT", "RRE", "EVENT",
     "DPD > 15 and overdraft > EUR 500 and GEC exposure > EUR 1m"),
    ("RRE_DPD_30_EVENT", "RRE", "EVENT",
     "DPD > 30 and overdraft > EUR 500 and GEC exposure > EUR 250k"),
    ("RRE_DPD_60_EVENT", "RRE", "EVENT",
     "DPD > 60 and overdraft > EUR 500 and GEC exposure > EUR 50k"),
    ("RRE_DOWNGRADE_TO_STAGE2", "RRE", "STATE",
     "Downgrade to Stage 2"),
    ("RRE_PERSONAL_CIRCUMSTANCES_EVENT", "RRE", "STATE",
     "Personal circumstances deterioration"),
    ("RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M", "RRE", "STATE",
     "Client downgrade by at least 6 notches vs 12 months ago"),
    ("RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M", "RRE", "STATE",
     "Client rating worse than BBB- and downgrade by at least 5 notches vs 12 months ago"),
    ("RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M", "RRE", "STATE",
     "Client rating worse than BB- and downgrade by at least 4 notches vs 12 months ago"),
    ("RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M", "RRE", "STATE",
     "Client rating worse than B+ and downgrade by at least 2 notches vs 12 months ago"),
    ("RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M", "RRE", "STATE",
     "Client rating worse than B and downgrade by at least 1 notch vs 12 months ago"),
    ("RRE_CASH_NEOBANK_ABOVE_4X_POSITIVE_BALANCES", "RRE", "STATE",
     "Cash advance / transfer to neobanks > 4x positive balances"),
    ("RRE_CASH_ADVANCE_LIMIT_AFTER_ABOVE_94", "RRE", "STATE",
     "Cash advance and limit utilisation crossing from <=94% to >94% on the same day"),
    ("RRE_LTV_ABOVE_95_CLIENT_DOWNGRADE_MORE_THAN_4_VS_12M", "RRE", "STATE",
     "LTV > 95% and client downgrade > 4 notches vs 12 months ago"),
    ("RRE_LTV_ABOVE_95_CLIENT_RATING_WORSE_B", "RRE", "STATE",
     "LTV > 95% and client rating worse than B"),
    ("RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M", "RRE", "STATE",
     "GEC head rating worse than B and downgrade by at least 4 notches vs 12 months ago"),
    ("RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M", "RRE", "STATE",
     "GEC head rating worse than B- and downgrade by at least 1 notch vs 12 months ago"),
    ("RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M", "RRE", "STATE",
     "GEC head downgrade by more than 7 notches vs 12 months ago"),
    ("RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS", "RRE", "STATE",
     "GEC head downgrade by at least 3 notches and client rating worse than B+"),
    ("RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6", "RRE", "STATE",
     "GEC head downgrade by at least 1 notch and client downgrade by at least 6 notches"),
    ("RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5", "RRE", "STATE",
     "GEC head downgrade by at least 3 notches and client downgrade by at least 5 notches"),
    ("RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3", "RRE", "STATE",
     "GEC head downgrade by at least 6 notches and client downgrade by at least 3 notches vs 12 months ago"),
    ("RRE_GCC_STAGE2_CONNECTED", "RRE", "STATE",
     "Connected client classified as Stage 2"),
    ("RRE_GCC_UTP_CONNECTED", "RRE", "STATE",
     "UTP activated for connected client"),

    ("OTHER_FORBEARANCE_PROBATION_DPD_GT_0", "OTHER_RETAIL", "STATE",
     "Forbearance / probation period and DPD > 0"),
    ("OTHER_DPD_15_EVENT", "OTHER_RETAIL", "EVENT",
     "DPD > 15 and overdraft > EUR 500 and GEC exposure > EUR 1m"),
    ("OTHER_DPD_30_EVENT", "OTHER_RETAIL", "EVENT",
     "DPD > 30 and overdraft > EUR 500 and GEC exposure > EUR 250k"),
    ("OTHER_DPD_60_EVENT", "OTHER_RETAIL", "EVENT",
     "DPD > 60 and overdraft > EUR 500 and GEC exposure > EUR 50k"),
    ("OTHER_DOWNGRADE_TO_STAGE2", "OTHER_RETAIL", "STATE",
     "Downgrade to Stage 2"),
    ("OTHER_FAILED_DIRECT_DEBITS_MORE_THAN_6_MTD", "OTHER_RETAIL", "STATE",
     "More than 6 failed direct debits in the current month"),
    ("OTHER_PERSONAL_CIRCUMSTANCES_EVENT", "OTHER_RETAIL", "STATE",
     "Personal circumstances deterioration"),
    ("OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M", "OTHER_RETAIL", "STATE",
     "Client downgrade by at least 6 notches vs 12 months ago"),
    ("OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M", "OTHER_RETAIL", "STATE",
     "Client rating worse than BBB- and downgrade by at least 5 notches vs 12 months ago"),
    ("OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M", "OTHER_RETAIL", "STATE",
     "Client rating worse than BB- and downgrade by at least 4 notches vs 12 months ago"),
    ("OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M", "OTHER_RETAIL", "STATE",
     "Client rating worse than B- and downgrade by at least 1 notch vs 12 months ago"),
    ("OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M", "OTHER_RETAIL", "STATE",
     "Client rating worse than B and downgrade by at least 3 notches vs 12 months ago"),
    ("OTHER_CASH_NEOBANK_ABOVE_HALF_POSITIVE_BALANCES", "OTHER_RETAIL", "STATE",
     "Cash advance / transfer to neobanks > 0.5x positive balances"),
    ("OTHER_CASH_ADVANCE_LIMIT_AFTER_ABOVE_97", "OTHER_RETAIL", "STATE",
     "Cash advance and limit utilisation crossing from <=97% to >97% on the same day"),
    ("OTHER_CASH_ADVANCE_AMOUNT_ABOVE_5000", "OTHER_RETAIL", "STATE",
     "Cash advance amount > EUR 5,000"),
    ("OTHER_NEOBANK_TOPUP_AMOUNT_ABOVE_1000", "OTHER_RETAIL", "STATE",
     "Neobank top-up amount > EUR 1,000"),
    ("OTHER_CC_UTILIZATION_ABOVE_100", "OTHER_RETAIL", "STATE",
     "Credit card limit utilization > 100%"),
    ("OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95", "OTHER_RETAIL", "STATE",
     "Credit card utilization absolute change > 95% vs average credit card utilization over the last 12 months and current utilization > 95%"),
    ("OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90", "OTHER_RETAIL", "STATE",
     "Credit card utilization relative change > 100% vs 3-month max and current utilization > 90%"),
    ("OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M", "OTHER_RETAIL", "STATE",
     "GEC head rating worse than B and downgrade by at least 4 notches vs 12 months ago"),
    ("OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M", "OTHER_RETAIL", "STATE",
     "GEC head rating worse than B- and downgrade by at least 1 notch vs 12 months ago"),
    ("OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M", "OTHER_RETAIL", "STATE",
     "GEC head downgrade by at least 6 notches vs 12 months ago"),
    ("OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B", "OTHER_RETAIL", "STATE",
     "GEC head downgrade by at least 3 notches and client rating worse than B"),
    ("OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3", "OTHER_RETAIL", "STATE",
     "GEC head downgrade by at least 6 notches and client downgrade by at least 3 notches"),
    ("OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5", "OTHER_RETAIL", "STATE",
     "GEC head downgrade by at least 3 notches and client downgrade by at least 5 notches"),
    ("OTHER_GCC_STAGE2_CONNECTED", "OTHER_RETAIL", "STATE",
     "Connected client classified as Stage 2"),
    ("OTHER_GCC_UTP_CONNECTED", "OTHER_RETAIL", "STATE",
     "UTP activated for connected client"),
]

if len([x for x in INDICATORS if x[1] == "RRE"]) != 24:
    raise ValueError("RRE alert catalogue must contain 24 indicators.")

if len([x for x in INDICATORS if x[1] == "OTHER_RETAIL"]) != 27:
    raise ValueError("Other Retail alert catalogue must contain 27 indicators.")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Rating bundling

RATING_BUNDLES = {
    # ------------------------------------------------------------------ RRE
    "RRE_CLIENT_RATING": [
        "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M",
        "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M",
        "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
        "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
        "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    ],
    "RRE_GRCC_RATING": [
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS",
        "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
        "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
        "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    ],

    # --------------------------------------------------------- OTHER RETAIL
    "OTHER_CLIENT_RATING": [
        "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M",
        "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
        "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
        "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
        "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    ],
    "OTHER_GRCC_RATING": [
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B",
        "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
        "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    ],
}

indicator_catalogue_codes = {
    code
    for code, _, _, _ in INDICATORS
}

bundle_code_to_group = {}
bundle_code_to_rank = {}

for bundle_group, ranked_codes in RATING_BUNDLES.items():
    if len(ranked_codes) != len(set(ranked_codes)):
        raise ValueError(
            "Duplicate indicator code inside rating bundle {}.".format(
                bundle_group
            )
        )

    missing_bundle_codes = [
        code
        for code in ranked_codes
        if code not in indicator_catalogue_codes
    ]

    if missing_bundle_codes:
        raise KeyError(
            "Rating bundle {} contains indicator(s) not present in the "
            "alert catalogue: {}".format(
                bundle_group,
                ", ".join(missing_bundle_codes),
            )
        )

    for rank, code in enumerate(ranked_codes, start=1):
        if code in bundle_code_to_group:
            raise ValueError(
                "Indicator {} is assigned to more than one rating bundle.".format(
                    code
                )
            )

        bundle_code_to_group[code] = bundle_group
        bundle_code_to_rank[code] = rank


# Category-level bundling: independent indicators, one displayed alert per category.
# All original indicator flags remain in Available_Indicators_Current / Previous.
CC_UTILISATION_BUNDLE = [
    "OTHER_CC_UTILIZATION_ABOVE_100",
    "OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95",
    "OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90",
]

RATING_CODES = set(bundle_code_to_group)
CC_UTILISATION_CODES = set(CC_UTILISATION_BUNDLE)

if not CC_UTILISATION_CODES.issubset(indicator_catalogue_codes):
    raise ValueError("CC utilisation bundle contains an unknown indicator.")
if RATING_CODES & CC_UTILISATION_CODES:
    raise ValueError("Indicator assigned to both rating and CC bundles.")

# Ranked order for selecting the *displayed* CC breach when multiple apply.
# The underlying indicator flags are not changed. This is a technical order
# for testing; confirm the preferred displayed EWI before deployment.
for rank, code in enumerate(CC_UTILISATION_BUNDLE, start=1):
    bundle_code_to_rank[code] = rank



# Helpers

def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
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


def to_flag(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype("int8")
    )


def validate_columns(frame, dataset_name, required_columns):
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "{} missing columns: {}".format(
                dataset_name,
                ", ".join(missing),
            )
        )


def build_empty_output(indicator_codes):
    data = {
        "ALERT_DATE": pd.Series(dtype="datetime64[ns]"),
        "PREVIOUS_DATE": pd.Series(dtype="datetime64[ns]"),
        "ENTITY_ID": pd.Series(dtype="string"),
        "RRE_FLAG": pd.Series(dtype="int8"),
        "OTHER_RETAIL_FLAG": pd.Series(dtype="int8"),
        "NEW_INDICATOR_COUNT": pd.Series(dtype="int64"),
        "TRIGGERED_INDICATORS": pd.Series(dtype="string"),
        "TRIGGERED_INDICATOR_CODES": pd.Series(dtype="string"),
        "GEC_NET_EXPOSURE": pd.Series(dtype="float64"),
        "HEAD_GEC_PARENT_ID": pd.Series(dtype="string"),
    }

    for code in indicator_codes:
        data["ALERT_{}".format(code)] = pd.Series(dtype="int8")

    return pd.DataFrame(data)


# Implementation dates

reference_df = dataiku.Dataset(
    REFERENCE_DATASET
).get_dataframe()

if len(reference_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(REFERENCE_DATASET)
    )

validate_columns(
    reference_df,
    REFERENCE_DATASET,
    ["REFERENCE_DATE", "PREVIOUS_DATE"],
)

REFERENCE_DATE = (
    pd.to_datetime(
        reference_df["REFERENCE_DATE"].iloc[0],
        errors="raise",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

PREVIOUS_DATE = (
    pd.to_datetime(
        reference_df["PREVIOUS_DATE"].iloc[0],
        errors="raise",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

if PREVIOUS_DATE >= REFERENCE_DATE:
    raise ValueError(
        "Expected PREVIOUS_DATE < REFERENCE_DATE."
    )


# Load snapshots

current = dataiku.Dataset(
    CURRENT_DATASET
).get_dataframe()

previous = dataiku.Dataset(
    PREVIOUS_DATASET
).get_dataframe()

required_core = [
    "ENTITY_ID",
    "MAP_DTE",
    "RRE_FLAG",
    "OTHER_RETAIL_FLAG",
    "GEC_NET_EXPOSURE",
    "HEAD_GEC_PARENT_ID",
]

indicator_columns = [
    code
    for code, _, _, _ in INDICATORS
]

validate_columns(
    current,
    CURRENT_DATASET,
    required_core + indicator_columns,
)

if current.empty:
    raise ValueError(
        "{} is empty after the current daily build.".format(
            CURRENT_DATASET
        )
    )

current["ENTITY_ID"] = normalize_id(
    current["ENTITY_ID"]
)
current["MAP_DTE"] = normalize_date(
    current["MAP_DTE"]
)

current_dates = (
    current["MAP_DTE"]
    .dropna()
    .drop_duplicates()
)

if len(current_dates) != 1:
    raise ValueError(
        "Current must contain exactly one MAP_DTE."
    )

current_snapshot_date = pd.Timestamp(
    current_dates.iloc[0]
).normalize()

if current_snapshot_date != REFERENCE_DATE:
    raise ValueError(
        "Current MAP_DTE is {}, expected {}.".format(
            current_snapshot_date.strftime("%Y-%m-%d"),
            REFERENCE_DATE.strftime("%Y-%m-%d"),
        )
    )

if current["ENTITY_ID"].isna().any():
    raise ValueError(
        "Current contains missing ENTITY_ID values."
    )

if current["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Current contains duplicate ENTITY_ID values."
    )


print()
print("=" * 110)
print("DAILY EWS ALERT GENERATION")
print("=" * 110)
print("Current snapshot :", REFERENCE_DATE.strftime("%Y-%m-%d"))
print("Previous snapshot:", PREVIOUS_DATE.strftime("%Y-%m-%d"))
print("Current entities :", "{:,}".format(len(current)))


if previous.empty:
    raise ValueError(
        "{} is empty after the previous daily build.".format(
            PREVIOUS_DATASET
        )
    )


# Daily alert generation

validate_columns(
    previous,
    PREVIOUS_DATASET,
    required_core + indicator_columns,
)

previous["ENTITY_ID"] = normalize_id(
    previous["ENTITY_ID"]
)
previous["MAP_DTE"] = normalize_date(
    previous["MAP_DTE"]
)

previous_dates = (
    previous["MAP_DTE"]
    .dropna()
    .drop_duplicates()
)

if len(previous_dates) != 1:
    raise ValueError(
        "Previous must contain exactly one MAP_DTE."
    )

previous_snapshot_date = pd.Timestamp(
    previous_dates.iloc[0]
).normalize()

if previous_snapshot_date != PREVIOUS_DATE:
    raise ValueError(
        "Previous MAP_DTE is {}, expected {}.".format(
            previous_snapshot_date.strftime("%Y-%m-%d"),
            PREVIOUS_DATE.strftime("%Y-%m-%d"),
        )
    )

if previous["ENTITY_ID"].isna().any():
    raise ValueError(
        "Previous contains missing ENTITY_ID values."
    )

if previous["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Previous contains duplicate ENTITY_ID values."
    )

print("Previous entities:", "{:,}".format(len(previous)))

current = current.set_index(
    "ENTITY_ID",
    drop=False,
)

previous = previous.set_index(
    "ENTITY_ID",
    drop=False,
)

previous_exists = pd.Series(
    current.index.isin(previous.index),
    index=current.index,
)

previous = previous.reindex(
    current.index
)

alert_rows = []
control_rows = []

for indicator_code, portfolio, comparison_type, indicator_name in INDICATORS:
    current_flag = to_flag(
        current[indicator_code]
    )
    previous_flag = to_flag(
        previous[indicator_code]
    )

    if portfolio == "RRE":
        previous_portfolio_flag = to_flag(
            previous["RRE_FLAG"]
        )
    else:
        previous_portfolio_flag = to_flag(
            previous["OTHER_RETAIL_FLAG"]
        )

    previous_same_portfolio = (
        previous_exists
        & previous_portfolio_flag.eq(1)
    )

    if comparison_type == "EVENT":
        new_alert = (
            current_flag.eq(1)
            & previous_exists
        )

        blocked = (
            current_flag.eq(1)
            & ~previous_exists
        )

    else:
        new_alert = (
            current_flag.eq(1)
            & previous_same_portfolio
            & previous_flag.eq(0)
        )

        blocked = (
            current_flag.eq(1)
            & ~previous_same_portfolio
        )

    entity_ids = current.index[
        new_alert
    ]

    if len(entity_ids) > 0:
        detail = pd.DataFrame(
            {
                "ALERT_DATE": REFERENCE_DATE,
                "PREVIOUS_DATE": PREVIOUS_DATE,
                "ENTITY_ID": entity_ids.astype("string"),
                "PORTFOLIO": portfolio,
                "INDICATOR_CODE": indicator_code,
                "INDICATOR": indicator_name,
                "COMPARISON_TYPE": comparison_type,
                "GEC_NET_EXPOSURE": current.loc[
                    entity_ids,
                    "GEC_NET_EXPOSURE",
                ].to_numpy(),
                "HEAD_GEC_PARENT_ID": current.loc[
                    entity_ids,
                    "HEAD_GEC_PARENT_ID",
                ].astype("string").to_numpy(),
            }
        )

        alert_rows.append(detail)

    control_rows.append(
        {
            "PORTFOLIO": portfolio,
            "INDICATOR_CODE": indicator_code,
            "INDICATOR": indicator_name,
            "TYPE": comparison_type,
            "CURRENT_ACTIVE": int(current_flag.sum()),
            "PREVIOUS_ACTIVE": int(previous_flag.sum()),
            "NEW_ALERTS": int(new_alert.sum()),
            "BLOCKED_PREVIOUS_UNAVAILABLE": int(blocked.sum()),
        }
    )

if alert_rows:
    alert_detail = pd.concat(
        alert_rows,
        ignore_index=True,
    )
else:
    alert_detail = pd.DataFrame(
        columns=[
            "ALERT_DATE",
            "PREVIOUS_DATE",
            "ENTITY_ID",
            "PORTFOLIO",
            "INDICATOR_CODE",
            "INDICATOR",
            "COMPARISON_TYPE",
            "GEC_NET_EXPOSURE",
            "HEAD_GEC_PARENT_ID",
        ]
    )

alert_control = pd.DataFrame(
    control_rows
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Category bundling: one rating alert and one CC alert per client

raw_alert_detail = alert_detail.copy()

if not alert_detail.empty:
    alert_detail["BUNDLE_CATEGORY"] = pd.NA
    rating_mask = alert_detail["INDICATOR_CODE"].isin(RATING_CODES)
    cc_mask = alert_detail["INDICATOR_CODE"].isin(CC_UTILISATION_CODES)

    alert_detail.loc[rating_mask, "BUNDLE_CATEGORY"] = "RATING"
    alert_detail.loc[cc_mask, "BUNDLE_CATEGORY"] = "CC_UTILISATION"
    alert_detail["BUNDLE_RANK"] = (
        alert_detail["INDICATOR_CODE"].map(bundle_code_to_rank)
    )

    # When an entity qualifies for RRE and Other Retail, its rating alerts
    # must still be grouped together. Portfolio is NOT part of the group key.
    alert_detail["PORTFOLIO_ORDER"] = (
        alert_detail["PORTFOLIO"].map({"RRE": 0, "OTHER_RETAIL": 1})
    )

    # Keep the old ranking order within each rating family. Across families,
    # prefer the lower rank and use RRE then client rating as tie breakers.
    alert_detail["RATING_FAMILY_ORDER"] = (
        alert_detail["INDICATOR_CODE"]
        .map(bundle_code_to_group)
        .astype("string")
        .str.contains("CLIENT_RATING", na=False)
        .map({True: 0, False: 1})
    )

    non_bundled = alert_detail.loc[
        alert_detail["BUNDLE_CATEGORY"].isna()
    ].copy()

    bundled_candidates = alert_detail.loc[
        alert_detail["BUNDLE_CATEGORY"].notna()
    ].copy()

    if bundled_candidates.empty:
        bundled_selected = bundled_candidates.copy()
    else:
        bundled_candidates["BUNDLE_RANK"] = pd.to_numeric(
            bundled_candidates["BUNDLE_RANK"], errors="raise"
        )
        bundled_selected = (
            bundled_candidates
            .sort_values(
                [
                    "ENTITY_ID",
                    "BUNDLE_CATEGORY",
                    "BUNDLE_RANK",
                    "PORTFOLIO_ORDER",
                    "RATING_FAMILY_ORDER",
                    "INDICATOR_CODE",
                ],
                ascending=True,
                kind="stable",
            )
            .drop_duplicates(
                subset=["ENTITY_ID", "BUNDLE_CATEGORY"], keep="first"
            )
        )

    alert_detail = (
        pd.concat([non_bundled, bundled_selected], ignore_index=True)
        .sort_values(["ENTITY_ID", "PORTFOLIO", "INDICATOR_CODE"])
        .reset_index(drop=True)
    )

    # Ensure neither overlapping portfolios nor rating families create two
    # displayed alerts for a single client and a single category.
    remaining = alert_detail.loc[
        alert_detail["BUNDLE_CATEGORY"].notna()
    ]
    duplicate_categories = int(
        remaining.groupby(["ENTITY_ID", "BUNDLE_CATEGORY"])
        .size().gt(1).sum()
    )
    if duplicate_categories:
        raise ValueError(
            "Category bundling failed: multiple alerts per entity/category."
        )

# Existing Check Head QA retained as an additional safeguard.
if alert_detail.empty:
    check_head_rating_duplicates = 0
else:
    rating_final = alert_detail.loc[
        alert_detail["INDICATOR_CODE"].isin(RATING_CODES)
    ].copy()
    is_head = (
        normalize_id(rating_final["ENTITY_ID"]).notna()
        & normalize_id(rating_final["HEAD_GEC_PARENT_ID"]).notna()
        & normalize_id(rating_final["ENTITY_ID"]).eq(
            normalize_id(rating_final["HEAD_GEC_PARENT_ID"])
        )
    )
    check_head_rating_duplicates = int(
        rating_final.loc[is_head].groupby("ENTITY_ID").size().gt(1).sum()
    )
    if check_head_rating_duplicates:
        raise ValueError("Check Head bundling failed.")


raw_indicator_alert_count = len(raw_alert_detail)
bundled_indicator_alert_count = len(alert_detail)

rating_alerts_suppressed = (
    int(raw_alert_detail["INDICATOR_CODE"].isin(RATING_CODES).sum())
    - int(alert_detail["INDICATOR_CODE"].isin(RATING_CODES).sum())
)
cc_alerts_suppressed = (
    int(raw_alert_detail["INDICATOR_CODE"].isin(CC_UTILISATION_CODES).sum())
    - int(alert_detail["INDICATOR_CODE"].isin(CC_UTILISATION_CODES).sum())
)
assert (
    raw_indicator_alert_count - bundled_indicator_alert_count
    == rating_alerts_suppressed + cc_alerts_suppressed
)

if alert_detail.empty:
    final_counts_by_indicator = {}
else:
    final_counts_by_indicator = (
        alert_detail
        .groupby("INDICATOR_CODE")
        .size()
        .to_dict()
    )

alert_control["FINAL_AFTER_BUNDLING"] = (
    alert_control["INDICATOR_CODE"]
    .map(final_counts_by_indicator)
    .fillna(0)
    .astype(int)
)

if alert_detail.empty:
    daily_alerts = build_empty_output(
        indicator_columns
    )

else:
    alert_detail = alert_detail.drop(
        columns=[
            "BUNDLE_CATEGORY",
            "BUNDLE_RANK",
            "PORTFOLIO_ORDER",
            "RATING_FAMILY_ORDER",
        ],
        errors="ignore",
    )

    grouped = (
        alert_detail
        .groupby(
            "ENTITY_ID",
            as_index=False,
        )
        .agg(
            NEW_INDICATOR_COUNT=(
                "INDICATOR_CODE",
                "nunique",
            ),
            TRIGGERED_INDICATORS=(
                "INDICATOR",
                lambda x: " | ".join(sorted(set(x))),
            ),
            TRIGGERED_INDICATOR_CODES=(
                "INDICATOR_CODE",
                lambda x: " | ".join(sorted(set(x))),
            ),
        )
    )

    current_client = (
        current[
            [
                "ENTITY_ID",
                "RRE_FLAG",
                "OTHER_RETAIL_FLAG",
                "GEC_NET_EXPOSURE",
                "HEAD_GEC_PARENT_ID",
            ]
        ]
        .reset_index(drop=True)
    )

    daily_alerts = grouped.merge(
        current_client,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )

    daily_alerts.insert(
        0,
        "PREVIOUS_DATE",
        PREVIOUS_DATE,
    )

    daily_alerts.insert(
        0,
        "ALERT_DATE",
        REFERENCE_DATE,
    )

    for indicator_code, _, _, _ in INDICATORS:
        alert_column = "ALERT_{}".format(
            indicator_code
        )

        triggered_entities = set(
            alert_detail.loc[
                alert_detail["INDICATOR_CODE"].eq(indicator_code),
                "ENTITY_ID",
            ].astype(str)
        )

        daily_alerts[alert_column] = (
            daily_alerts["ENTITY_ID"]
            .astype(str)
            .isin(triggered_entities)
            .astype("int8")
        )


indicator_alert_count = len(
    alert_detail
)

unique_client_count = int(
    alert_detail["ENTITY_ID"].nunique()
)

multiple_indicator_count = (
    int(
        daily_alerts["NEW_INDICATOR_COUNT"]
        .gt(1)
        .sum()
    )
    if not daily_alerts.empty
    else 0
)

blocked_count = int(
    alert_control[
        "BLOCKED_PREVIOUS_UNAVAILABLE"
    ].sum()
)


print()
print("=" * 110)
print("ALERTS BY INDICATOR")
print("=" * 110)

alert_table = (
    alert_control.loc[
        alert_control["NEW_ALERTS"].gt(0),
        [
            "PORTFOLIO",
            "INDICATOR",
            "NEW_ALERTS",
            "FINAL_AFTER_BUNDLING",
            "BLOCKED_PREVIOUS_UNAVAILABLE",
        ],
    ]
    .sort_values(
        [
            "FINAL_AFTER_BUNDLING",
            "NEW_ALERTS",
            "PORTFOLIO",
            "INDICATOR",
        ],
        ascending=[False, False, True, True],
    )
)

if alert_table.empty:
    print("No new alerts.")
else:
    print(
        alert_table.to_string(
            index=False
        )
    )


print()
print("=" * 110)
print("RATING AND CC CATEGORY BUNDLING QA")
print("=" * 110)

print(
    "Raw indicator-level alerts          : {:,}".format(
        raw_indicator_alert_count
    )
)

print(
    "Final indicator-level alerts        : {:,}".format(
        bundled_indicator_alert_count
    )
)

print(
    "Rating alerts suppressed by bundling: {:,}".format(
        rating_alerts_suppressed
    )
)

print(
    "Check Head groups with >1 final rating/check alert: {:,}".format(
        check_head_rating_duplicates
    )
)

print(
    "CC utilisation alerts suppressed      : {:,}".format(
        cc_alerts_suppressed
    )
)
print(
    "Note: source indicator flags remain independent in Available_Indicators."
)

for bundle_group, ranked_codes in RATING_BUNDLES.items():
    if raw_alert_detail.empty:
        raw_group_count = 0
        final_group_count = 0
    else:
        raw_group_count = int(
            raw_alert_detail["INDICATOR_CODE"]
            .isin(ranked_codes)
            .sum()
        )

        final_group_count = int(
            alert_detail["INDICATOR_CODE"]
            .isin(ranked_codes)
            .sum()
        )

    print(
        "{:<35} raw={:>5,} final={:>5,} suppressed={:>5,}".format(
            bundle_group,
            raw_group_count,
            final_group_count,
            raw_group_count - final_group_count,
        )
    )


print()
print("=" * 110)
print("FINAL DAILY EWS ALERT VOLUME")
print("=" * 110)

print(
    "Indicator-level alerts             : {:,}".format(
        indicator_alert_count
    )
)

print(
    "Unique clients with new alert      : {:,}".format(
        unique_client_count
    )
)

print(
    "Clients with >1 new indicator      : {:,}".format(
        multiple_indicator_count
    )
)

print(
    "Blocked - previous unavailable     : {:,}".format(
        blocked_count
    )
)


# Write output

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    daily_alerts
)

print()

print(
    "{:,} entity-level alerts written to {}.".format(
        len(daily_alerts),
        OUTPUT_DATASET,
    )
)