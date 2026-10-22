# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


MISSING_LABEL = "<MISSING>"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference dates

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
    )

required_reference_columns = [
    "REFERENCE_DATE",
    "PREVIOUS_DATE",
]

missing_reference_columns = [
    column
    for column in required_reference_columns
    if column not in reference_df.columns
]

if missing_reference_columns:
    raise KeyError(
        "Implementation_Reference_Date is missing: {}".format(
            ", ".join(
                missing_reference_columns
            )
        )
    )


def normalize_scalar_date(value):
    return (
        pd.to_datetime(
            value,
            errors="raise",
            utc=True,
        )
        .tz_localize(None)
        .normalize()
    )


REFERENCE_DATE = normalize_scalar_date(
    reference_df[
        "REFERENCE_DATE"
    ].iloc[0]
)

PREVIOUS_DATE = normalize_scalar_date(
    reference_df[
        "PREVIOUS_DATE"
    ].iloc[0]
)

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)

PREVIOUS_DATE_STR = (
    PREVIOUS_DATE.strftime(
        "%Y-%m-%d"
    )
)

if PREVIOUS_DATE >= REFERENCE_DATE:
    raise ValueError(
        "PREVIOUS_DATE must be earlier than REFERENCE_DATE."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Master scale

MASTER_SCALE = pd.DataFrame([
    ("AAA+", 0.00000, 0.00005, 1),
    ("AAA",  0.00005, 0.00015, 2),
    ("AA+",  0.00015, 0.00025, 3),
    ("AA",   0.00025, 0.00035, 4),
    ("AA-",  0.00035, 0.00045, 5),
    ("A+",   0.00045, 0.00055, 6),
    ("A",    0.00055, 0.00065, 7),
    ("A-",   0.00065, 0.00125, 8),
    ("BBB+", 0.00125, 0.00260, 9),
    ("BBB",  0.00260, 0.00525, 10),
    ("BBB-", 0.00525, 0.00795, 11),
    ("BB+",  0.00795, 0.01015, 12),
    ("BB",   0.01015, 0.01915, 13),
    ("BB-",  0.01915, 0.03315, 14),
    ("B+",   0.03315, 0.06510, 15),
    ("B",    0.06510, 0.11455, 16),
    ("B-",   0.11455, 0.22355, 17),
    ("CCC",  0.22355, 0.99975, 18),
    ("D",    0.99975, 1.00500, 19),
], columns=[
    "RATING",
    "LOWER_BOUND",
    "UPPER_BOUND",
    "NOTCH",
])

MASTER_SCALE_BOUNDS = np.r_[
    MASTER_SCALE[
        "LOWER_BOUND"
    ].to_numpy(),
    MASTER_SCALE[
        "UPPER_BOUND"
    ].iloc[-1],
]

MASTER_SCALE_LABELS = (
    MASTER_SCALE[
        "RATING"
    ].tolist()
)

MASTER_SCALE_NOTCH_MAP = dict(
    zip(
        MASTER_SCALE[
            "RATING"
        ],
        MASTER_SCALE[
            "NOTCH"
        ],
    )
)


def map_pd_to_master_scale(pd_series):
    values = pd.to_numeric(
        pd_series,
        errors="coerce",
    )

    rating = pd.cut(
        values,
        bins=MASTER_SCALE_BOUNDS,
        labels=MASTER_SCALE_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")

    notch = (
        rating
        .map(
            MASTER_SCALE_NOTCH_MAP
        )
        .astype(float)
    )

    return rating, notch


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility

def get_sql_source(dataset_name):
    dataset = dataiku.Dataset(
        dataset_name
    )

    location = (
        dataset.get_location_info()
    )

    info = location.get(
        "info",
        {},
    )

    schema = (
        info.get("schema")
        or info.get("schemaName")
    )

    table = (
        info.get("table")
        or info.get("tableName")
    )

    if not table:
        raise ValueError(
            "{} must be a SQL-backed dataset.".format(
                dataset_name
            )
        )

    if schema and "." not in table:
        table = "{}.{}".format(
            schema,
            table,
        )

    return dataset, table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve source tables

entity_dataset, entity_table = get_sql_source(
    "ENTITY_VW"
)

_, customer_table = get_sql_source(
    "CUSTOMER_VW"
)

_, entity_customer_table = get_sql_source(
    "ENTITY_CUSTOMER_VW_1"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Entity, customer and GCC information

query = """
WITH current_entity_ranked AS (

    SELECT
        e.entity_id,
        e.entity_name,
        e.gcc_parent_id,
        e.gcc_parent_name,
        e.special_mention_flag,
        e.flag_watch_list,
        e.default_status,
        e.litigation_strategy,
        e.economic_activity,
        e.pd_crr,
        e.utp_flag,
        e.flag_utp_insolvency_conciliation_procedure,
        e.probation_period_flag,
        e.probation_entry_date,
        e.probation_end_date,

        ROW_NUMBER() OVER (
            PARTITION BY e.entity_id
            ORDER BY
                e.tek_strt_dte DESC,
                NVL(
                    e.tek_end_dte,
                    DATE '9999-12-31'
                ) DESC
        ) AS rn

    FROM {entity_table} e

    WHERE e.tek_strt_dte < (DATE '{reference_date}' + 1)

      AND NVL(e.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

      AND e.internal_segment LIKE 'RET%'
      AND (
          e.counterparty_subtype = 'NATURAL_PERSON'
          OR SUBSTR(UPPER(TRIM(e.entity_name)), 1, 3) = 'JO_'
      )
      AND e.entity_id IS NOT NULL

),

current_entity AS (

    SELECT
        entity_id,
        entity_name,
        gcc_parent_id,
        gcc_parent_name,
        special_mention_flag,
        flag_watch_list,
        default_status,
        litigation_strategy,
        economic_activity,
        pd_crr,
        utp_flag,
        flag_utp_insolvency_conciliation_procedure,
        probation_period_flag,
        probation_entry_date,
        probation_end_date

    FROM current_entity_ranked

    WHERE rn = 1

),

previous_entity_ranked AS (

    SELECT
        e.entity_id,
        e.gcc_parent_id,
        e.pd_crr,

        ROW_NUMBER() OVER (
            PARTITION BY e.entity_id
            ORDER BY
                e.tek_strt_dte DESC,
                NVL(
                    e.tek_end_dte,
                    DATE '9999-12-31'
                ) DESC
        ) AS rn

    FROM {entity_table} e

    WHERE e.tek_strt_dte < (DATE '{previous_date}' + 1)

      AND NVL(e.tek_end_dte, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)

      AND e.internal_segment LIKE 'RET%'
      AND (
          e.counterparty_subtype = 'NATURAL_PERSON'
          OR SUBSTR(UPPER(TRIM(e.entity_name)), 1, 3) = 'JO_'
      )
      AND e.entity_id IS NOT NULL

),

previous_entity AS (

    SELECT
        entity_id,
        gcc_parent_id,
        pd_crr

    FROM previous_entity_ranked

    WHERE rn = 1

),

standard_customer AS (

    SELECT DISTINCT
        TO_CHAR(
            counterparty_identifier_globe
        ) AS entity_id,

        TO_CHAR(
            customer_id
        ) AS customer_id

    FROM {customer_table}

    WHERE tek_strt_dte < (DATE '{reference_date}' + 1)

      AND NVL(tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

      AND counterparty_identifier_globe
          IS NOT NULL

      AND customer_id
          IS NOT NULL

),

joint_obligor_customer AS (

    SELECT DISTINCT
        TO_CHAR(
            ec.entity_id
        ) AS entity_id,

        TO_CHAR(
            ec.customer_id
        ) AS customer_id

    FROM {entity_customer_table} ec

    INNER JOIN current_entity curr
        ON TO_CHAR(ec.entity_id)
           = TO_CHAR(curr.entity_id)

    WHERE ec.tek_strt_dte < (DATE '{reference_date}' + 1)

      AND NVL(ec.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

      AND SUBSTR(
            UPPER(
                TRIM(
                    curr.entity_name
                )
            ),
            1,
            3
          ) = 'JO_'

      AND ec.entity_id
          IS NOT NULL

      AND ec.customer_id
          IS NOT NULL

),

current_customer AS (

    SELECT
        entity_id,
        customer_id

    FROM standard_customer

    UNION

    SELECT
        entity_id,
        customer_id

    FROM joint_obligor_customer

)

SELECT
    curr.entity_id
        AS ENTITY_ID,

    curr.entity_name
        AS ENTITY_NAME,

    cust.customer_id
        AS CUSTOMER_ID,

    curr.gcc_parent_id
        AS GCC_PARENT_ID,

    curr.gcc_parent_name
        AS GCC_PARENT_NAME,

    prev.gcc_parent_id
        AS GCC_PARENT_ID_PREVIOUS_DAY,

    curr.special_mention_flag
        AS SPECIAL_MENTION_FLAG,

    curr.flag_watch_list
        AS FLAG_WATCH_LIST,

    curr.default_status
        AS DEFAULT_STATUS,

    curr.litigation_strategy
        AS LITIGATION_STRATEGY,

    curr.economic_activity
        AS ECONOMIC_ACTIVITY,

    curr.pd_crr
        AS PD_CRR,

    curr.utp_flag
        AS UTP_FLAG,

    curr.flag_utp_insolvency_conciliation_procedure
        AS FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE,

    curr.probation_period_flag
        AS PROBATION_PERIOD_FLAG,

    curr.probation_entry_date
        AS PROBATION_ENTRY_DATE,

    curr.probation_end_date
        AS PROBATION_END_DATE,

    prev.pd_crr
        AS PD_CRR_PREVIOUS_DAY,

    curr_parent.pd_crr
        AS PD_CRR_GCC_PARENT,

    prev_parent.pd_crr
        AS PD_CRR_GCC_PARENT_PREVIOUS_DAY,

    curr.pd_crr
        - prev.pd_crr
        AS PD_CRR_CHANGE_ABS,

    CASE
        WHEN prev.pd_crr IS NULL
          OR prev.pd_crr = 0
            THEN NULL

        ELSE
            (
                curr.pd_crr
                - prev.pd_crr
            )
            / prev.pd_crr

    END AS PD_CRR_CHANGE_REL,

    DATE '{reference_date}'
        AS MAP_DTE

FROM current_entity curr

INNER JOIN current_customer cust
    ON TO_CHAR(curr.entity_id)
       = cust.entity_id

LEFT JOIN previous_entity prev
    ON curr.entity_id
       = prev.entity_id

LEFT JOIN current_entity curr_parent
    ON curr.gcc_parent_id
       = curr_parent.entity_id

LEFT JOIN previous_entity prev_parent
    ON prev.gcc_parent_id
       = prev_parent.entity_id

""".format(
    entity_table=entity_table,
    customer_table=customer_table,
    entity_customer_table=entity_customer_table,
    reference_date=REFERENCE_DATE_STR,
    previous_date=PREVIOUS_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query

start_time = time.time()

ews_output_df = SQLExecutor2(
    dataset=entity_dataset
).query_to_df(
    query
)

print(
    "Time to perform the query {:.1f} seconds".format(
        time.time()
        - start_time
    )
)

if ews_output_df.empty:
    raise ValueError(
        "Entity query returned no rows for {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise identifiers

for column in [
    "ENTITY_ID",
    "CUSTOMER_ID",
    "GCC_PARENT_ID",
    "GCC_PARENT_ID_PREVIOUS_DAY",
]:
    ews_output_df[column] = (
        ews_output_df[
            column
        ]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise categorical fields

for column in [
    "DEFAULT_STATUS",
    "LITIGATION_STRATEGY",
    "PROBATION_PERIOD_FLAG",
]:
    ews_output_df[column] = (
        ews_output_df[
            column
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise dates

for column in [
    "MAP_DTE",
    "PROBATION_ENTRY_DATE",
    "PROBATION_END_DATE",
]:
    ews_output_df[column] = (
        pd.to_datetime(
            ews_output_df[column],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise numeric fields

numeric_columns = [
    "PD_CRR",
    "PD_CRR_PREVIOUS_DAY",
    "PD_CRR_GCC_PARENT",
    "PD_CRR_GCC_PARENT_PREVIOUS_DAY",
    "PD_CRR_CHANGE_ABS",
    "PD_CRR_CHANGE_REL",
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
]

for column in numeric_columns:
    ews_output_df[column] = (
        pd.to_numeric(
            ews_output_df[
                column
            ],
            errors="coerce",
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### GCC PD


has_current_gcc = (
    ews_output_df[
        "GCC_PARENT_ID"
    ].notna()
)

has_previous_gcc = (
    ews_output_df[
        "GCC_PARENT_ID_PREVIOUS_DAY"
    ].notna()
)

ews_output_df[
    "PD_CRR_GCC"
] = np.where(
    has_current_gcc,
    ews_output_df[
        "PD_CRR_GCC_PARENT"
    ],
    ews_output_df[
        "PD_CRR"
    ],
)

ews_output_df[
    "PD_CRR_GCC_PREVIOUS_DAY"
] = np.where(
    has_previous_gcc,
    ews_output_df[
        "PD_CRR_GCC_PARENT_PREVIOUS_DAY"
    ],
    ews_output_df[
        "PD_CRR_PREVIOUS_DAY"
    ],
)

ews_output_df[
    "PD_CRR_GCC"
] = pd.to_numeric(
    ews_output_df[
        "PD_CRR_GCC"
    ],
    errors="coerce",
)

ews_output_df[
    "PD_CRR_GCC_PREVIOUS_DAY"
] = pd.to_numeric(
    ews_output_df[
        "PD_CRR_GCC_PREVIOUS_DAY"
    ],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Client rating master scale

(
    pd_master_rating,
    pd_master_notch,
) = map_pd_to_master_scale(
    ews_output_df[
        "PD_CRR"
    ]
)

(
    pd_previous_master_rating,
    pd_previous_master_notch,
) = map_pd_to_master_scale(
    ews_output_df[
        "PD_CRR_PREVIOUS_DAY"
    ]
)

ews_output_df[
    "PD_MASTER_RATING"
] = pd_master_rating

ews_output_df[
    "PD_MASTER_NOTCH"
] = pd_master_notch

ews_output_df[
    "PD_MASTER_RATING_PREVIOUS_DAY"
] = pd_previous_master_rating

ews_output_df[
    "PD_MASTER_NOTCH_PREVIOUS_DAY"
] = pd_previous_master_notch


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### GCC rating master scale

(
    gcc_master_rating,
    gcc_master_notch,
) = map_pd_to_master_scale(
    ews_output_df[
        "PD_CRR_GCC"
    ]
)

(
    gcc_previous_master_rating,
    gcc_previous_master_notch,
) = map_pd_to_master_scale(
    ews_output_df[
        "PD_CRR_GCC_PREVIOUS_DAY"
    ]
)

ews_output_df[
    "GCCPD_MASTER_RATING"
] = gcc_master_rating

ews_output_df[
    "GCCPD_MASTER_NOTCH"
] = gcc_master_notch

ews_output_df[
    "GCCPD_MASTER_RATING_PREVIOUS_DAY"
] = gcc_previous_master_rating

ews_output_df[
    "GCCPD_MASTER_NOTCH_PREVIOUS_DAY"
] = gcc_previous_master_notch


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### One-day rating movements

ews_output_df[
    "PD_MASTER_NOTCH_CHANGE_1D"
] = (
    ews_output_df[
        "PD_MASTER_NOTCH"
    ]
    - ews_output_df[
        "PD_MASTER_NOTCH_PREVIOUS_DAY"
    ]
)

ews_output_df[
    "GCCPD_MASTER_NOTCH_CHANGE_1D"
] = (
    ews_output_df[
        "GCCPD_MASTER_NOTCH"
    ]
    - ews_output_df[
        "GCCPD_MASTER_NOTCH_PREVIOUS_DAY"
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA

duplicate_entity_customer_days = int(
    ews_output_df.duplicated(
        subset=[
            "ENTITY_ID",
            "CUSTOMER_ID",
            "MAP_DTE",
        ]
    ).sum()
)

customers_per_entity = (
    ews_output_df
    .groupby(
        "ENTITY_ID"
    )[
        "CUSTOMER_ID"
    ]
    .nunique()
)

multi_customer_entities = (
    customers_per_entity[
        customers_per_entity > 1
    ]
)

# Joint Obligor QA
joint_obligor_mask = (
    ews_output_df[
        "ENTITY_NAME"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.startswith("JO_")
)

joint_obligor_df = (
    ews_output_df.loc[
        joint_obligor_mask
    ]
    .copy()
)

joint_obligor_entities = int(
    joint_obligor_df[
        "ENTITY_ID"
    ].nunique()
)

joint_obligor_customers = int(
    joint_obligor_df[
        "CUSTOMER_ID"
    ].nunique()
)

joint_obligor_without_customer = int(
    joint_obligor_df[
        "CUSTOMER_ID"
    ].isna().sum()
)

joint_obligor_customer_counts = (
    joint_obligor_df
    .groupby(
        "ENTITY_ID"
    )[
        "CUSTOMER_ID"
    ]
    .nunique()
)

joint_obligor_with_multiple_customers = int(
    joint_obligor_customer_counts
    .gt(1)
    .sum()
)

# Entity-level QA uses one row per entity.
entities_df = (
    ews_output_df[
        [
            "ENTITY_ID",
            "PD_CRR",
            "PD_CRR_PREVIOUS_DAY",
            "GCC_PARENT_ID",
            "PD_CRR_GCC",
            "PD_CRR_GCC_PREVIOUS_DAY",
            "PD_MASTER_RATING",
            "PD_MASTER_NOTCH",
            "GCCPD_MASTER_NOTCH",
            "DEFAULT_STATUS",
            "LITIGATION_STRATEGY",
            "UTP_FLAG",
            "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
            "PROBATION_PERIOD_FLAG",
            "PROBATION_ENTRY_DATE",
            "PROBATION_END_DATE",
        ]
    ]
    .drop_duplicates(
        subset=[
            "ENTITY_ID"
        ]
    )
)

entities_with_gcc = int(
    entities_df[
        "GCC_PARENT_ID"
    ].notna().sum()
)

entities_with_gcc_pd = int(
    (
        entities_df[
            "GCC_PARENT_ID"
        ].notna()
        &
        entities_df[
            "PD_CRR_GCC"
        ].notna()
    ).sum()
)

entities_with_previous_pd = int(
    entities_df[
        "PD_CRR_PREVIOUS_DAY"
    ].notna().sum()
)

entities_with_previous_gcc_pd = int(
    entities_df[
        "PD_CRR_GCC_PREVIOUS_DAY"
    ].notna().sum()
)

entities_with_rating = int(
    entities_df[
        "PD_MASTER_NOTCH"
    ].notna().sum()
)

entities_with_gcc_rating = int(
    entities_df[
        "GCCPD_MASTER_NOTCH"
    ].notna().sum()
)

entities_with_default_status = int(
    entities_df[
        "DEFAULT_STATUS"
    ].notna().sum()
)

entities_with_litigation_strategy = int(
    entities_df[
        "LITIGATION_STRATEGY"
    ].notna().sum()
)

entities_with_utp_flag = int(
    entities_df[
        "UTP_FLAG"
    ].notna().sum()
)

entities_with_probation_flag = int(
    entities_df[
        "PROBATION_PERIOD_FLAG"
    ].notna().sum()
)

entities_with_probation_entry_date = int(
    entities_df[
        "PROBATION_ENTRY_DATE"
    ].notna().sum()
)

entities_with_probation_end_date = int(
    entities_df[
        "PROBATION_END_DATE"
    ].notna().sum()
)

# Operational exclusion flags are used for QA only. No rows are removed.

entities_pd_100 = int(
    np.isclose(
        pd.to_numeric(
            entities_df[
                "PD_CRR"
            ],
            errors="coerce",
        ),
        1.0,
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ).sum()
)

entities_rating_d = int(
    entities_df[
        "PD_MASTER_RATING"
    ]
    .astype("string")
    .str.upper()
    .eq("D")
    .sum()
)

entities_liquidation = int(
    entities_df[
        "LITIGATION_STRATEGY"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
    .isin(
        [
            "LIQ",
            "LIQUIDATION",
        ]
    )
    .sum()
)


print()
print("ENTITY INFORMATION QA")
print("---------------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Previous date:",
    PREVIOUS_DATE_STR,
)

print(
    "Rows:",
    "{:,}".format(
        len(
            ews_output_df
        )
    ),
)

print(
    "Entities:",
    "{:,}".format(
        ews_output_df[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Customers:",
    "{:,}".format(
        ews_output_df[
            "CUSTOMER_ID"
        ].nunique()
    ),
)

print(
    "Entities with previous-day PD:",
    "{:,}".format(
        entities_with_previous_pd
    ),
)

print(
    "Entities with GCC parent:",
    "{:,}".format(
        entities_with_gcc
    ),
)

print(
    "Entities with available GCC PD:",
    "{:,}".format(
        entities_with_gcc_pd
    ),
)

print(
    "Entities with previous GCC PD:",
    "{:,}".format(
        entities_with_previous_gcc_pd
    ),
)

print(
    "Entities with mapped client rating:",
    "{:,}".format(
        entities_with_rating
    ),
)

print(
    "Entities with mapped GCC rating:",
    "{:,}".format(
        entities_with_gcc_rating
    ),
)

print(
    "Duplicated ENTITY_ID + CUSTOMER_ID + MAP_DTE:",
    duplicate_entity_customer_days,
)

print(
    "Entities linked to multiple customers:",
    "{:,}".format(
        len(
            multi_customer_entities
        )
    ),
)

if len(
    multi_customer_entities
) > 0:

    print(
        "Maximum customers linked to one entity:",
        int(
            multi_customer_entities.max()
        ),
    )



print()
print("JOINT OBLIGOR QA")
print("-----------------")

print(
    "Joint Obligor rows:",
    "{:,}".format(
        len(
            joint_obligor_df
        )
    ),
)

print(
    "Joint Obligor entities:",
    "{:,}".format(
        joint_obligor_entities
    ),
)

print(
    "Joint Obligor customers:",
    "{:,}".format(
        joint_obligor_customers
    ),
)

print(
    "Joint Obligors with >1 CUSTOMER_ID:",
    "{:,}".format(
        joint_obligor_with_multiple_customers
    ),
)

print(
    "Joint Obligor rows without CUSTOMER_ID:",
    "{:,}".format(
        joint_obligor_without_customer
    ),
)

if not joint_obligor_df.empty:
    print()
    print(
        joint_obligor_df[
            [
                "ENTITY_ID",
                "ENTITY_NAME",
                "CUSTOMER_ID",
                "GCC_PARENT_ID",
                "MAP_DTE",
            ]
        ]
        .head(50)
        .to_string(
            index=False
        )
    )


print()
print("UTP / PROBATION QA")
print("------------------")

print(
    "Entities with UTP_FLAG populated:",
    "{:,}".format(
        entities_with_utp_flag
    ),
)

print(
    "Entities with PROBATION_PERIOD_FLAG populated:",
    "{:,}".format(
        entities_with_probation_flag
    ),
)

print(
    "Entities with PROBATION_ENTRY_DATE populated:",
    "{:,}".format(
        entities_with_probation_entry_date
    ),
)

print(
    "Entities with PROBATION_END_DATE populated:",
    "{:,}".format(
        entities_with_probation_end_date
    ),
)

print()
print("UTP_FLAG DISTRIBUTION")
print("---------------------")
print(
    entities_df[
        "UTP_FLAG"
    ]
    .fillna(MISSING_LABEL)
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()
print("UTP INSOLVENCY FLAG DISTRIBUTION")
print("--------------------------------")
print(
    entities_df[
        "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE"
    ]
    .fillna(MISSING_LABEL)
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()
print("PROBATION PERIOD FLAG DISTRIBUTION")
print("----------------------------------")
print(
    entities_df[
        "PROBATION_PERIOD_FLAG"
    ]
    .fillna(MISSING_LABEL)
    .value_counts(
        dropna=False
    )
    .to_string()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Operational exclusion QA

print()
print("OPERATIONAL EXCLUSION QA")
print("------------------------")

print(
    "Entities with DEFAULT_STATUS populated:",
    "{:,}".format(
        entities_with_default_status
    ),
)

print(
    "Entities with LITIGATION_STRATEGY populated:",
    "{:,}".format(
        entities_with_litigation_strategy
    ),
)

print(
    "Entities with PD = 100%:",
    "{:,}".format(
        entities_pd_100
    ),
)

print(
    "Entities mapped to rating D:",
    "{:,}".format(
        entities_rating_d
    ),
)

print(
    "Entities with liquidation strategy (LIQ):",
    "{:,}".format(
        entities_liquidation
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA distributions

print()
print("DEFAULT_STATUS DISTRIBUTION")
print("---------------------------")

default_status_distribution = (
    entities_df[
        "DEFAULT_STATUS"
    ]
    .fillna(MISSING_LABEL)
    .value_counts(
        dropna=False
    )
)

print(
    default_status_distribution.to_string()
)


print()
print("LITIGATION_STRATEGY DISTRIBUTION")
print("--------------------------------")

litigation_strategy_distribution = (
    entities_df[
        "LITIGATION_STRATEGY"
    ]
    .fillna(MISSING_LABEL)
    .value_counts(
        dropna=False
    )
)

print(
    litigation_strategy_distribution.to_string()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate QA

if duplicate_entity_customer_days > 0:
    raise ValueError(
        "Entity_Information contains duplicate "
        "ENTITY_ID + CUSTOMER_ID + MAP_DTE rows."
    )

if joint_obligor_without_customer > 0:
    raise ValueError(
        "Joint Obligor rows without CUSTOMER_ID remain in Entity_Information."
    )

if joint_obligor_entities > 0 and joint_obligor_with_multiple_customers == 0:
    raise ValueError(
        "Joint Obligors were found, but none has multiple CUSTOMER_ID mappings."
    )

if entities_with_rating == 0:
    raise ValueError(
        "No client ratings could be mapped to the master scale."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output

_entity_write_started = time.perf_counter()

dataiku.Dataset(
    "Entity_Information"
).write_with_schema(
    ews_output_df
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

print("Entity information write seconds:", round(time.perf_counter() - _entity_write_started, 1))
