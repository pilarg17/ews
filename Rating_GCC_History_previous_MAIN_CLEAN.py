# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# Read the previous branch reference date used to anchor the historical rating
# comparison.

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
    )

if "REFERENCE_DATE" not in reference_df.columns:
    raise KeyError(
        "Implementation_Reference_Date is missing REFERENCE_DATE."
    )

REFERENCE_DATE = (
    pd.to_datetime(
        reference_df[
            "REFERENCE_DATE"
        ].iloc[0],
        errors="raise",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Master scale
# Keep the established PD-to-rating thresholds and their notch positions. Higher notch
# numbers indicate worse grades.

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
# Resolve the underlying SQL tables through the Dataiku source datasets.

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
### Resolve SQL sources
# Use entity history for client and GCC parent PDs, and customer records to establish the
# reference population.

entity_dataset, entity_table = get_sql_source(
    "ENTITY_VW"
)

_, customer_table = get_sql_source(
    "CUSTOMER_VW"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Historical monthly snapshots
# For each eligible Retail entity, retrieve twelve monthly PD observations and the
# corresponding GCC parent PD where available. Summarise the exact 1-, 3-, 6- and
# 12-month values and the best/worst available history.
#

query = """
WITH monthly_dates AS (

    SELECT
        LEVEL AS LAG_MONTH,
        ADD_MONTHS(
            DATE '{reference_date}',
            -LEVEL
        ) AS SNAPSHOT_DATE

    FROM dual

    CONNECT BY LEVEL <= 12

),

current_population_ranked AS (

    SELECT
        e.entity_id,

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

    WHERE TRUNC(
    e.tek_strt_dte
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            e.tek_end_dte,
            DATE '9999-12-31'
        )
    )

      AND e.internal_segment LIKE 'RET%'
      AND e.counterparty_subtype = 'NATURAL_PERSON'
      AND e.entity_id IS NOT NULL

      AND EXISTS (

          SELECT 1

          FROM {customer_table} c

          WHERE TRIM(
              CAST(
                  c.counterparty_identifier_globe
                  AS VARCHAR2(100)
              )
          )
          =
          TRIM(
              CAST(
                  e.entity_id
                  AS VARCHAR2(100)
              )
          )

            AND TRUNC(
    c.tek_strt_dte
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            c.tek_end_dte,
            DATE '9999-12-31'
        )
    )

            AND c.customer_id IS NOT NULL

      )

),

current_population AS (

    SELECT
        entity_id

    FROM current_population_ranked

    WHERE rn = 1

),

client_snapshot_raw AS (

    SELECT
        cp.entity_id,
        md.lag_month,
        md.snapshot_date,

        h.pd_crr,
        h.gcc_parent_id,

        ROW_NUMBER() OVER (
            PARTITION BY
                cp.entity_id,
                md.lag_month
            ORDER BY
                h.tek_strt_dte DESC,
                NVL(
                    h.tek_end_dte,
                    DATE '9999-12-31'
                ) DESC
        ) AS rn

    FROM current_population cp

    CROSS JOIN monthly_dates md

    LEFT JOIN {entity_table} h
        ON h.entity_id
           = cp.entity_id

       AND TRUNC(
    h.tek_strt_dte
) <= md.snapshot_date

AND md.snapshot_date
    < TRUNC(
        NVL(
            h.tek_end_dte,
            DATE '9999-12-31'
        )
    )

),

client_snapshot AS (

    SELECT
        entity_id,
        lag_month,
        snapshot_date,
        pd_crr,
        gcc_parent_id

    FROM client_snapshot_raw

    WHERE rn = 1

),

gcc_snapshot_raw AS (

    SELECT
        cs.entity_id,
        cs.lag_month,
        cs.snapshot_date,
        cs.pd_crr,
        cs.gcc_parent_id,

        p.pd_crr
            AS gcc_parent_pd_crr,

        ROW_NUMBER() OVER (
            PARTITION BY
                cs.entity_id,
                cs.lag_month
            ORDER BY
                p.tek_strt_dte DESC,
                NVL(
                    p.tek_end_dte,
                    DATE '9999-12-31'
                ) DESC
        ) AS rn

    FROM client_snapshot cs

    LEFT JOIN {entity_table} p
        ON cs.gcc_parent_id
           = p.entity_id

       AND TRUNC(
    p.tek_strt_dte
) <= cs.snapshot_date

AND cs.snapshot_date
    < TRUNC(
        NVL(
            p.tek_end_dte,
            DATE '9999-12-31'
        )
    )

),

monthly_history AS (

    SELECT
        entity_id,
        lag_month,
        snapshot_date,

        pd_crr,

        CASE
            WHEN gcc_parent_id IS NOT NULL
                THEN gcc_parent_pd_crr
            ELSE pd_crr
        END AS gcc_pd_crr

    FROM gcc_snapshot_raw

    WHERE rn = 1

)

SELECT
    entity_id AS ENTITY_ID,

    MAX(
        CASE
            WHEN lag_month = 1
                THEN pd_crr
        END
    ) AS PD_CRR_1M,

    MAX(
        CASE
            WHEN lag_month = 3
                THEN pd_crr
        END
    ) AS PD_CRR_3M,

    MAX(
        CASE
            WHEN lag_month = 6
                THEN pd_crr
        END
    ) AS PD_CRR_6M,

    MAX(
        CASE
            WHEN lag_month = 12
                THEN pd_crr
        END
    ) AS PD_CRR_12M,

    MIN(pd_crr)
        AS MIN_PD_CRR_12M,

    MAX(pd_crr)
        AS MAX_PD_CRR_12M,

    COUNT(pd_crr)
        AS CLIENT_HISTORY_MONTHS_AVAILABLE,

    MAX(
        CASE
            WHEN lag_month = 1
                THEN gcc_pd_crr
        END
    ) AS GCC_PD_CRR_1M,

    MAX(
        CASE
            WHEN lag_month = 3
                THEN gcc_pd_crr
        END
    ) AS GCC_PD_CRR_3M,

    MAX(
        CASE
            WHEN lag_month = 6
                THEN gcc_pd_crr
        END
    ) AS GCC_PD_CRR_6M,

    MAX(
        CASE
            WHEN lag_month = 12
                THEN gcc_pd_crr
        END
    ) AS GCC_PD_CRR_12M,

    MIN(gcc_pd_crr)
        AS MIN_GCC_PD_CRR_12M,

    MAX(gcc_pd_crr)
        AS MAX_GCC_PD_CRR_12M,

    COUNT(gcc_pd_crr)
        AS GCC_HISTORY_MONTHS_AVAILABLE

FROM monthly_history

GROUP BY
    entity_id

""".format(
    reference_date=REFERENCE_DATE_STR,
    entity_table=entity_table,
    customer_table=customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute history query
# Query the monthly snapshots and require at least one record to continue.

start_time = time.time()

history_df = SQLExecutor2(
    dataset=entity_dataset
).query_to_df(
    query
)

print(
    "Time to perform history query: {:.1f} seconds".format(
        time.time()
        - start_time
    )
)

if history_df.empty:
    raise ValueError(
        "Rating/GCC history query returned no rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current entity information
# Load the entity information for this branch, which provides the reference-day PDs and
# rating notches.

entity_information_df = dataiku.Dataset(
    "Entity_Information_Previous"
).get_dataframe()

required_current_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "PD_CRR",
    "PD_CRR_GCC",
    "PD_MASTER_RATING",
    "PD_MASTER_NOTCH",
    "GCCPD_MASTER_RATING",
    "GCCPD_MASTER_NOTCH",
]

missing_columns = [
    column
    for column in required_current_columns
    if column not in entity_information_df.columns
]

if missing_columns:
    raise KeyError(
        "Entity_Information_Previous is missing required columns: {}".format(
            ", ".join(
                missing_columns
            )
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise current entity data
# Normalise identifiers and dates; retain records belonging to the branch reference date.

entity_information_df[
    "ENTITY_ID"
] = (
    entity_information_df[
        "ENTITY_ID"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)

entity_information_df[
    "MAP_DTE"
] = (
    pd.to_datetime(
        entity_information_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)

entity_information_df = (
    entity_information_df[
        entity_information_df[
            "MAP_DTE"
        ] == REFERENCE_DATE
    ]
    .copy()
)

if entity_information_df.empty:
    raise ValueError(
        "Entity_Information_Previous has no rows for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate entity-level consistency
# Check that customers mapped to the same entity do not disagree on its rating measures.

current_consistency = (
    entity_information_df
    .groupby(
        "ENTITY_ID"
    )
    .agg(
        PD_VALUES=(
            "PD_CRR",
            "nunique",
        ),
        CLIENT_NOTCH_VALUES=(
            "PD_MASTER_NOTCH",
            "nunique",
        ),
        GCC_NOTCH_VALUES=(
            "GCCPD_MASTER_NOTCH",
            "nunique",
        ),
    )
)

inconsistent_entities = current_consistency[
    (current_consistency["PD_VALUES"] > 1)
    | (
        current_consistency[
            "CLIENT_NOTCH_VALUES"
        ] > 1
    )
    | (
        current_consistency[
            "GCC_NOTCH_VALUES"
        ] > 1
    )
]

if not inconsistent_entities.empty:
    raise ValueError(
        "{} entities have inconsistent current "
        "Rating/GCC values across customers.".format(
            len(
                inconsistent_entities
            )
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep one current row per entity
# Reduce the reference-day input to one rating record per entity.

current_df = (
    entity_information_df[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "PD_CRR",
            "PD_CRR_GCC",
            "PD_MASTER_RATING",
            "PD_MASTER_NOTCH",
            "GCCPD_MASTER_RATING",
            "GCCPD_MASTER_NOTCH",
        ]
    ]
    .sort_values(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .drop_duplicates(
        subset=[
            "ENTITY_ID",
        ],
        keep="last",
    )
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise history identifiers
# Make historical entity keys consistent with the current entity data before joining.

history_df[
    "ENTITY_ID"
] = (
    history_df[
        "ENTITY_ID"
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
### Merge current and historical information
# Keep the reference-day entity population while adding its past rating observations.

rating_history_df = current_df.merge(
    history_df,
    how="left",
    on="ENTITY_ID",
    validate="1:1",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Map historical client PDs to master scale
# Convert historical client PDs into notch numbers using the same defined master scale.

client_pd_columns = {
    "PD_CRR_1M":
        "RATING_GRANULAR_SCALE_1M",

    "PD_CRR_3M":
        "RATING_GRANULAR_SCALE_3M",

    "PD_CRR_6M":
        "RATING_GRANULAR_SCALE_6M",

    "PD_CRR_12M":
        "RATING_GRANULAR_SCALE_12M",

    "MIN_PD_CRR_12M":
        "MIN_RATING_GRANULAR_SCALE_12M",

    "MAX_PD_CRR_12M":
        "MAX_RATING_GRANULAR_SCALE_12M",
}

for pd_column, notch_column in client_pd_columns.items():

    _, notch = map_pd_to_master_scale(
        rating_history_df[
            pd_column
        ]
    )

    rating_history_df[
        notch_column
    ] = notch


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Map historical GCC PDs to master scale
# Apply the master scale to historical GCC PDs for consistent group-rating comparisons.

gcc_pd_columns = {
    "GCC_PD_CRR_1M":
        "GCC_RATING_GRANULAR_SCALE_1M",

    "GCC_PD_CRR_3M":
        "GCC_RATING_GRANULAR_SCALE_3M",

    "GCC_PD_CRR_6M":
        "GCC_RATING_GRANULAR_SCALE_6M",

    "GCC_PD_CRR_12M":
        "GCC_RATING_GRANULAR_SCALE_12M",

    "MIN_GCC_PD_CRR_12M":
        "MIN_GCC_RATING_GRANULAR_SCALE_12M",

    "MAX_GCC_PD_CRR_12M":
        "MAX_GCC_RATING_GRANULAR_SCALE_12M",
}

for pd_column, notch_column in gcc_pd_columns.items():

    _, notch = map_pd_to_master_scale(
        rating_history_df[
            pd_column
        ]
    )

    rating_history_df[
        notch_column
    ] = notch


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current model-facing scales
# Expose the current client and GCC rating notches with the output field names.

rating_history_df[
    "RATING_GRANULAR_SCALE"
] = pd.to_numeric(
    rating_history_df[
        "PD_MASTER_NOTCH"
    ],
    errors="coerce",
)

rating_history_df[
    "GCC_RATING_GRANULAR_SCALE"
] = pd.to_numeric(
    rating_history_df[
        "GCCPD_MASTER_NOTCH"
    ],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Client rating downgrades
# Calculate changes in client rating notches versus historical months and best/worst
# 12-month levels. Positive differences mean deterioration.

for lag in [
    1,
    3,
    6,
    12,
]:

    rating_history_df[
        "rating_granular_scale_downgrade_{}m".format(
            lag
        )
    ] = (
        rating_history_df[
            "RATING_GRANULAR_SCALE"
        ]
        -
        rating_history_df[
            "RATING_GRANULAR_SCALE_{}M".format(
                lag
            )
        ]
    )


rating_history_df[
    "rating_granular_scale_downgrade_best_12m"
] = (
    rating_history_df[
        "RATING_GRANULAR_SCALE"
    ]
    -
    rating_history_df[
        "MIN_RATING_GRANULAR_SCALE_12M"
    ]
)


rating_history_df[
    "rating_granular_scale_downgrade_worst_12m"
] = (
    rating_history_df[
        "RATING_GRANULAR_SCALE"
    ]
    -
    rating_history_df[
        "MAX_RATING_GRANULAR_SCALE_12M"
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### GCC rating downgrades
# Calculate comparable rating changes for the GCC parent or group measure.

for lag in [
    1,
    3,
    6,
    12,
]:

    rating_history_df[
        "gcc_rating_granular_scale_downgrade_{}m".format(
            lag
        )
    ] = (
        rating_history_df[
            "GCC_RATING_GRANULAR_SCALE"
        ]
        -
        rating_history_df[
            "GCC_RATING_GRANULAR_SCALE_{}M".format(
                lag
            )
        ]
    )


rating_history_df[
    "gcc_rating_granular_scale_downgrade_best_12m"
] = (
    rating_history_df[
        "GCC_RATING_GRANULAR_SCALE"
    ]
    -
    rating_history_df[
        "MIN_GCC_RATING_GRANULAR_SCALE_12M"
    ]
)


rating_history_df[
    "gcc_rating_granular_scale_downgrade_worst_12m"
] = (
    rating_history_df[
        "GCC_RATING_GRANULAR_SCALE"
    ]
    -
    rating_history_df[
        "MAX_GCC_RATING_GRANULAR_SCALE_12M"
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Summarise historical coverage and check for duplicate entities.

duplicate_entity_rows = int(
    rating_history_df.duplicated(
        subset=[
            "ENTITY_ID",
        ]
    ).sum()
)

client_12m_available = int(
    rating_history_df[
        "RATING_GRANULAR_SCALE_12M"
    ].notna().sum()
)

gcc_12m_available = int(
    rating_history_df[
        "GCC_RATING_GRANULAR_SCALE_12M"
    ].notna().sum()
)

client_best_available = int(
    rating_history_df[
        "MIN_RATING_GRANULAR_SCALE_12M"
    ].notna().sum()
)

gcc_best_available = int(
    rating_history_df[
        "MIN_GCC_RATING_GRANULAR_SCALE_12M"
    ].notna().sum()
)

full_client_history = int(
    (
        pd.to_numeric(
            rating_history_df[
                "CLIENT_HISTORY_MONTHS_AVAILABLE"
            ],
            errors="coerce",
        ) == 12
    ).sum()
)

full_gcc_history = int(
    (
        pd.to_numeric(
            rating_history_df[
                "GCC_HISTORY_MONTHS_AVAILABLE"
            ],
            errors="coerce",
        ) == 12
    ).sum()
)


print()
print("RATING / GCC HISTORY QA")
print("-----------------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Rows:",
    "{:,}".format(
        len(
            rating_history_df
        )
    ),
)

print(
    "Entities:",
    "{:,}".format(
        rating_history_df[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Entities with exact client rating 12m:",
    "{:,}".format(
        client_12m_available
    ),
)

print(
    "Entities with exact GCC rating 12m:",
    "{:,}".format(
        gcc_12m_available
    ),
)

print(
    "Entities with client best-12m history:",
    "{:,}".format(
        client_best_available
    ),
)

print(
    "Entities with GCC best-12m history:",
    "{:,}".format(
        gcc_best_available
    ),
)

print(
    "Entities with all 12 client monthly observations:",
    "{:,}".format(
        full_client_history
    ),
)

print(
    "Entities with all 12 GCC monthly observations:",
    "{:,}".format(
        full_gcc_history
    ),
)

print(
    "Duplicate ENTITY_ID rows:",
    duplicate_entity_rows,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Rating movement QA
# Count how many entities meet the reported notch deterioration thresholds; these are
# diagnostics, not the final daily alert rules.

client_downgrade_12m = pd.to_numeric(
    rating_history_df[
        "rating_granular_scale_downgrade_12m"
    ],
    errors="coerce",
)

gcc_downgrade_12m = pd.to_numeric(
    rating_history_df[
        "gcc_rating_granular_scale_downgrade_12m"
    ],
    errors="coerce",
)

client_best_downgrade = pd.to_numeric(
    rating_history_df[
        "rating_granular_scale_downgrade_best_12m"
    ],
    errors="coerce",
)

gcc_best_downgrade = pd.to_numeric(
    rating_history_df[
        "gcc_rating_granular_scale_downgrade_best_12m"
    ],
    errors="coerce",
)


print()
print("RATING MOVEMENT QA")
print("------------------")

for threshold in [
    1,
    2,
    3,
    4,
    5,
    6,
]:

    print(
        "Client downgrade >= {} vs 12m: {:,}".format(
            threshold,
            int(
                (
                    client_downgrade_12m
                    >= threshold
                ).sum()
            ),
        )
    )

print()

for threshold in [
    1,
    2,
    3,
    4,
    5,
]:

    print(
        "GCC downgrade >= {} vs 12m: {:,}".format(
            threshold,
            int(
                (
                    gcc_downgrade_12m
                    >= threshold
                ).sum()
            ),
        )
    )

print()

print(
    "Client downgrade >= 6 vs best 12m:",
    "{:,}".format(
        int(
            (
                client_best_downgrade
                >= 6
            ).sum()
        )
    ),
)

print(
    "GCC downgrade >= 1 vs best 12m:",
    "{:,}".format(
        int(
            (
                gcc_best_downgrade
                >= 1
            ).sum()
        )
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate QA
# Keep one output row per reference-date entity and stop if the population no longer
# reconciles.

if duplicate_entity_rows > 0:
    raise ValueError(
        "Rating_GCC_History contains duplicate ENTITY_ID rows."
    )

if len(
    rating_history_df
) != current_df[
    "ENTITY_ID"
].nunique():

    raise ValueError(
        "Rating_GCC_History does not preserve "
        "the Entity_Information_Previous population."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep implementation fields
# Select the model-facing history, notch and downgrade fields.

output_columns = [
    "ENTITY_ID",
    "MAP_DTE",

    "RATING_GRANULAR_SCALE",
    "GCC_RATING_GRANULAR_SCALE",

    "RATING_GRANULAR_SCALE_1M",
    "RATING_GRANULAR_SCALE_3M",
    "RATING_GRANULAR_SCALE_6M",
    "RATING_GRANULAR_SCALE_12M",

    "MIN_RATING_GRANULAR_SCALE_12M",
    "MAX_RATING_GRANULAR_SCALE_12M",

    "GCC_RATING_GRANULAR_SCALE_1M",
    "GCC_RATING_GRANULAR_SCALE_3M",
    "GCC_RATING_GRANULAR_SCALE_6M",
    "GCC_RATING_GRANULAR_SCALE_12M",

    "MIN_GCC_RATING_GRANULAR_SCALE_12M",
    "MAX_GCC_RATING_GRANULAR_SCALE_12M",

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

    "CLIENT_HISTORY_MONTHS_AVAILABLE",
    "GCC_HISTORY_MONTHS_AVAILABLE",
]

rating_history_output_df = (
    rating_history_df[
        output_columns
    ]
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the previous-branch historical ratings for downstream indicators.

dataiku.Dataset(
    "Rating_GCC_History_previous"
).write_with_schema(
    rating_history_output_df
)