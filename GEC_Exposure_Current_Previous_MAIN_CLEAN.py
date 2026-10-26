# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Set the GEC exposure dataset and the ranges shown in the final checks.

OUTPUT_DATASET = "GEC_Exposure_Current_Previous"

GEC_THRESHOLDS = [
    50000.0,
    100000.0,
    250000.0,
    500000.0,
    800000.0,
    900000.0,
    1000000.0,
    1500000.0,
]

EUR_FORMAT = "EUR {:,.2f}"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Prepare consistent entity IDs and dates, and locate the SQL tables configured in Dataiku.

def normalize_id(series):

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
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


def get_sql_source(dataset_name):

    dataset = dataiku.Dataset(
        dataset_name
    )

    location = dataset.get_location_info()

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

    if (
        schema
        and "." not in table
    ):
        table = "{}.{}".format(
            schema,
            table,
        )

    return dataset, table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Use the reference date for this snapshot of group exposure.

reference_date_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()


if reference_date_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )


if "REFERENCE_DATE" not in reference_date_df.columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing REFERENCE_DATE."
    )


reference_dates = (
    normalize_date(
        reference_date_df[
            "REFERENCE_DATE"
        ]
    )
    .dropna()
    .drop_duplicates()
)


if len(reference_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain "
        "exactly one valid REFERENCE_DATE."
    )


REFERENCE_DATE = (
    reference_dates.iloc[0]
)

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


print(
    "Reference date:",
    REFERENCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve ENTITY_VW
# Find the entity records used to identify each GEC head.

entity_dataset, entity_table = (
    get_sql_source(
        "ENTITY_VW"
    )
)


print(
    "Entity source:",
    entity_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Retail Natural Person GEC mapping
# For each Retail natural person, identify the GEC head recorded on the reference date.

gec_mapping_query = """
WITH retail_entity_raw AS
(
    SELECT
        e.ENTITY_ID,
        e.HEAD_GEC_PARENT_ID,

        ROW_NUMBER() OVER
        (
            PARTITION BY
                e.ENTITY_ID

            ORDER BY
                e.TEK_STRT_DTE DESC,

                NVL(
                    e.TEK_END_DTE,
                    DATE '9999-12-31'
                ) DESC
        ) AS RN

    FROM {entity_table} e

    WHERE
        TRUNC(
            e.TEK_STRT_DTE
        ) <= DATE '{reference_date}'

        AND DATE '{reference_date}'
            < TRUNC(
                NVL(
                    e.TEK_END_DTE,
                    DATE '9999-12-31'
                )
            )

        AND e.INTERNAL_SEGMENT
            LIKE 'RET%'

        AND e.COUNTERPARTY_SUBTYPE
            = 'NATURAL_PERSON'

        AND e.ENTITY_ID
            IS NOT NULL
)

SELECT
    ENTITY_ID,
    HEAD_GEC_PARENT_ID

FROM retail_entity_raw

WHERE RN = 1
""".format(
    entity_table=entity_table,
    reference_date=REFERENCE_DATE_STR,
)


start_time = time.time()


gec_mapping = SQLExecutor2(
    dataset=entity_dataset
).query_to_df(
    gec_mapping_query
)


mapping_query_elapsed = (
    time.time()
    - start_time
)


print(
    "Time to perform GEC mapping query: "
    "{:.1f} seconds".format(
        mapping_query_elapsed
    )
)


if gec_mapping.empty:
    raise ValueError(
        "Retail GEC mapping returned 0 rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise GEC mapping
# Keep one GEC mapping per entity and reject duplicate entity records.

required_mapping_columns = [
    "ENTITY_ID",
    "HEAD_GEC_PARENT_ID",
]


missing_mapping_columns = [
    column
    for column in required_mapping_columns
    if column not in gec_mapping.columns
]


if missing_mapping_columns:
    raise KeyError(
        "ENTITY_VW GEC mapping is missing required "
        "column(s): {}".format(
            ", ".join(
                missing_mapping_columns
            )
        )
    )


gec_mapping[
    "ENTITY_ID"
] = normalize_id(
    gec_mapping[
        "ENTITY_ID"
    ]
)


gec_mapping[
    "HEAD_GEC_PARENT_ID"
] = normalize_id(
    gec_mapping[
        "HEAD_GEC_PARENT_ID"
    ]
)


gec_mapping = (
    gec_mapping
    .dropna(
        subset=[
            "ENTITY_ID",
        ]
    )
    .copy()
)


mapping_duplicate_entities = int(
    gec_mapping
    .duplicated(
        subset=[
            "ENTITY_ID",
        ],
        keep=False,
    )
    .sum()
)


if mapping_duplicate_entities > 0:
    raise ValueError(
        "GEC mapping contains duplicate ENTITY_ID rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read current loan exposure
# Load the loan exposure corresponding to this reference date.

loan = dataiku.Dataset(
    "vars_loan_Previous"
).get_dataframe()


required_loan_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "NET_EXPOSURE",
]


missing_loan_columns = [
    column
    for column in required_loan_columns
    if column not in loan.columns
]


if missing_loan_columns:
    raise KeyError(
        "vars_loan is missing required column(s): {}".format(
            ", ".join(
                missing_loan_columns
            )
        )
    )


loan = (
    loan[
        required_loan_columns
    ]
    .copy()
)


loan[
    "ENTITY_ID"
] = normalize_id(
    loan[
        "ENTITY_ID"
    ]
)


loan[
    "MAP_DTE"
] = normalize_date(
    loan[
        "MAP_DTE"
    ]
)


loan[
    "NET_EXPOSURE"
] = pd.to_numeric(
    loan[
        "NET_EXPOSURE"
    ],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate vars_loan date
# Make sure the loan exposure belongs to the selected snapshot.

loan_dates = (
    loan[
        "MAP_DTE"
    ]
    .dropna()
    .drop_duplicates()
    .sort_values()
)


if len(loan_dates) != 1:
    raise ValueError(
        "vars_loan must contain exactly one MAP_DTE "
        "for the daily implementation. Found {}.".format(
            len(
                loan_dates
            )
        )
    )


loan_date = (
    loan_dates.iloc[0]
)


if (
    loan_date.normalize()
    != REFERENCE_DATE.normalize()
):
    raise ValueError(
        "vars_loan MAP_DTE {} does not match "
        "REFERENCE_DATE {}.".format(
            loan_date.strftime(
                "%Y-%m-%d"
            ),
            REFERENCE_DATE_STR,
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Aggregate exposure by Retail entity
# Add up net exposure from the individual loans of each entity.

loan_entity = (
    loan
    .dropna(
        subset=[
            "ENTITY_ID",
        ]
    )
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        NET_EXPOSURE=(
            "NET_EXPOSURE",
            "sum",
        )
    )
)


loan_entity_duplicates = int(
    loan_entity
    .duplicated(
        subset=[
            "ENTITY_ID",
        ]
    )
    .sum()
)


if loan_entity_duplicates > 0:
    raise ValueError(
        "Aggregated vars_loan contains duplicate ENTITY_ID rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Join Retail scope to exposure
# Link the entity-level exposure with its GEC membership, using zero when no loan exposure is available.

retail_exposure = (
    gec_mapping
    .merge(
        loan_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
)


retail_exposure[
    "NET_EXPOSURE"
] = (
    retail_exposure[
        "NET_EXPOSURE"
    ]
    .fillna(
        0.0
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Aggregate mapped exposure by GEC
# Calculate the total net exposure and Retail member count for each GEC.

mapped_retail_exposure = (
    retail_exposure
    .dropna(
        subset=[
            "HEAD_GEC_PARENT_ID",
        ]
    )
    .copy()
)


gec_totals = (
    mapped_retail_exposure
    .groupby(
        "HEAD_GEC_PARENT_ID",
        as_index=False,
    )
    .agg(
        GEC_NET_EXPOSURE=(
            "NET_EXPOSURE",
            "sum",
        ),
        GEC_RETAIL_MEMBER_COUNT=(
            "ENTITY_ID",
            "nunique",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Build entity-level output
# Attach the GEC totals to each mapped entity, while leaving unmapped group fields empty.

output = (
    gec_mapping
    .merge(
        gec_totals,
        how="left",
        on="HEAD_GEC_PARENT_ID",
        validate="m:1",
    )
)


mapped_mask = (
    output[
        "HEAD_GEC_PARENT_ID"
    ]
    .notna()
)


output.loc[
    mapped_mask,
    "GEC_NET_EXPOSURE",
] = (
    output.loc[
        mapped_mask,
        "GEC_NET_EXPOSURE",
    ]
    .fillna(
        0.0
    )
)


output.loc[
    mapped_mask,
    "GEC_RETAIL_MEMBER_COUNT",
] = (
    output.loc[
        mapped_mask,
        "GEC_RETAIL_MEMBER_COUNT",
    ]
    .fillna(
        0
    )
)


output.loc[
    ~mapped_mask,
    [
        "GEC_NET_EXPOSURE",
        "GEC_RETAIL_MEMBER_COUNT",
    ],
] = pd.NA


output[
    "MAP_DTE"
] = REFERENCE_DATE


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA - Mapping
# Count entities with and without a GEC head.

retail_entities = int(
    len(
        gec_mapping
    )
)


mapped_entities = int(
    gec_mapping[
        "HEAD_GEC_PARENT_ID"
    ]
    .notna()
    .sum()
)


unmapped_entities = int(
    gec_mapping[
        "HEAD_GEC_PARENT_ID"
    ]
    .isna()
    .sum()
)


n_gecs = int(
    gec_mapping[
        "HEAD_GEC_PARENT_ID"
    ]
    .nunique(
        dropna=True
    )
)


print()
print("=" * 100)
print("RETAIL GEC MAPPING QA")
print("=" * 100)

print(
    "Reference date                   :",
    REFERENCE_DATE_STR,
)

print(
    "Retail entities                  :",
    "{:,}".format(
        retail_entities
    ),
)

print(
    "With HEAD_GEC_PARENT_ID          :",
    "{:,}".format(
        mapped_entities
    ),
)

print(
    "Without HEAD_GEC_PARENT_ID       :",
    "{:,}".format(
        unmapped_entities
    ),
)

print(
    "Distinct GECs                    :",
    "{:,}".format(
        n_gecs
    ),
)

print(
    "Duplicate ENTITY_ID rows         :",
    mapping_duplicate_entities,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA - Exposure coverage
# Check that mapped and unmapped exposure totals reconcile.

total_vars_loan_exposure = float(
    loan[
        "NET_EXPOSURE"
    ]
    .fillna(
        0.0
    )
    .sum()
)


total_retail_exposure = float(
    retail_exposure[
        "NET_EXPOSURE"
    ]
    .sum()
)


mapped_exposure = float(
    mapped_retail_exposure[
        "NET_EXPOSURE"
    ]
    .sum()
)


unmapped_exposure = float(
    retail_exposure.loc[
        retail_exposure[
            "HEAD_GEC_PARENT_ID"
        ].isna(),
        "NET_EXPOSURE",
    ]
    .sum()
)


if total_retail_exposure != 0:

    mapped_pct = (
        mapped_exposure
        / total_retail_exposure
        * 100
    )

    unmapped_pct = (
        unmapped_exposure
        / total_retail_exposure
        * 100
    )

else:

    mapped_pct = 0.0
    unmapped_pct = 0.0


print()
print("=" * 100)
print("GEC EXPOSURE COVERAGE QA")
print("=" * 100)

print(
    "vars_loan total NET_EXPOSURE     :",
    EUR_FORMAT.format(
        total_vars_loan_exposure
    ),
)

print(
    "Retail NET_EXPOSURE              :",
    EUR_FORMAT.format(
        total_retail_exposure
    ),
)

print(
    "Mapped to GEC                    :",
    EUR_FORMAT.format(
        mapped_exposure
    ),
)

print(
    "Unmapped Retail exposure         :",
    EUR_FORMAT.format(
        unmapped_exposure
    ),
)

print(
    "Mapped exposure %                :",
    "{:.2f}%".format(
        mapped_pct
    ),
)

print(
    "Unmapped exposure %              :",
    "{:.2f}%".format(
        unmapped_pct
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA - Threshold distribution
# Show how many groups and members exceed the selected exposure thresholds.

print()
print("=" * 110)
print("GEC EXPOSURE THRESHOLD DISTRIBUTION")
print("=" * 110)

print(
    "{:<15} {:>15} {:>22}".format(
        "THRESHOLD",
        "GEC_COUNT",
        "RETAIL_MEMBERS",
    )
)


for threshold in GEC_THRESHOLDS:

    eligible_gecs = (
        gec_totals.loc[
            gec_totals[
                "GEC_NET_EXPOSURE"
            ].gt(
                threshold
            )
        ]
    )

    gec_count = int(
        len(
            eligible_gecs
        )
    )

    retail_members = int(
        eligible_gecs[
            "GEC_RETAIL_MEMBER_COUNT"
        ]
        .sum()
    )

    print(
        "{:<15,.0f} {:>15,} {:>22,}".format(
            threshold,
            gec_count,
            retail_members,
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA - Largest GECs
# Check the largest group exposures.

print()
print("=" * 100)
print("TOP GEC EXPOSURES")
print("=" * 100)

top_gecs = (
    gec_totals
    .sort_values(
        "GEC_NET_EXPOSURE",
        ascending=False,
    )
    .head(
        10
    )
)


print(
    top_gecs.to_string(
        index=False
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final structural validations
# Check the expected number of entities, unique dates and exposure reconciliation.

duplicate_output_rows = int(
    output
    .duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)


if duplicate_output_rows > 0:
    raise ValueError(
        "GEC_Exposure_Current_Previous contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


if len(output) != retail_entities:
    raise ValueError(
        "GEC_Exposure_Current_Previous does not preserve "
        "the Retail Natural Person population."
    )


if (
    mapped_entities
    + unmapped_entities
    != retail_entities
):
    raise ValueError(
        "Mapped + unmapped GEC population does not "
        "reconcile with Retail population."
    )


exposure_reconciliation_difference = abs(
    total_retail_exposure
    - mapped_exposure
    - unmapped_exposure
)


if exposure_reconciliation_difference > 0.01:
    raise ValueError(
        "Retail exposure does not reconcile between "
        "mapped and unmapped populations. Difference: "
        "{:.2f}".format(
            exposure_reconciliation_difference
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final clean schema
# Keep the group head, group exposure and member count required by the indicators.

output[
    "GEC_RETAIL_MEMBER_COUNT"
] = pd.to_numeric(
    output[
        "GEC_RETAIL_MEMBER_COUNT"
    ],
    errors="coerce",
)


output[
    "GEC_NET_EXPOSURE"
] = pd.to_numeric(
    output[
        "GEC_NET_EXPOSURE"
    ],
    errors="coerce",
)


output = (
    output[
        [
            "ENTITY_ID",
            "HEAD_GEC_PARENT_ID",
            "GEC_NET_EXPOSURE",
            "GEC_RETAIL_MEMBER_COUNT",
            "MAP_DTE",
        ]
    ]
    .sort_values(
        [
            "ENTITY_ID",
        ]
    )
    .reset_index(
        drop=True
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the GEC exposure information for this snapshot.

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    output
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final log
# Report the size of the output and the time spent retrieving the mapping.

print()
print("=" * 100)
print("GEC EXPOSURE CURRENT - COMPLETED")
print("=" * 100)

print(
    "Output dataset :",
    OUTPUT_DATASET,
)

print(
    "Snapshot date  :",
    REFERENCE_DATE_STR,
)

print(
    "Rows written   :",
    "{:,}".format(
        len(
            output
        )
    ),
)

print(
    "Mapped entities:",
    "{:,}".format(
        mapped_entities
    ),
)

print(
    "Distinct GECs  :",
    "{:,}".format(
        n_gecs
    ),
)

print(
    "Mapping query  : {:.1f}s".format(
        mapping_query_elapsed
    )
)