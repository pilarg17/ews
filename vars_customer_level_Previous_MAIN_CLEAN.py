# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Use the selected snapshot date to retrieve customer details for the previous daily comparison.

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
    pd.to_datetime(
        reference_date_df["REFERENCE_DATE"],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
    .dropna()
    .drop_duplicates()
)


if len(reference_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain "
        "exactly one valid REFERENCE_DATE."
    )


REFERENCE_DATE = reference_dates.iloc[0]

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Find the SQL tables behind the customer datasets.

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
### Resolve source tables
# Use customer records and the individual customer details valid on the reference date.

customer_dataset, customer_table = (
    get_sql_source(
        "CUSTOMER_VW"
    )
)


individual_dataset, individual_table = (
    get_sql_source(
        "INDIVIDUAL_CUSTOMER_VW"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily customer-level data
# Link customers to their entities and retrieve household and personal information.

query = """
SELECT DISTINCT
    c.COUNTERPARTY_IDENTIFIER_GLOBE
        AS ENTITY_ID,

    c.CUSTOMER_ID
        AS CUSTOMER_ID,

    ic.MARITAL_STATUS
        AS MARITAL_STATUS,

    ic.NUMBER_OF_CHILDREN
        AS NUMBER_OF_CHILDREN,

    ic.HOME_OWNERSHIP
        AS HOME_OWNERSHIP,

    ic.BIRTH_DATE
        AS BIRTH_DATE,

    DATE '{reference_date}'
        AS MAP_DTE

FROM {customer_table} c

LEFT JOIN {individual_table} ic

    ON c.CUSTOMER_SK = ic.CUSTOMER_SK

    AND TRUNC(
    ic.TEK_STRT_DTE
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            ic.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

WHERE TRUNC(
    c.TEK_STRT_DTE
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            c.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

  AND c.COUNTERPARTY_IDENTIFIER_GLOBE IS NOT NULL
  AND c.CUSTOMER_ID IS NOT NULL

""".format(
    customer_table=customer_table,
    individual_table=individual_table,
    reference_date=REFERENCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Retrieve the customer information from the source database.

start_time = time.time()


vars_customer_level_df = SQLExecutor2(
    dataset=customer_dataset
).query_to_df(
    query
)


print(
    "Time to perform the query {:.1f} seconds".format(
        time.time() - start_time
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise output
# Align IDs, dates and the number of children for downstream use.

vars_customer_level_df[
    "ENTITY_ID"
] = (
    vars_customer_level_df[
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


vars_customer_level_df[
    "CUSTOMER_ID"
] = (
    vars_customer_level_df[
        "CUSTOMER_ID"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)


vars_customer_level_df[
    "MAP_DTE"
] = (
    pd.to_datetime(
        vars_customer_level_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)


vars_customer_level_df[
    "BIRTH_DATE"
] = (
    pd.to_datetime(
        vars_customer_level_df[
            "BIRTH_DATE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
)


vars_customer_level_df[
    "NUMBER_OF_CHILDREN"
] = pd.to_numeric(
    vars_customer_level_df[
        "NUMBER_OF_CHILDREN"
    ],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check how many customers belong to each entity and whether any daily records repeat.

duplicate_customer_days = int(
    vars_customer_level_df
    .duplicated(
        subset=[
            "ENTITY_ID",
            "CUSTOMER_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)


customers_per_entity = (
    vars_customer_level_df
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


n_dates = int(
    vars_customer_level_df[
        "MAP_DTE"
    ]
    .nunique()
)


min_date = (
    vars_customer_level_df[
        "MAP_DTE"
    ]
    .min()
)


max_date = (
    vars_customer_level_df[
        "MAP_DTE"
    ]
    .max()
)


print()
print("VARS_CUSTOMER_LEVEL PREVIOUS QA")
print("----------------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Rows:",
    "{:,}".format(
        len(
            vars_customer_level_df
        )
    ),
)

print(
    "Entities:",
    "{:,}".format(
        vars_customer_level_df[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Customers:",
    "{:,}".format(
        vars_customer_level_df[
            "CUSTOMER_ID"
        ].nunique()
    ),
)

print(
    "Entities linked to multiple customers:",
    "{:,}".format(
        len(
            multi_customer_entities
        )
    ),
)

print(
    "Distinct MAP_DTE:",
    n_dates,
)

print(
    "Minimum MAP_DTE:",
    min_date,
)

print(
    "Maximum MAP_DTE:",
    max_date,
)

print(
    "Duplicated ENTITY_ID + CUSTOMER_ID + MAP_DTE:",
    duplicate_customer_days,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Stop if the dataset is empty, contains duplicate customer records or has the wrong date.

if vars_customer_level_df.empty:
    raise ValueError(
        "vars_customer_level_Previous is empty."
    )


if duplicate_customer_days > 0:
    raise ValueError(
        "vars_customer_level_Previous contains duplicate "
        "ENTITY_ID + CUSTOMER_ID + MAP_DTE rows."
    )


if n_dates != 1:
    raise ValueError(
        "vars_customer_level_Previous should contain exactly "
        "one MAP_DTE."
    )


if (
    pd.Timestamp(
        max_date
    ).normalize()
    != REFERENCE_DATE.normalize()
):
    raise ValueError(
        "vars_customer_level_Previous output date does not "
        "match the implementation reference date."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the daily customer details for the rest of the flow.

dataiku.Dataset(
    "vars_customer_level_Previous"
).write_with_schema(
    vars_customer_level_df
)
