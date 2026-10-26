# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read inputs
# Load the customer details and failed direct debit information for the same daily snapshot.

customer_df = dataiku.Dataset(
    "vars_customer_level_Previous"
).get_dataframe()

payment_df = dataiku.Dataset(
    "vars_payment_level_Previous"
).get_dataframe()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Standardise IDs and dates so customer records can be matched reliably.

def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
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


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise keys
# Use the same customer identifiers and dates in both datasets before joining them.

customer_df["ENTITY_ID"] = normalize_id(
    customer_df["ENTITY_ID"]
)

customer_df["CUSTOMER_ID"] = normalize_id(
    customer_df["CUSTOMER_ID"]
)

payment_df["CUSTOMER_ID"] = normalize_id(
    payment_df["CUSTOMER_ID"]
)

customer_df["MAP_DTE"] = normalize_date(
    customer_df["MAP_DTE"]
)

payment_df["MAP_DTE"] = normalize_date(
    payment_df["MAP_DTE"]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Merge customer and payment information
# Add payment activity to each customer, retaining customers with no matching payment records.

entity_level_merged_df = (
    customer_df
    .merge(
        payment_df,
        how="left",
        on=[
            "CUSTOMER_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Fill customers without failed debits
# Treat missing payment activity as zero failed direct debits on that day.

for column in [
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
]:
    entity_level_merged_df[column] = (
        pd.to_numeric(
            entity_level_merged_df[column],
            errors="coerce",
        )
        .fillna(0)
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check that each entity/customer combination appears only once for the daily snapshot.

duplicate_rows = int(
    entity_level_merged_df.duplicated(
        subset=[
            "ENTITY_ID",
            "CUSTOMER_ID",
            "MAP_DTE",
        ]
    ).sum()
)

print(
    "Rows:",
    "{:,}".format(
        len(entity_level_merged_df)
    )
)

print(
    "Duplicate entity/customer/day:",
    duplicate_rows,
)

if duplicate_rows > 0:
    raise ValueError(
        "entity_level_merged_Previous contains duplicate "
        "ENTITY_ID + CUSTOMER_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the combined customer and payment data for the next entity-level step.

dataiku.Dataset(
    "entity_level_merged_Previous"
).write_with_schema(
    entity_level_merged_df
)
