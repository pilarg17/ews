# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read inputs
# Load the entity-level information and the combined customer payment data.

entity_information_df = dataiku.Dataset(
    "Entity_Information"
).get_dataframe()

entity_level_df = dataiku.Dataset(
    "entity_level_merged"
).get_dataframe()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Align identifiers and dates before matching records across the datasets.

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
        .dt.tz_convert(None)
        .dt.normalize()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise keys
# Match records using the entity, customer and snapshot date together.

for frame in [
    entity_information_df,
    entity_level_df,
]:

    frame["ENTITY_ID"] = normalize_id(
        frame["ENTITY_ID"]
    )

    frame["CUSTOMER_ID"] = normalize_id(
        frame["CUSTOMER_ID"]
    )

    frame["MAP_DTE"] = normalize_date(
        frame["MAP_DTE"]
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep required entity information
# Retain the entity's default status and probability of default alongside its identifiers.

entity_information_df = entity_information_df[
    [
        "ENTITY_ID",
        "CUSTOMER_ID",
        "MAP_DTE",
        "DEFAULT_STATUS",
        "PD_CRR",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep required failed debit information
# Use the daily failed debit count and amount for the matching customer.

entity_level_df = entity_level_df[
    [
        "ENTITY_ID",
        "CUSTOMER_ID",
        "MAP_DTE",
        "FAILED_DEBIT_COUNT",
        "FAILED_DEBIT_AMOUNT",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Merge
# Enrich each entity/customer record with its failed debit activity, if available.

vars_entity_level_merged_df = (
    entity_information_df
    .merge(
        entity_level_df,
        how="left",
        on=[
            "ENTITY_ID",
            "CUSTOMER_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Fill missing failed debit activity
# Customers with no matching payment record have zero failed debits.

for column in [
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
]:

    vars_entity_level_merged_df[column] = (
        pd.to_numeric(
            vars_entity_level_merged_df[column],
            errors="coerce",
        )
        .fillna(0)
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final columns
# Keep the entity attributes and daily payment measures needed downstream.

vars_entity_level_merged_df = (
    vars_entity_level_merged_df[
        [
            "ENTITY_ID",
            "CUSTOMER_ID",
            "MAP_DTE",
            "DEFAULT_STATUS",
            "PD_CRR",
            "FAILED_DEBIT_COUNT",
            "FAILED_DEBIT_AMOUNT",
        ]
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check output coverage and prevent duplicate records for the same entity and customer.

duplicate_rows = int(
    vars_entity_level_merged_df.duplicated(
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
        len(vars_entity_level_merged_df)
    )
)

print(
    "Entities:",
    "{:,}".format(
        vars_entity_level_merged_df[
            "ENTITY_ID"
        ].nunique()
    )
)

print(
    "Columns:",
    vars_entity_level_merged_df.columns.tolist()
)

print(
    "Duplicate entity/customer/day rows:",
    duplicate_rows,
)

if duplicate_rows > 0:
    raise ValueError(
        "Duplicate ENTITY_ID + CUSTOMER_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the entity-level view of customer risk and daily failed debit activity.

dataiku.Dataset(
    "vars_entity_level_merged"
).write_with_schema(
    vars_entity_level_merged_df
)
