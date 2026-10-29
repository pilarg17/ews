# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

# Reference date and source tables
reference = dataiku.Dataset("Implementation_Reference_Date").get_dataframe()
if len(reference) != 1 or "REFERENCE_DATE" not in reference.columns:
    raise ValueError("Implementation_Reference_Date must have one row with REFERENCE_DATE.")

reference_date = pd.to_datetime(
    reference["REFERENCE_DATE"].iloc[0], errors="raise"
).strftime("%Y-%m-%d")


def sql_table(dataset_name):
    dataset = dataiku.Dataset(dataset_name)
    info = dataset.get_location_info().get("info", {})
    table = info.get("table") or info.get("tableName")
    schema = info.get("schema") or info.get("schemaName")
    if not table:
        raise ValueError("SQL source not found for {}".format(dataset_name))
    if schema and "." not in table:
        table = "{}.{}".format(schema, table)
    return dataset, table


def query_df(dataset, query):
    return SQLExecutor2(dataset=dataset).query_to_df(query)


entity_dataset, entity_table = sql_table("ENTITY_VW")
instrument_dataset, instrument_table = sql_table("INSTRUMENT_ENTITY_VW")
customer_dataset, customer_table = sql_table("ENTITY_CUSTOMER_VW_1")
account_dataset, account_table = sql_table("PLY_T24_ACCOUNT_VW")

population_sql = """
WITH entity_ranked AS (
    SELECT
        e.ENTITY_ID,
        e.INTERNAL_SEGMENT,
        e.SECTOR,
        ROW_NUMBER() OVER (
            PARTITION BY e.ENTITY_ID
            ORDER BY e.TEK_STRT_DTE DESC,
                     NVL(e.TEK_END_DTE, DATE '9999-12-31') DESC
        ) AS RN
    FROM {entity_table} e
    WHERE e.TEK_STRT_DTE < DATE '{date}' + 1
      AND NVL(e.TEK_END_DTE, DATE '9999-12-31') >= DATE '{date}' + 1
      AND e.COUNTERPARTY_SUBTYPE = 'NATURAL_PERSON'
      AND e.ENTITY_ID IS NOT NULL
),
population AS (
    SELECT ENTITY_ID, INTERNAL_SEGMENT, SECTOR
    FROM entity_ranked
    WHERE RN = 1
),
oth AS (
    SELECT ENTITY_ID, SECTOR
    FROM population
    WHERE INTERNAL_SEGMENT = 'OTH_01_01'
)
""".format(entity_table=entity_table, date=reference_date)

# Population
population_result = query_df(entity_dataset, population_sql + """
SELECT
    CASE
        WHEN INTERNAL_SEGMENT = 'OTH_01_01' THEN 'OTH_01_01'
        WHEN INTERNAL_SEGMENT LIKE 'RET%' THEN 'RETAIL'
    END AS POPULATION,
    COUNT(*) AS NATURAL_PERSONS,
    SUM(CASE WHEN TRIM(TO_CHAR(SECTOR)) = '1002' THEN 1 ELSE 0 END)
        AS BIL_EMPLOYEES
FROM population
WHERE INTERNAL_SEGMENT = 'OTH_01_01'
   OR INTERNAL_SEGMENT LIKE 'RET%'
GROUP BY CASE
    WHEN INTERNAL_SEGMENT = 'OTH_01_01' THEN 'OTH_01_01'
    WHEN INTERNAL_SEGMENT LIKE 'RET%' THEN 'RETAIL'
END
ORDER BY POPULATION
""")

# Instrument relationships (historical and valid on reference date)
instrument_result = query_df(instrument_dataset, population_sql + """
SELECT
    COUNT(DISTINCT o.ENTITY_ID) AS OTH_CLIENTS,
    COUNT(DISTINCT CASE WHEN i.INSTRUMENT_ID IS NOT NULL
        THEN o.ENTITY_ID END) AS CLIENTS_WITH_ANY_INSTRUMENT,
    COUNT(DISTINCT CASE WHEN i.INSTRUMENT_ID IS NOT NULL
         AND i.TEK_STRT_DTE < DATE '{date}' + 1
         AND NVL(i.TEK_END_DTE, DATE '9999-12-31') >= DATE '{date}' + 1
        THEN o.ENTITY_ID END) AS CLIENTS_WITH_VALID_INSTRUMENT,
    COUNT(DISTINCT CASE WHEN i.INSTRUMENT_ID IS NOT NULL
         AND i.TEK_STRT_DTE < DATE '{date}' + 1
         AND NVL(i.TEK_END_DTE, DATE '9999-12-31') >= DATE '{date}' + 1
        THEN i.INSTRUMENT_ID END) AS VALID_INSTRUMENTS
FROM oth o
LEFT JOIN {instrument_table} i ON i.ENTITY_ID = o.ENTITY_ID
""".format(date=reference_date, instrument_table=instrument_table))

# Entity-to-customer relationships
customer_result = query_df(customer_dataset, population_sql + """
SELECT
    COUNT(DISTINCT o.ENTITY_ID) AS OTH_CLIENTS,
    COUNT(DISTINCT CASE WHEN ec.CUSTOMER_ID IS NOT NULL
        THEN o.ENTITY_ID END) AS CLIENTS_WITH_ANY_CUSTOMER_LINK,
    COUNT(DISTINCT CASE WHEN ec.CUSTOMER_ID IS NOT NULL
         AND TRUNC(ec.TEK_STRT_DTE) <= DATE '{date}'
         AND DATE '{date}' < TRUNC(NVL(ec.TEK_END_DTE, DATE '9999-12-31'))
        THEN o.ENTITY_ID END) AS CLIENTS_WITH_VALID_CUSTOMER_LINK,
    COUNT(DISTINCT CASE WHEN ec.CUSTOMER_ID IS NOT NULL
         AND TRUNC(ec.TEK_STRT_DTE) <= DATE '{date}'
         AND DATE '{date}' < TRUNC(NVL(ec.TEK_END_DTE, DATE '9999-12-31'))
        THEN ec.CUSTOMER_ID END) AS VALID_CUSTOMER_IDS
FROM oth o
LEFT JOIN {customer_table} ec ON ec.ENTITY_ID = o.ENTITY_ID
""".format(date=reference_date, customer_table=customer_table))

# Currency is needed before using T24 balances as EUR-based proxies.
try:
    account_fields = {
        item["name"].upper(): item["name"]
        for item in account_dataset.read_schema()
    }
except Exception:
    account_fields = {}

currency_field = next(
    (account_fields[field] for field in ["CURRENCY", "ACCOUNT_CURRENCY", "CCY"]
     if field in account_fields),
    None,
)

euro_condition = (
    "UPPER(TRIM(CAST(acc.{} AS VARCHAR2(30)))) IN ('EUR', '978')".format(currency_field)
    if currency_field else "1 = 0"
)

# Current account balances from T24, using the same mapping as Customer_Balances.
account_result = query_df(account_dataset, population_sql + """
, valid_customers AS (
    SELECT DISTINCT
        ec.ENTITY_ID,
        TRIM(CAST(ec.CUSTOMER_ID AS VARCHAR2(100))) AS CUSTOMER_ID
    FROM {customer_table} ec
    INNER JOIN oth o ON o.ENTITY_ID = ec.ENTITY_ID
    WHERE TRUNC(ec.TEK_STRT_DTE) <= DATE '{date}'
      AND DATE '{date}' < TRUNC(NVL(ec.TEK_END_DTE, DATE '9999-12-31'))
      AND ec.CUSTOMER_ID IS NOT NULL
),
accounts_by_entity AS (
    SELECT
        o.ENTITY_ID,
        COUNT(acc.CUSTOMER) AS ACCOUNT_ROWS,
        MAX(CASE WHEN acc.CUSTOMER IS NOT NULL THEN 1 ELSE 0 END) AS HAS_ACCOUNT,
        MAX(CASE WHEN TRIM(TO_CHAR(acc.CATEGORY)) = '1001'
                 THEN 1 ELSE 0 END) AS HAS_CURRENT_ACCOUNT,
        MAX(CASE WHEN TRIM(TO_CHAR(acc.CATEGORY)) = '1001'
                   AND acc.WORKING_BALANCE < 0
                 THEN 1 ELSE 0 END) AS HAS_NEGATIVE_CURRENT_ACCOUNT,
        MAX(CASE WHEN acc.WORKING_BALANCE < 0
                 THEN 1 ELSE 0 END) AS HAS_ANY_NEGATIVE_BALANCE,
        SUM(CASE WHEN TRIM(TO_CHAR(acc.CATEGORY)) = '1001'
                   AND acc.WORKING_BALANCE < 0
                   AND {euro_condition}
                 THEN ABS(acc.WORKING_BALANCE) ELSE 0 END)
                 AS EUR_CURRENT_ACCOUNT_OVERDRAFT_PROXY
    FROM oth o
    LEFT JOIN valid_customers ec ON o.ENTITY_ID = ec.ENTITY_ID
    LEFT JOIN {account_table} acc
        ON TRIM(CAST(acc.CUSTOMER AS VARCHAR2(100))) = ec.CUSTOMER_ID
       AND TRUNC(acc.TEK_STRT_DTE) <= DATE '{date}'
       AND DATE '{date}' < TRUNC(NVL(acc.TEK_END_DTE, DATE '9999-12-31'))
    GROUP BY o.ENTITY_ID
)
SELECT
    COUNT(*) AS OTH_CLIENTS,
    SUM(HAS_ACCOUNT) AS CLIENTS_WITH_T24_ACCOUNTS,
    SUM(HAS_CURRENT_ACCOUNT) AS CLIENTS_WITH_CURRENT_ACCOUNT,
    SUM(HAS_NEGATIVE_CURRENT_ACCOUNT) AS CLIENTS_WITH_NEGATIVE_CURRENT_ACCOUNT,
    SUM(HAS_ANY_NEGATIVE_BALANCE) AS CLIENTS_WITH_ANY_NEGATIVE_BALANCE,
    SUM(ACCOUNT_ROWS) AS ACTIVE_ACCOUNT_SOURCE_ROWS,
    SUM(CASE WHEN EUR_CURRENT_ACCOUNT_OVERDRAFT_PROXY > 500
             THEN 1 ELSE 0 END) AS CLIENTS_WITH_EUR_CURRENT_ACCOUNT_OVERDRAFT_GT_500,
    SUM(CASE WHEN EUR_CURRENT_ACCOUNT_OVERDRAFT_PROXY >= 5000
             THEN 1 ELSE 0 END) AS CLIENTS_WITH_EUR_CURRENT_ACCOUNT_OVERDRAFT_GE_5000
FROM accounts_by_entity
""".format(
    customer_table=customer_table,
    account_table=account_table,
    date=reference_date,
    euro_condition=euro_condition,
))

# Results
if not currency_field:
    account_result[
        [
            "CLIENTS_WITH_EUR_CURRENT_ACCOUNT_OVERDRAFT_GT_500",
            "CLIENTS_WITH_EUR_CURRENT_ACCOUNT_OVERDRAFT_GE_5000",
        ]
    ] = pd.NA


def print_counts(title, frame, label_column=None):
    print()
    print(title)
    print("-" * len(title))

    if frame.empty:
        print("No rows")
        return

    for _, row in frame.iterrows():
        if label_column:
            print(str(row[label_column]))
        for column in frame.columns:
            if column == label_column:
                continue
            value = row[column]
            if pd.isna(value):
                value_text = "N/A"
            elif isinstance(value, (int, float)) or pd.api.types.is_number(value):
                value_text = "{:,}".format(int(value)) if float(value).is_integer() else "{:,.2f}".format(float(value))
            else:
                value_text = str(value)
            print("{}: {}".format(column, value_text))
        print()


print("OTH_01_01 - Retail scope analysis")
print("Reference date:", reference_date)

print_counts("Population", population_result, "POPULATION")
print_counts("Instrument relationships", instrument_result)
print_counts("Customer relationships", customer_result)
print_counts("T24 accounts", account_result)

print("T24 currency field:", currency_field if currency_field else "Not available")

valid_instruments = int(instrument_result.iloc[0]["CLIENTS_WITH_VALID_INSTRUMENT"])
print("Valid OTH instrument links:", valid_instruments)
print("Full EWS alert simulation: Not calculated")
