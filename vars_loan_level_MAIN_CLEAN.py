# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Set the upper limit used to filter out invalid source values.

DUMMY_HIGH_VALUE_CUTOFF = 1e10


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Use the selected date to retrieve the valid loan records.

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date"
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

REFERENCE_DATE_STR = REFERENCE_DATE.strftime(
    "%Y-%m-%d"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Find the underlying SQL tables without hardcoding their location.

def get_sql_source(dataset_name):
    dataset = dataiku.Dataset(dataset_name)

    location = dataset.get_location_info()
    info = location.get("info", {})

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
# Use the instrument and entity links available in Credit Insight.

instrument_dataset, instrument_table = get_sql_source(
    "INSTRUMENT_VW"
)

instrument_entity_dataset, instrument_entity_table = (
    get_sql_source(
        "INSTRUMENT_ENTITY_VW"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily loan-level data
# Bring together loan details, repayment status, product type and linked entities.

query = """
SELECT DISTINCT
    ins.INSTRUMENT_ID AS INSTRUMENT_ID,
    ins.CONTRACT_ID AS CONTRACT_ID,

    SUBSTR(
        ins.CONTRACT_ID,
        INSTR(ins.CONTRACT_ID, '_211') + 4
    ) AS CONTRACT_ID_SHORT,

    CASE
        WHEN ABS(ins.DPD) >= {dummy_cutoff}
            THEN 0
        ELSE ins.DPD
    END AS DPD,

    ins.PAST_DUE_DATE AS PAST_DUE_DATE,

    CASE
        WHEN ABS(ins.ARREARS_AMOUNT) >= {dummy_cutoff}
            THEN 0
        ELSE ins.ARREARS_AMOUNT
    END AS ARREARS_AMOUNT,

    ins.DEFAULT_STATUS AS DEFAULT_STATUS,
    ins.FORBEARANCE_STATUS AS FORBEARANCE_STATUS,

    ins.LEGAL_FINAL_MATURITY_DATE
        AS LEGAL_FINAL_MATURITY_DATE,

    ins.ORIGINAL_MATURITY_DATE
        AS ORIGINAL_MATURITY_DATE,

    ins.BRIDGE_LOAN_AMOUNT_LCY
        AS BRIDGE_LOAN_AMOUNT_LCY,

    ins.OUTSTANDING_NOMINAL_AMOUNT
        AS NOMINAL_AMOUNT,

    ins.CONTRACT_INTERNAL_PURPOSE_CODE
        AS CONTRACT_INTERNAL_PURPOSE_CODE,

    ins.INSTRUMENT_TYPE
        AS INSTRUMENT_TYPE,

    i2e.ENTITY_ID AS ENTITY_ID,

    DATE '{reference_date}' AS MAP_DTE

FROM {instrument_table} ins

INNER JOIN {instrument_entity_table} i2e
    ON TRIM(
        CAST(
            ins.INSTRUMENT_ID AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            i2e.INSTRUMENT_ID AS VARCHAR2(100)
        )
    )

WHERE TRUNC(ins.TEK_STRT_DTE) <= DATE '{reference_date}'

  AND DATE '{reference_date}'
      < TRUNC(
          NVL(
              ins.TEK_END_DTE,
              DATE '9999-12-31'
          )
      )

  AND TRUNC(i2e.TEK_STRT_DTE) <= DATE '{reference_date}'

  AND DATE '{reference_date}'
      < TRUNC(
          NVL(
              i2e.TEK_END_DTE,
              DATE '9999-12-31'
          )
      )

  AND i2e.ENTITY_ID IS NOT NULL

""".format(
    instrument_table=instrument_table,
    instrument_entity_table=instrument_entity_table,
    reference_date=REFERENCE_DATE_STR,
    dummy_cutoff=DUMMY_HIGH_VALUE_CUTOFF,
)


start_time = time.time()

vars_loan_level_df = SQLExecutor2(
    dataset=instrument_dataset
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
# Align identifiers, dates and amounts before using the loan data.

for column in [
    "INSTRUMENT_ID",
    "ENTITY_ID",
    "INSTRUMENT_TYPE",
]:
    vars_loan_level_df[column] = (
        vars_loan_level_df[column]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

vars_loan_level_df["MAP_DTE"] = pd.to_datetime(
    vars_loan_level_df["MAP_DTE"],
    errors="coerce",
).dt.normalize()

for column in [
    "PAST_DUE_DATE",
    "LEGAL_FINAL_MATURITY_DATE",
    "ORIGINAL_MATURITY_DATE",
]:
    vars_loan_level_df[column] = pd.to_datetime(
        vars_loan_level_df[column],
        errors="coerce",
    )

for column in [
    "DPD",
    "ARREARS_AMOUNT",
    "BRIDGE_LOAN_AMOUNT_LCY",
    "NOMINAL_AMOUNT",
]:
    vars_loan_level_df[column] = pd.to_numeric(
        vars_loan_level_df[column],
        errors="coerce",
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Loan purpose
# Translate the source purpose codes into readable descriptions.

purpose_mapping = {
    11: "11 MEANS",
    12: "12 OTHER MEANS OF TRANSPORT",
    14: "14 NEW ELECTRIC CAR",
    15: "15 NEW NON-ELECTRICAL CAR",
    16: "16 SECOND-HAND ELECTRIC CAR",
    17: "17 SECOND-HAND NON-ELECTRIC CAR",
    21: "21 HOUSEHOLD EQUIPMENT",
    41: "41 MISCELLANEOUS NEEDS",
    42: "42 STUDIES FINANCING",
    43: "43 MEDICAL EXPENSES",
    44: "44 DENTAL SURGERY",
    45: "45 EYE SURGERY",
    46: "46 POST COLLEGE STUDIES FINANCING",
    47: "47 RENTAL GUARANTEE FINANCING",
    101: "101 LAND ACQUISITION PRIVATE USE",
    111: "111 HOUSE CONSTRUCTION OWNER OCCUPIER",
    112: "112 HOUSE ACQUISITION OWNER OCCUPIER",
    113: "113 OTHER PROPERTY PERSONAL USE",
    114: "114 APARTMENT CONSTRUCTION OWNER OCCUPIER",
    115: "115 APARTMENT ACQUISITION OWNER OCCUPIER",
    116: "116 ACQUIS/CONSTR. PROPERTY ABROAD",
    117: "117 TRANSF / RENOVAT HOUSING",
    118: "118 ACQUIS./CONSTR. PROPERTY ABROAD (SECONDARY RESIDENCE)",
    119: "119 LIFE ANNUITY PRIVATE DESTINATION",
    120: "120 GREEN TRANSF / RENOVAT OWNER OCCUPIER",
    121: "121 ACQUIS./CONSTR. PROPERTY FOR RENTAL",
    123: "123 OTHER PROPERTY INTENDED FOR RENTAL",
    126: "126 ACQUIS./CONSTR. PROPERTY ABROAD (RENTAL)",
    127: "127 TRANSFER/IMPROVEMENTS OF RENTAL PROPERTY",
    128: "128 GREEN TRANSF / RENOVAT RENTAL PROP",
    130: "130 INTERIM FINANCING",
    140: "140 RENEWABLE ENERGY",
    201: "201 LAND FOR PROFESSIONAL USE",
    211: "211 SHOP PREMISES",
    214: "214 HOTEL/RESTAURANT",
    215: "215 PROPERTY FOR PROFESSIONAL USE",
    216: "216 LIFE ANNUITY COMMERCIAL DESTINATION",
    301: "301 WORKING CAPITAL",
    311: "311 PROFESSIONAL EQUIPMENT",
    321: "321 COMMERCIAL OPERATIONS",
    331: "331 PROFESSIONAL REGULATION/RESTRUCTURING",
    341: "341 TRANSPORT EQUIPMENT",
    351: "351 CONVIS HERDBOOK",
    361: "361 TREASURY PROBLEMS COVID-19",
    362: "362 UKRAINE ENERGY CRISIS LIQUIDITY PROBLEMS",
    401: "401 ADC LAND OUT OF PAG / SPECULATIVE",
    402: "402 ADC RESIDENTIAL BUILDING PAG",
    403: "403 ADC OFFICE BUILDING PAG",
    404: "404 ADC INDUS. OR WAREHOUSE SPACE PAG",
    405: "405 ADC RETAIL BUILDING PAG",
    406: "406 ADC HOTEL PAG",
    407: "407 ADC NURSING HOME PAG",
    412: "412 ADC RESIDENTIAL BUILDING PAP",
    413: "413 ADC OFFICE BUILDING PAP",
    414: "414 ADC INDUS. OR WAREHOUSE SPACE PAP",
    415: "415 RETAIL BUILD. PAP",
    416: "416 ADC HOTEL PAP",
    417: "417 ADC NURSING HOME PAP",
    422: "422 ADC RESIDENTIAL PRE-COMMERC. < 20%",
    423: "423 ADC OFFICE BUILDING PRE-COMMERC. < 20%",
    424: "424 ADC INDUS./WAREHOUSE PRE-COMMERC. < 20%",
    425: "425 ADC RETAIL BUILDING PRE-COMMERC. < 20%",
    426: "426 ADC HOTEL PRE-COMMERCIAL < 20%",
    427: "427 ADC NURSING HOME PRE-COMMERCIALISED < 20%",
    432: "432 ADC RESIDENTIAL PRE-COMMERC. 20-70%",
    433: "433 ADC OFFICE PRE-COMMERCIALISED 20-70%",
    434: "434 ADC INDUS./WAREHOUSE PRE-COMMERC. 20%-70%",
    435: "435 ADC RETAIL PRE-COMMERCIALISED 20-70%",
    436: "436 ADC HOTEL PRE-COMMERCIALISED 20-70%",
    437: "437 ADC NURSING HOME PRE-COMMERC. 20%-70%",
    442: "442 ADC RESID. PRE-COMMERCE > 70%",
    443: "443 ADC OFFICE PRE-COMMERCIALISED > 70%",
    444: "444 ADC INDUS./WAREHOUSE PRE-COMMERC. > 70%",
    445: "445 ADC RETAIL PRE-COMMERCIALISED > 70%",
    446: "446 ADC HOTEL PRE-COMMERCIALISED > 70%",
    447: "447 ADC NURSING HOME PRE-COMMERCIALISED >70%",
    452: "452 IPRE RESIDENTIAL LETTING <20%",
    453: "453 IPRE OFFICE LETTING <20%",
    454: "454 IPRE INDUS./WAREHOUSE LETTING <20%",
    455: "455 IPRE RETAIL LETTING <20%",
    456: "456 IPRE HOTEL LETTING <20%",
    457: "457 IPRE NURSING HOME LETTING <20%",
    462: "462 IPRE RESIDENTIAL LETTING 20-70%",
    463: "463 IPRE OFFICE LETTING 20-70%",
    464: "464 IPRE INDUS./WAREHOUSE LETTING 20%-70%",
    465: "465 IPRE RETAIL BUILD. LETTING 20-70%",
    466: "466 IPRE HOTEL LETTING 20-70%",
    467: "467 IPRE NURSING HOME LETTING 20%-70%",
    472: "472 IPRE RESIDENTIAL LETTING > 70%",
    473: "473 IPRE OFFICE PRE-COMMERCIALISED > 70%",
    474: "474 IPRE INDUS./WAREHOUSE LETTING > 70%",
    475: "475 IPRE RETAIL BUILD. LETTING > 70%",
    476: "476 IPRE HOTEL LETTING > 70%",
    477: "477 IPRE NURSING HOME LETTING > 70%",
    481: "481 PROJECT FINANCE",
    482: "482 OBJECT FINANCE",
    483: "483 COMMODITIES FINANCE",
    484: "484 LBO-LEVERAGEBUYOUT",
    485: "485 ENERG. RENEWABLE ENERGY RENEWABLE",
    499: "499 ADC SPECIAL LINE - OTHER NON-SPECIFIED",
    501: "501 SECURITIES OPERATIONS",
    502: "502 INSURANCE INVESTMENTS",
    503: "503 TEMPORARY CASH SHORTFALL",
    504: "504 LIFE INSURANCE PREMIUM FINANCING (ULI)",
    555: "555 TO BE AMENDED",
    602: "602 MB CLASSIC",
    603: "603 VW CLASSIC",
    900: "900 CONSUMPTION",
    901: "901 MEANS OF TRANSPORT",
    902: "902 HOUSEHOLD EQUIPMENT",
    903: "903 MISCELLANEOUS NEEDS",
    904: "904 MEDICAL EXPENSES",
    920: "920 PROPERTY",
    921: "921 PRIVATE USE",
    922: "922 LAND",
    923: "923 PROPERTY - OWNER OCCUPIER",
    924: "924 PROPERTY - RENTAL",
    925: "925 BRIDGING LOANS",
    926: "926 RENEWABLE ENERGY",
    930: "930 PROFESSIONAL/COMMERCIAL USE",
    931: "931 LAND, AGRICULTURAL, FIELD, FOREST",
    932: "932 PROPERTY",
    940: "940 FAMILY COMPANY AND ENTERPRISE",
    941: "941 COMPANY AND FAMILY BUSINESS",
    960: "960 SPECIALISED LENDING",
    961: "961 ADC LAND FINANCING (ACQUISITION)",
    962: "962 ADC WITH PAP",
    963: "963 ADC - CONSTR. PERMIT AND/OR PRE-COMMERCISED <20%",
    964: "964 ADC - PRE-COMMERCIALISED 20% - 70% (< PRIME COSTS)",
    965: "965 ADC - PRE-COMMERCIALISED >70% OR > PRIME COSTS",
    966: "966 IPRE - LETTING RATIO <20%",
    967: "967 IPRE - LETTING RATIO 20% - 70%",
    968: "968 IPRE - LETTING RATIO >70%",
    969: "969 OTHER SPECIALISED LENDING",
    970: "970 ADC - ACQUISITION, DEVELOPMENT AND CONSTRUCTION",
    971: "971 IPRE - INCOME PRODUCING REAL ESTATE",
    980: "980 PORTFOLIO PRODUCTS",
    981: "981 LOMBARDS",
    982: "982 INVESTMENT INSURANCE",
    991: "991 SOUTIEN TRESORERIE CV-19",
}


def get_loan_purpose(row):
    code = row.get(
        "CONTRACT_INTERNAL_PURPOSE_CODE"
    )

    if pd.notna(code):
        try:
            code_int = int(code)
        except (ValueError, TypeError):
            code_int = code

        return purpose_mapping.get(
            code_int,
            "Unknown Code ({})".format(
                code
            )
        )

    return "Unknown / Not Provided"


vars_loan_level_df["purpose_of_the_loan"] = (
    vars_loan_level_df.apply(
        get_loan_purpose,
        axis=1,
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check the number of loans and entities, and make sure each loan-entity-date record is unique.

duplicate_loan_days = int(
    vars_loan_level_df.duplicated(
        subset=[
            "INSTRUMENT_ID",
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ).sum()
)

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Rows:",
    "{:,}".format(
        len(vars_loan_level_df)
    ),
)

print(
    "Instruments:",
    "{:,}".format(
        vars_loan_level_df[
            "INSTRUMENT_ID"
        ].nunique()
    ),
)

print(
    "Entities:",
    "{:,}".format(
        vars_loan_level_df[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Rows with nominal amount:",
    "{:,}".format(
        vars_loan_level_df[
            "NOMINAL_AMOUNT"
        ].notna().sum()
    ),
)

print(
    "Duplicated INSTRUMENT_ID + ENTITY_ID + MAP_DTE:",
    duplicate_loan_days,
)

if duplicate_loan_days > 0:
    raise ValueError(
        "vars_loan_level contains duplicate "
        "INSTRUMENT_ID + ENTITY_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the loan-level details needed by the other datasets.

dataiku.Dataset(
    "vars_loan_level"
).write_with_schema(
    vars_loan_level_df
)