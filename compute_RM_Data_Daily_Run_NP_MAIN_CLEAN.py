# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Database connection used to retrieve the RM and organisational hierarchy.

DB_CONNECTION = "PRD_BDWH"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the reference date so that RM assignments reflect the same day
# as the rest of the Retail EWS data.

reference_date_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()


if reference_date_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
    )


if "REFERENCE_DATE" not in reference_date_df.columns:
    raise KeyError(
        "Implementation_Reference_Date is missing REFERENCE_DATE."
    )


# Require one valid date to avoid mixing RM assignments from different days.
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
        "Implementation_Reference_Date must contain "
        "exactly one valid REFERENCE_DATE."
    )


REFERENCE_DATE = reference_dates.iloc[0]

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Run a SQL query and report its execution time.

def execute_query(database, query):

    start_time = time.time()

    result = SQLExecutor2(
        connection=database
    ).query_to_df(
        query
    )

    print(
        "Time to perform the query {:.1f} seconds".format(
            time.time() - start_time
        )
    )

    return result


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Link customers to their entities and retrieve the assigned RM,
# business line, team and management hierarchy for the reference date.
# Certain business lines use a specific RM assignment defined in the source rules.

# RM details come from the party and organisational hierarchy tables.
# Customer-to-entity links are taken at the selected reference date.
query = """
SELECT DISTINCT

    ec.entity_id AS ENTITY_ID,

    prty.prty_id AS ID_ACTEUR,

    main_dao.dao_area_desc,
    
    case
        when
            hier_flat2.business_line in ('Wealth Management','Group Head Internat. CEO BIL Suisse')
        then 
            'RM7B'
        else 
            NVL(
                emp.empl_bil_cd,
                main_dao.empl_cd
            )
    END
    
    AS ID_RM,
    
    case
        when
            hier_flat2.business_line in ('Wealth Management','Group Head Internat. CEO BIL Suisse')
        then 
            'Catherine BASTIEN'
        else
            main_dao.dao_lbl
            
     END
     AS RM,

    main_dao.cost_cntr_lgcy_cd AS AGENCE,

    main_dao.dao_lvl_cd,

    main_dao.dao_typ,

    hier_flat2.business_line,

    hier_flat2.line,

    hier_flat2.team,

    dao_resp_head.empl_cd
        AS ID_BUSINESS_HEAD,

    dao_resp_head.dao_area_desc
        AS BUSINESS_HEAD,

    case
        when
            hier_flat2.business_line in ('Wealth Management','Group Head Internat. CEO BIL Suisse')
        then 
            'RM7B'
        else 
            dao_resp_n_1.empl_cd
     END
     AS "ID_RM_N+1",

    case
        when
            hier_flat2.business_line in ('Wealth Management','Group Head Internat. CEO BIL Suisse')
        then 
            'Catherine BASTIEN'
        else
            dao_resp_n_1.dao_area_desc
            
     END AS "RM_N+1",

    dao_resp_n_2.empl_cd
        AS "ID_RM_N+2",

    dao_resp_n_2.dao_area_desc
        AS "RM_N+2",

    dao_resp_n_3.empl_cd
        AS "ID_RM_N+3",

    dao_resp_n_3.dao_area_desc
        AS "RM_N+3"


FROM BDWH_PLY.REF_PRTY_VW prty


LEFT JOIN (

    SELECT DISTINCT
        entity_id,
        customer_id

    FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_VW

    WHERE tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

      AND entity_id IS NOT NULL
      AND customer_id IS NOT NULL

) ec

    ON TRIM(
        CAST(
            ec.customer_id
            AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            prty.prty_id
            AS VARCHAR2(100)
        )
    )


LEFT JOIN BDWH_PLY.REF_DAO_VW main_dao

    ON main_dao.dao_tid =
       prty.main_dao_tid

    AND main_dao.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(main_dao.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


LEFT JOIN BDWH_PLY.REF_EMPL_VW emp

    ON emp.empl_tid =
       main_dao.empl_tid

    AND emp.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(emp.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


LEFT JOIN (

    SELECT

        hier_flat.dao_tid,

        hier_flat.lvl_lbl_1
            AS BUSINESS_LINE,

        hier_flat.lvl_lbl_2
            AS LINE,

        hier_flat.lvl_lbl_6
            AS TEAM,

        hier_flat.resp_dao_tid_1
            AS DAO_TID_HEAD,

        CASE

            WHEN hier_flat.dao_tid
                 <> hier_flat.resp_dao_tid_6
                THEN hier_flat.resp_dao_tid_6

            WHEN hier_flat.dao_tid
                 <> hier_flat.resp_dao_tid_5
                THEN hier_flat.resp_dao_tid_5

            WHEN hier_flat.dao_tid
                 <> hier_flat.resp_dao_tid_4
                THEN hier_flat.resp_dao_tid_4

            WHEN hier_flat.dao_tid
                 <> hier_flat.resp_dao_tid_3
                THEN hier_flat.resp_dao_tid_3

            WHEN hier_flat.dao_tid
                 <> hier_flat.resp_dao_tid_2
                THEN hier_flat.resp_dao_tid_2

            ELSE NULL

        END AS DAO_TID_N_1,


        CASE

            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_5
                THEN hier_flat.resp_dao_tid_5

            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_4
                THEN hier_flat.resp_dao_tid_4

            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_3
                THEN hier_flat.resp_dao_tid_3

            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_2
                THEN hier_flat.resp_dao_tid_2

            ELSE NULL

        END AS DAO_TID_N_2,


        CASE

            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_5

             AND hier_flat.resp_dao_tid_5
                 <> hier_flat.resp_dao_tid_4

                THEN hier_flat.resp_dao_tid_4


            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_5

             AND hier_flat.resp_dao_tid_5
                 <> hier_flat.resp_dao_tid_3

                THEN hier_flat.resp_dao_tid_3


            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_5

             AND hier_flat.resp_dao_tid_5
                 <> hier_flat.resp_dao_tid_2

                THEN hier_flat.resp_dao_tid_2


            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_4

             AND hier_flat.resp_dao_tid_4
                 <> hier_flat.resp_dao_tid_3

                THEN hier_flat.resp_dao_tid_3


            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_4

             AND hier_flat.resp_dao_tid_4
                 <> hier_flat.resp_dao_tid_2

                THEN hier_flat.resp_dao_tid_2


            WHEN hier_flat.resp_dao_tid_6
                 <> hier_flat.resp_dao_tid_3

             AND hier_flat.resp_dao_tid_3
                 <> hier_flat.resp_dao_tid_2

                THEN hier_flat.resp_dao_tid_2


            ELSE NULL

        END AS DAO_TID_N_3,


        CASE

            WHEN hier_flat.resp_user_idnt_6
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_6

            WHEN hier_flat.resp_user_idnt_5
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_5

            WHEN hier_flat.resp_user_idnt_4
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_4

            WHEN hier_flat.resp_user_idnt_3
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_3

            WHEN hier_flat.resp_user_idnt_2
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_2

            WHEN hier_flat.resp_user_idnt_1
                 IS NOT NULL
                THEN hier_flat.resp_user_idnt_1

            ELSE NULL

        END AS RESP_IDNT


    FROM BDWH_PLY.LNK_EUC_HCHY_FLAT_VW hier_flat

    WHERE hier_flat.hchy_clss_cd =
          'HIERARCHIORG'

      AND hier_flat.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(hier_flat.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

) hier_flat2

    ON hier_flat2.dao_tid =
       COALESCE(
           main_dao.dao_par_tid,
           main_dao.dao_tid
       )


LEFT JOIN BDWH_PLY.REF_DAO_VW dao_resp_head

    ON dao_resp_head.dao_tid =
       hier_flat2.dao_tid_head

    AND dao_resp_head.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(dao_resp_head.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


LEFT JOIN BDWH_PLY.REF_DAO_VW dao_resp_n_1

    ON TRIM(
        CAST(
            dao_resp_n_1.empl_cd
            AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            hier_flat2.resp_idnt
            AS VARCHAR2(100)
        )
    )

    AND dao_resp_n_1.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(dao_resp_n_1.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


LEFT JOIN BDWH_PLY.REF_DAO_VW dao_resp_n_2

    ON dao_resp_n_2.dao_tid =
       hier_flat2.dao_tid_n_2

    AND dao_resp_n_2.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(dao_resp_n_2.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


LEFT JOIN BDWH_PLY.REF_DAO_VW dao_resp_n_3

    ON dao_resp_n_3.dao_tid =
       hier_flat2.dao_tid_n_3

    AND dao_resp_n_3.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(dao_resp_n_3.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)


WHERE prty.tek_strt_dte < (DATE '{reference_date}' + 1)

AND NVL(prty.tek_end_dte, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)

""".format(
    reference_date=REFERENCE_DATE_STR
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Retrieve the RM assignments and organisational details.

rmdata_df = execute_query(
    DB_CONNECTION,
    query,
)
print("RM query rows:", len(rmdata_df))


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardise the identifiers used to match RM information with
# other entity and customer datasets.

for id_column in [
    "ENTITY_ID",
    "ID_ACTEUR",
    "ID_RM",
    "ID_RM_N+1",
    "ID_RM_N+2",
    "ID_RM_N+3",
]:

    if id_column in rmdata_df.columns:

        rmdata_df[
            id_column
        ] = (
            rmdata_df[
                id_column
            ]
            .astype("string")
            .str.strip()
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )


rmdata_df[
    "REFERENCE_DATE"
] = REFERENCE_DATE


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check how many records contain the main RM and organisational fields.

print()
print("=" * 70)
print("RM DATA DAILY RUN QA")
print("=" * 70)

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Rows:",
    "{:,}".format(
        len(
            rmdata_df
        )
    ),
)

print(
    "Rows with ENTITY_ID:",
    "{:,}".format(
        int(
            rmdata_df[
                "ENTITY_ID"
            ]
            .notna()
            .sum()
        )
    ),
)

print(
    "Distinct ENTITY_ID:",
    "{:,}".format(
        rmdata_df[
            "ENTITY_ID"
        ]
        .nunique(
            dropna=True
        )
    ),
)

print(
    "Rows with RM:",
    "{:,}".format(
        int(
            rmdata_df[
                "RM"
            ]
            .notna()
            .sum()
        )
    ),
)

print(
    "Rows with RM N+1:",
    "{:,}".format(
        int(
            rmdata_df[
                "RM_N+1"
            ]
            .notna()
            .sum()
        )
    ),
)

print(
    "Rows with BUSINESS_LINE:",
    "{:,}".format(
        int(
            rmdata_df[
                "BUSINESS_LINE"
            ]
            .notna()
            .sum()
        )
    ),
)


# Report the Corporate and Institutional Banking population for visibility.
# This check does not exclude those records; the Retail filter is applied below.
corporate_mask = (
    rmdata_df[
        "BUSINESS_LINE"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.casefold()
    .eq(
        "corporate and institutional banking"
    )
)


print(
    "Corporate and Institutional Banking entities:",
    "{:,}".format(
        rmdata_df.loc[
            corporate_mask,
            "ENTITY_ID",
        ]
        .nunique(
            dropna=True
        )
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Stop if no RM records were found.

if rmdata_df.empty:
    raise ValueError(
        "RM_Data_Daily_Run is empty."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Keep only the entities included in RRE or Other Retail on the
# reference date, so the RM output matches the EWS Retail population.

# Use the daily portfolio flags rather than RM business-line labels
# to decide which entities belong in the Retail output.
retail_portfolio = dataiku.Dataset("Retail_Portfolio_Daily").get_dataframe()
required_retail_columns = ["ENTITY_ID", "MAP_DTE", "RRE_FLAG", "OTHER_RETAIL_FLAG"]
missing_retail_columns = [c for c in required_retail_columns if c not in retail_portfolio.columns]
if missing_retail_columns:
    raise KeyError("Retail_Portfolio_Daily is missing: {}".format(missing_retail_columns))

retail_portfolio["ENTITY_ID"] = (
    retail_portfolio["ENTITY_ID"].astype("string").str.strip()
    .str.replace(r"\.0$", "", regex=True)
)
retail_dates = (
    pd.to_datetime(retail_portfolio["MAP_DTE"], errors="coerce", utc=True)
    .dt.tz_localize(None).dt.normalize()
)
retail_portfolio = retail_portfolio.loc[retail_dates.eq(REFERENCE_DATE)].copy()
if retail_portfolio.empty:
    raise ValueError("Retail_Portfolio_Daily has no rows on the reference date.")
if retail_portfolio["ENTITY_ID"].duplicated().any():
    raise ValueError("Duplicate Retail portfolio ENTITY_ID for reference date.")

in_scope = (
    pd.to_numeric(retail_portfolio["RRE_FLAG"], errors="coerce").eq(1)
    | pd.to_numeric(retail_portfolio["OTHER_RETAIL_FLAG"], errors="coerce").eq(1)
)
retail_ids = set(retail_portfolio.loc[in_scope, "ENTITY_ID"].dropna())
if not retail_ids:
    raise ValueError("Retail portfolio scope is empty.")

# Keep the full hierarchy available for QA, but only write Retail rows.
rm_all_rows = len(rmdata_df)
rmdata_df = rmdata_df.loc[rmdata_df["ENTITY_ID"].isin(retail_ids)].copy()
if rmdata_df.empty:
    raise ValueError("No RM rows matched the Retail scope; output was not written.")

print("RM output original rows:", rm_all_rows, flush=True)
print("RM output Retail rows:", len(rmdata_df), flush=True)
print("RM rows avoided:", rm_all_rows - len(rmdata_df), flush=True)
print("Retail entities with RM mapping:", rmdata_df["ENTITY_ID"].nunique(), flush=True)
print("Retail entities without RM mapping:", len(retail_ids - set(rmdata_df["ENTITY_ID"].dropna())), flush=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the RM information used to enrich the final EWS breaches.

_rm_write_started = time.perf_counter()

dataiku.Dataset(
    "RM_Data_Daily_Run"
).write_with_schema(
    rmdata_df
)

print("RM write seconds:", round(time.perf_counter() - _rm_write_started, 1))