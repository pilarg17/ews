# CORPORATE VALIDATION — Apply daily RFO as-of logic to one recipe only.
# Paste this whole file into a new notebook cell inside Dataiku.
# Does NOT run the recipe or build any dataset.
#
# Changes: compute_vars_loan_level/get_month_data only
# Backup: writes original recipe code to a local .py file before saving.
import ast
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import dataiku

PROJECT_KEY = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
RECIPE_NAME = "compute_vars_loan_level"
PATCH_MARKER = "CORPORATE_RFO_ASOF_V1"

new_function = r'''
def get_month_data(month):
    """Corporate daily instruments joined to latest RFO snapshot <= each day.

    Uses a shared LOANDEPO/ACCOUNT REPORT_DATEKEY, independently dated
    PROVISION and FX snapshots. Does not use a later REPORT_DATEKEY than MAP_DTE.
    """
    start_date = month.start_time
    end_date = (month + 1).start_time
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    month_key = month.strftime("%Y%m")
    print("\nProcessing month:", month_key)

    # CORPORATE_RFO_ASOF_V1: latest eligible RFO snapshot per daily date.
    # Cache report dates across all months in this recipe execution.
    # Metadata query returns report dates only, not customer data.
    if not hasattr(get_month_data, "_asof_source_dates"):
        cap_int = int(pd.Timestamp(max_map_dte).strftime("%Y%m%d"))
        source_dates_sql = f"""
        SELECT 'LOANDEPO' AS SOURCE_NAME, report_datekey
        FROM DTM_RFO_PLY.LOANDEPO_VW
        WHERE report_datekey <= {cap_int}
        GROUP BY report_datekey
        UNION ALL
        SELECT 'ACCOUNT' AS SOURCE_NAME, report_datekey
        FROM DTM_RFO_PLY.ACCOUNT_VW
        WHERE report_datekey <= {cap_int}
        GROUP BY report_datekey
        UNION ALL
        SELECT 'FX' AS SOURCE_NAME, report_datekey
        FROM DTM_RFO_PLY.MKT_FX_INPUT_VW
        WHERE report_datekey <= {cap_int}
        GROUP BY report_datekey
        UNION ALL
        SELECT 'PROVISION' AS SOURCE_NAME, report_datekey
        FROM DTM_RFO_PLY.PROVISION_VW
        WHERE report_datekey <= {cap_int}
        GROUP BY report_datekey
        """
        source_df = executeQuery(db_connection, source_dates_sql)
        source_df.columns = [str(x).upper() for x in source_df.columns]
        source_df["REPORT_DATEKEY"] = pd.to_numeric(
            source_df["REPORT_DATEKEY"], errors="coerce"
        )
        all_dates = {}
        for source in ["LOANDEPO", "ACCOUNT", "FX", "PROVISION"]:
            a = source_df.loc[
                source_df["SOURCE_NAME"].astype("string").str.upper().eq(source),
                "REPORT_DATEKEY",
            ].dropna().astype("int64")
            all_dates[source] = set(a.tolist())
        shared_rfo = sorted(all_dates["LOANDEPO"] & all_dates["ACCOUNT"])
        if not shared_rfo:
            raise ValueError("No shared LOANDEPO/ACCOUNT RFO snapshot dates found")
        get_month_data._asof_source_dates = {
            "RFO": shared_rfo,
            "FX": sorted(all_dates["FX"]),
            "PROVISION": sorted(all_dates["PROVISION"]),
        }

    available_dates = get_month_data._asof_source_dates

    # Read only the available instrument observation dates in this month.
    day_query = f"""
    SELECT DISTINCT TRUNC(map_dte) AS MAP_DTE
    FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_DAY_VW
    WHERE map_dte >= DATE '{start_str}'
      AND map_dte < DATE '{end_str}'
    ORDER BY MAP_DTE
    """
    day_frame = executeQuery(db_connection, day_query)
    day_frame.columns = [str(c).upper() for c in day_frame.columns]
    days = pd.to_datetime(day_frame["MAP_DTE"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    days = sorted(set(days.dropna().tolist()))
    if not days:
        print("No instrument days in selected period")
        return pd.DataFrame()

    from bisect import bisect_right
    def latest_not_after(sorted_dates, cutoff):
        pos = bisect_right(sorted_dates, cutoff)
        return sorted_dates[pos - 1] if pos else None

    # Group days sharing the same as-of source dates to avoid one big SQL
    # query per daily observation. Provision and FX can have separate dates.
    buckets = {}
    for day in days:
        day_int = int(pd.Timestamp(day).strftime("%Y%m%d"))
        rfo_date = latest_not_after(available_dates["RFO"], day_int)
        fx_date = latest_not_after(available_dates["FX"], rfo_date) if rfo_date else None
        provision_date = latest_not_after(available_dates["PROVISION"], day_int)
        key = (rfo_date, fx_date, provision_date)
        buckets.setdefault(key, []).append(pd.Timestamp(day).strftime("%Y-%m-%d"))

    parts = []
    print("Distinct as-of source combinations:", len(buckets))
    for (rfo_date, fx_date, provision_date), date_group in buckets.items():
        print("Daily range:", date_group[0], "to", date_group[-1],
              "| RFO:", rfo_date, "| FX:", fx_date,
              "| PROVISION:", provision_date)
        rfo_sql_date = str(rfo_date) if rfo_date is not None else "NULL"
        fx_sql_date = str(fx_date) if fx_date is not None else "NULL"
        provision_sql_date = (
            str(provision_date) if provision_date is not None else "NULL"
        )
        day_literals = ", ".join(f"DATE '{d}'" for d in date_group)

        query = f"""    WITH

    loandepo AS (
        SELECT *
        FROM (
            SELECT
                ld.contract_reference,
                ld.report_datekey,
                ld.currency,
                ld.accruals + ld.outstanding AS gross_carrying_amount,
                ld.outstanding AS nominal_amount,
                'CONTRACT_REFERENCE' AS id_type,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        ld.contract_reference,
                        SUBSTR(
                            TRIM(
                                TO_CHAR(
                                    ld.report_datekey
                                )
                            ),
                            1,
                            6
                        )
                    ORDER BY
                        ld.report_datekey DESC,
                        ld.report_versionid DESC
                ) AS rn

            FROM DTM_RFO_PLY.LOANDEPO_VW ld

            WHERE
                (ld.accruals + ld.outstanding) > 0
                AND ld.report_datekey = {rfo_sql_date}
        )

        WHERE rn = 1
    ),


    account_data AS (
        SELECT *
        FROM (
            SELECT
                acc.account_reference AS contract_reference,
                acc.report_datekey,
                acc.currency,
                acc.accruals + acc.balance AS gross_carrying_amount,
                acc.balance AS nominal_amount,
                'ACCOUNT_REFERENCE' AS id_type,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        acc.account_reference,
                        SUBSTR(
                            TRIM(
                                TO_CHAR(
                                    acc.report_datekey
                                )
                            ),
                            1,
                            6
                        )
                    ORDER BY
                        acc.report_datekey DESC,
                        acc.report_versionid DESC
                ) AS rn

            FROM DTM_RFO_PLY.ACCOUNT_VW acc

            WHERE
                (acc.accruals + acc.balance) > 0
                AND acc.report_datekey = {rfo_sql_date}
        )

        WHERE rn = 1
    ),


    exposure_raw AS (
        SELECT
            contract_reference,
            report_datekey,
            currency,
            gross_carrying_amount,
            nominal_amount,
            id_type
        FROM loandepo

        UNION ALL

        SELECT
            contract_reference,
            report_datekey,
            currency,
            gross_carrying_amount,
            nominal_amount,
            id_type
        FROM account_data
    ),


    fx_data AS (
        SELECT *
        FROM (
            SELECT
                fx.ccy_code,
                fx.report_datekey,
                fx.rate,
                fx.value_date,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        fx.ccy_code,
                        SUBSTR(
                            TRIM(
                                TO_CHAR(
                                    fx.report_datekey
                                )
                            ),
                            1,
                            6
                        )
                    ORDER BY
                        fx.report_datekey DESC,
                        fx.report_versionid DESC
                ) AS rn

            FROM DTM_RFO_PLY.MKT_FX_INPUT_VW fx

            WHERE fx.report_datekey = {fx_sql_date}
        )

        WHERE rn = 1
    ),


    exposure AS (
        SELECT
            exp.contract_reference,
            exp.currency,

            exp.gross_carrying_amount
                * fx.rate
                AS gross_carrying_amount,

            exp.nominal_amount
                * fx.rate
                AS nominal_amount,

            fx.rate,
            exp.id_type

        FROM exposure_raw exp

        LEFT JOIN fx_data fx
            ON exp.currency = fx.ccy_code

    ),


    stage_data AS (
        SELECT *
        FROM (
            SELECT
                s.contract_reference,
                s.dim1,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        s.contract_reference,
                        SUBSTR(
                            TRIM(
                                TO_CHAR(
                                    s.report_datekey
                                )
                            ),
                            1,
                            6
                        )
                    ORDER BY
                        s.report_datekey DESC,
                        s.report_versionid DESC
                ) AS rn

            FROM DTM_RFO_PLY.PROVISION_VW s

            WHERE s.report_datekey = {provision_sql_date}
        )

        WHERE rn = 1
    )


    SELECT
        ins.instrument_id,
        ins.contract_id,

        CASE
            WHEN INSTR(
                ins.contract_id,
                '_211'
            ) > 0

            THEN SUBSTR(
                ins.contract_id,
                INSTR(
                    ins.contract_id,
                    '_211'
                ) + 4
            )

            ELSE NULL
        END AS contract_id_short,

        ins.map_dte,
        ins.dpd,
        ins.past_due_date,
        ins.arrears_amount,
        ins.default_status,
        ins.forbearance_status,
        ins.legal_final_maturity_date,
        ins.original_maturity_date,
        ins.bridge_loan_amount_lcy,
        ins.contract_internal_purpose_code,
        ins.tek_strt_dte,
        ins.tek_end_dte,

        i2e.entity_id,

        lim.limit_amount_ctv,

        exp.contract_reference,
        exp.currency,
        exp.gross_carrying_amount,
        exp.nominal_amount,
        exp.rate,
        exp.id_type,

        stg.dim1

    FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_DAY_VW ins

    INNER JOIN BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_DAY_VW i2e

        ON ins.instrument_id = i2e.instrument_id
        AND ins.map_dte = i2e.map_dte


    LEFT JOIN BDL_LOANS_PLY.LIMIT_DAY_VW lim

        ON lim.limit_id =
            CASE
                WHEN INSTR(
                    ins.contract_id,
                    '_211'
                ) > 0

                THEN SUBSTR(
                    ins.contract_id,
                    INSTR(
                        ins.contract_id,
                        '_211'
                    ) + 4
                )

                ELSE NULL
            END

        AND lim.map_dte = ins.map_dte


    LEFT JOIN exposure exp

        ON TRIM(
            ins.instrument_id
        ) = TRIM(
            exp.contract_reference
        )


    LEFT JOIN stage_data stg

        ON TRIM(
            ins.instrument_id
        ) = TRIM(
            stg.contract_reference
        )


    WHERE ins.map_dte >= TO_DATE(
        '{start_str}',
        'YYYY-MM-DD'
    )

    AND ins.map_dte < TO_DATE(
        '{end_str}',
        'YYYY-MM-DD'
    )

    AND TRUNC(ins.map_dte) IN ({day_literals})
        """

        df = executeQuery(db_connection, query)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
'''

client = dataiku.api_client()
project = client.get_project(PROJECT_KEY)
recipe = project.get_recipe(RECIPE_NAME)
settings = recipe.get_settings()

if settings.type != "python":
    raise RuntimeError("Expected a Python recipe, got: " + str(settings.type))

original_code = settings.get_code()
if PATCH_MARKER in original_code:
    print("Already updated: patch marker found; no changes saved.")
else:
    original_tree = ast.parse(original_code)
    funcs = [
        node for node in original_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_month_data"
    ]
    if len(funcs) != 1:
        raise RuntimeError(
            "Expected exactly one top-level get_month_data function; found "
            + str(len(funcs))
        )

    # Stop rather than modifying an unexpected recipe revision.
    existing_fn = funcs[0]
    original_lines = original_code.splitlines(keepends=True)
    old_function = "".join(
        original_lines[existing_fn.lineno - 1:existing_fn.end_lineno]
    )

    required = [
        "DTM_RFO_PLY.LOANDEPO_VW",
        "DTM_RFO_PLY.ACCOUNT_VW",
        "DTM_RFO_PLY.MKT_FX_INPUT_VW",
        "DTM_RFO_PLY.PROVISION_VW",
        "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_DAY_VW",
        "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_DAY_VW",
        "BDL_LOANS_PLY.LIMIT_DAY_VW",
        "month_key",
    ]
    missing = [token for token in required if token not in old_function]
    if missing:
        raise RuntimeError(
            "Recipe function differs from the reviewed version; missing: "
            + ", ".join(missing)
        )

    if old_function.count("= '{month_key}'") != 4:
        raise RuntimeError(
            "Expected four same-month RFO/FX/Provision filters. "
            "The recipe may have changed; aborting."
        )

    # Construct replacement only at this AST function boundary.
    prefix = "".join(original_lines[:existing_fn.lineno - 1])
    suffix = "".join(original_lines[existing_fn.end_lineno:])
    revised_code = prefix + new_function.rstrip("\n") + suffix
    ast.parse(revised_code)  # Syntax check, no execution.

    # Local backup is mandatory; any failure aborts before the API update.
    backup_name = (
        "compute_vars_loan_level_BEFORE_RFO_ASOF_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + ".py"
    )
    backup_path = Path.cwd() / backup_name
    with backup_path.open("x", encoding="utf-8") as backup:
        backup.write(original_code)

    before_hash = hashlib.sha256(
        original_code.encode("utf-8")
    ).hexdigest()
    after_hash = hashlib.sha256(
        revised_code.encode("utf-8")
    ).hexdigest()

    print("Project:", PROJECT_KEY)
    print("Recipe:", RECIPE_NAME)
    print("Backup:", backup_path)
    print("Original recipe lines:", len(original_lines))
    print("New recipe lines:", len(revised_code.splitlines()))
    print("Original SHA256:", before_hash)
    print("Updated  SHA256:", after_hash)
    print("Changing only get_month_data; no output datasets built.")

    settings.set_code(revised_code)
    settings.save()

    # Verify persisted version, not only the in-memory settings object.
    persisted = project.get_recipe(RECIPE_NAME).get_settings().get_code()
    persisted_hash = hashlib.sha256(
        persisted.encode("utf-8")
    ).hexdigest()
    if persisted_hash != after_hash:
        raise RuntimeError(
            "Recipe was saved but verification hash differs! "
            "Check recipe immediately; original backup is at " + str(backup_path)
        )

    print()
    print("SUCCESS — Corporate RFO as-of criterion saved in Dataiku.")
    print("Only code was saved; no recipe/Flow execution occurred.")
    print("Keep the backup available for rollback.")
