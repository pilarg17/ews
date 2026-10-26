# -*- coding: utf-8 -*-
# Backfill / Main review: read-only comparison of full-group and Retail-scoped connected flags.
# Run this in a Python notebook belonging to the Backfill project.
# No dataset writes, no schema changes, no real customer or entity IDs in output.

import dataiku
import pandas as pd

STAGE2_EXPOSURE_THRESHOLD = 150000.0


def norm_id(values):
    return (
        values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def norm_date(values):
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def on_date(df, label, date):
    if "MAP_DTE" not in df:
        raise ValueError("MAP_DTE missing in " + label)
    tmp = df.loc[norm_date(df["MAP_DTE"]).eq(date)].copy()
    if tmp.empty:
        raise ValueError("No rows at reference date in " + label)
    return tmp


def show(label, value):
    print("{:<53} {:>10,}".format(label, int(value)))


reference = dataiku.Dataset("Implementation_Reference_Date").get_dataframe()
if len(reference) != 1 or "REFERENCE_DATE" not in reference.columns:
    raise ValueError("Reference date must contain exactly one row")
run_date = norm_date(reference["REFERENCE_DATE"]).iloc[0]
if pd.isna(run_date):
    raise ValueError("Reference date missing")

print("=" * 81)
print("CONNECTED GROUP PARITY QA (READ ONLY)")
print("=" * 81)
print("Reference date:", run_date.strftime("%Y-%m-%d"))

entities = on_date(dataiku.Dataset("Entity_Information").get_dataframe(), "Entity_Information", run_date)
portfolio = on_date(dataiku.Dataset("Retail_Portfolio_Daily").get_dataframe(), "Retail_Portfolio_Daily", run_date)
loans = on_date(dataiku.Dataset("vars_loan").get_dataframe(), "vars_loan", run_date)
exposures = on_date(dataiku.Dataset("GEC_Exposure_Current").get_dataframe(), "GEC_Exposure_Current", run_date)

required = {
    "Entity_Information": (entities, ["ENTITY_ID", "GCC_PARENT_ID", "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE"]),
    "Retail_Portfolio_Daily": (portfolio, ["ENTITY_ID", "RRE_FLAG", "OTHER_RETAIL_FLAG"]),
    "vars_loan": (loans, ["ENTITY_ID", "STAGE"]),
    "GEC_Exposure_Current": (exposures, ["ENTITY_ID", "GEC_NET_EXPOSURE"]),
}
for name, (df, cols) in required.items():
    missing = [c for c in cols if c not in df]
    if missing:
        raise ValueError("Missing {} column(s): {}".format(name, missing))
    df["ENTITY_ID"] = norm_id(df["ENTITY_ID"])

entities["GCC_PARENT_ID"] = norm_id(entities["GCC_PARENT_ID"])
entities["FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE"] = pd.to_numeric(
    entities["FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE"], errors="coerce"
)
entity_full = (
    entities.groupby("ENTITY_ID", as_index=False)
    .agg(GCC_PARENT_ID=("GCC_PARENT_ID", "first"),
         UTP_SOURCE=("FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE", "first"))
)
entity_full["UTP_FLAG"] = entity_full["UTP_SOURCE"].gt(0.5).fillna(False).astype("int8")

for c in ["RRE_FLAG", "OTHER_RETAIL_FLAG"]:
    portfolio[c] = pd.to_numeric(portfolio[c], errors="coerce").fillna(0)
if portfolio["ENTITY_ID"].duplicated().any():
    raise ValueError("Retail portfolio has duplicate ENTITY_ID rows")
portfolio = portfolio.loc[
    portfolio["RRE_FLAG"].eq(1) | portfolio["OTHER_RETAIL_FLAG"].eq(1),
    ["ENTITY_ID", "RRE_FLAG", "OTHER_RETAIL_FLAG"],
].copy()
entity_scoped = entity_full.loc[entity_full["ENTITY_ID"].isin(portfolio["ENTITY_ID"])].copy()

loans["STAGE"] = pd.to_numeric(loans["STAGE"], errors="coerce")
loan_stages = (
    loans.assign(STAGE2=loans["STAGE"].eq(2).astype("int8"))
    .groupby("ENTITY_ID", as_index=False)
    .agg(STAGE2=("STAGE2", "max"))
)
exposures["GEC_NET_EXPOSURE"] = pd.to_numeric(exposures["GEC_NET_EXPOSURE"], errors="coerce")
if exposures["ENTITY_ID"].duplicated().any():
    raise ValueError("GEC_Exposure_Current contains duplicate ENTITY_ID at reference date")


def connected_flags(entity_base, prefix):
    related = entity_base[["ENTITY_ID", "GCC_PARENT_ID", "UTP_FLAG"]].merge(
        loan_stages, how="left", on="ENTITY_ID", validate="1:1"
    )
    related["STAGE2"] = related["STAGE2"].fillna(0).astype("int8")
    totals = (
        related.dropna(subset=["GCC_PARENT_ID"])
        .groupby("GCC_PARENT_ID", as_index=False)
        .agg(MEMBERS=("ENTITY_ID", "nunique"),
             STAGE2_COUNT=("STAGE2", "sum"),
             UTP_COUNT=("UTP_FLAG", "sum"))
    )
    related = related.merge(totals, how="left", on="GCC_PARENT_ID", validate="m:1")
    related["STAGE2_OTHER"] = (
        related["STAGE2_COUNT"].fillna(0) - related["STAGE2"]
    ).gt(0).astype("int8")
    related["UTP_OTHER"] = (
        related["UTP_COUNT"].fillna(0) - related["UTP_FLAG"]
    ).gt(0).astype("int8")
    return related[["ENTITY_ID", "STAGE2_OTHER", "UTP_OTHER", "MEMBERS"]].rename(
        columns={"STAGE2_OTHER": prefix + "_STAGE2", "UTP_OTHER": prefix + "_UTP",
                 "MEMBERS": prefix + "_MEMBERS"}
    )

full_calc = connected_flags(entity_full, "FULL")
retail_calc = connected_flags(entity_scoped, "FAST")
q = (
    portfolio.merge(full_calc, on="ENTITY_ID", how="left", validate="1:1")
    .merge(retail_calc, on="ENTITY_ID", how="left", validate="1:1")
    .merge(exposures[["ENTITY_ID", "GEC_NET_EXPOSURE"]], on="ENTITY_ID", how="left", validate="1:1")
)
if q[["FULL_STAGE2", "FAST_STAGE2", "FULL_UTP", "FAST_UTP"]].isna().any().any():
    raise ValueError("Retail entities missing from entity information")

show("Full entity population", len(entity_full))
show("Retail entity population", len(entity_scoped))
show("Retail entities evaluated", len(q))
print("-" * 81)

maturity = q["GEC_NET_EXPOSURE"].gt(STAGE2_EXPOSURE_THRESHOLD)
for code, portcol in [("RRE", "RRE_FLAG"), ("OTHER", "OTHER_RETAIL_FLAG")]:
    in_port = q[portcol].eq(1)
    for alert_type in ["STAGE2", "UTP"]:
        mat = maturity if alert_type == "STAGE2" else pd.Series(True, index=q.index)
        full = in_port & mat & q["FULL_" + alert_type].eq(1)
        fast = in_port & mat & q["FAST_" + alert_type].eq(1)
        show("{} {} full-group flags".format(code, alert_type), full.sum())
        show("{} {} FAST-scoped flags".format(code, alert_type), fast.sum())
        show("{} {} DIFFERENCES".format(code, alert_type), full.ne(fast).sum())
        show("{} {} lost in FAST".format(code, alert_type), (full & ~fast).sum())
        print("-" * 81)

show("Entities with different full vs scoped group sizes", q["FULL_MEMBERS"].fillna(0).ne(q["FAST_MEMBERS"].fillna(0)).sum())

# Optional check that the successfully produced FAST output agrees with the calculation.
try:
    actual = on_date(dataiku.Dataset("Available_Indicators_Current").get_dataframe(),
                     "Available_Indicators_Current", run_date)
    actual["ENTITY_ID"] = norm_id(actual["ENTITY_ID"])
    check = q[["ENTITY_ID", "RRE_FLAG", "OTHER_RETAIL_FLAG", "GEC_NET_EXPOSURE", "FAST_STAGE2", "FAST_UTP"]].merge(
        actual, on="ENTITY_ID", how="left", validate="1:1", suffixes=("", "_ACTUAL")
    )
    for output_col, portcol, basecol, require_mat in [
        ("RRE_GCC_STAGE2_CONNECTED", "RRE_FLAG", "FAST_STAGE2", True),
        ("OTHER_GCC_STAGE2_CONNECTED", "OTHER_RETAIL_FLAG", "FAST_STAGE2", True),
        ("RRE_GCC_UTP_CONNECTED", "RRE_FLAG", "FAST_UTP", False),
        ("OTHER_GCC_UTP_CONNECTED", "OTHER_RETAIL_FLAG", "FAST_UTP", False),
    ]:
        if output_col not in check:
            print("MISSING OUTPUT FIELD:", output_col)
            continue
        expected = (check[portcol].eq(1) & check[basecol].eq(1) &
                    (check["GEC_NET_EXPOSURE"].gt(STAGE2_EXPOSURE_THRESHOLD) if require_mat else True))
        observed = pd.to_numeric(check[output_col], errors="coerce").fillna(0).eq(1)
        show("FAST output mismatch for " + output_col, expected.ne(observed).sum())
except Exception as ex:
    print("Actual output comparison unavailable:", type(ex).__name__, str(ex)[:150])

print("=" * 81)
print("INTERPRETATION: Any FULL vs FAST difference requires review before Main deployment.")
print("No datasets written. No IDs printed. All results are counts only.")
