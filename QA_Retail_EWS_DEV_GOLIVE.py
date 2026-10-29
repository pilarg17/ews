import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2


def normalize_id(values):
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))


reference = dataiku.Dataset("Implementation_Reference_Date").get_dataframe()
alerts = dataiku.Dataset("EWS_Daily_Alerts").get_dataframe()
breach = dataiku.Dataset("ews_ewi_breach_information").get_dataframe()
entity = dataiku.Dataset("ews_entity_static_data").get_dataframe()
exposure = dataiku.Dataset("ews_exposure_static_data").get_dataframe()

if len(reference) != 1:
    raise ValueError("Implementation_Reference_Date must contain one row")

reference_date = pd.to_datetime(reference["REFERENCE_DATE"].iloc[0]).normalize()

for df in [breach, entity, exposure]:
    df["ENTITY_ID"] = normalize_id(df["ENTITY_ID"])

results = []

# Output structure
check(
    "SEGMENT removed",
    all("SEGMENT" not in df.columns for df in [breach, entity, exposure]),
)
check("Breach IDs unique", breach["BREACH_ID"].notna().all() and breach["BREACH_ID"].is_unique)
check(
    "No duplicate EWIs",
    not breach.duplicated(["ENTITY_ID", "EWI_NAME"]).any(),
)

# Alert exclusions
business_line = (
    breach["BUSINESS_LINE"]
    .astype("string").str.strip()
    .str.replace(r"\s+", " ", regex=True).str.casefold()
)
cib = business_line.str.contains(
    r"^corporate\s*(?:(?:&|and)\s*)?institutional\s*banking$",
    regex=True, na=False,
)
check("CIB excluded", not cib.any(), f"{int(cib.sum())} remaining")

stage2_names = {
    "Downgrade from Stage 1 to Stage 2",
    "Connected client classified as Stage 2",
}
stage2 = breach["EWI_NAME"].astype("string").str.strip().isin(stage2_names)
check("Stage 2 excluded", not stage2.any(), f"{int(stage2.sum())} remaining")

exit_text = (
    breach[["RM", "BUSINESS_LINE", "TEAM"]]
    .astype("string").fillna("").agg(" ".join, axis=1)
    .str.upper().str.replace(r"[^A-Z0-9]+", " ", regex=True)
)
remediation = (
    exit_text.str.contains("KYC EXIT CIC", regex=False, na=False)
    | exit_text.str.contains("BIL REMEDIATION EXIT", regex=False, na=False)
)
check("Remediation excluded", not remediation.any(), f"{int(remediation.sum())} remaining")

# Employee exclusion
if breach.empty:
    remaining_employees = 0
else:
    employee_query = """
        SELECT DISTINCT TRIM(CAST(e.ENTITY_ID AS VARCHAR2(200))) AS ENTITY_ID
        FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_VW e
        WHERE TRIM(CAST(e.SECTOR AS VARCHAR2(50))) = '1002'
          AND e.TEK_END_DTE = DATE '9999-12-31'
          AND e.ENTITY_ID IS NOT NULL
    """
    employees = SQLExecutor2(
        dataset=dataiku.Dataset("ACCOUNT_VW_1")
    ).query_to_df(employee_query)
    employee_ids = set(normalize_id(employees["ENTITY_ID"]).dropna())
    remaining_employees = int(breach["ENTITY_ID"].isin(employee_ids).sum())

check("BIL employees excluded", remaining_employees == 0, f"{remaining_employees} remaining")

# Output consistency
breach_ids = set(breach["ENTITY_ID"].dropna())
entity_ids = set(entity["ENTITY_ID"].dropna())
exposure_ids = set(exposure["ENTITY_ID"].dropna())

check(
    "Entity static matches breaches",
    entity_ids == breach_ids and len(entity) == len(entity_ids),
)
check("Exposure entities match breaches", exposure_ids.issubset(breach_ids))

alert_dates = pd.to_datetime(breach["ALERT_DATE"], errors="coerce").dt.normalize()
check(
    "Alert date matches reference",
    breach.empty or (alert_dates.notna().all() and alert_dates.eq(reference_date).all()),
)

print("RETAIL EWS - DEV OUTPUT QA")
print("Reference date:", reference_date.strftime("%Y-%m-%d"))
print("Daily alert rows:", len(alerts))
print("Final breaches:", len(breach))
print("Final clients:", len(breach_ids))
print("Entity static rows:", len(entity))
print("Exposure static rows:", len(exposure))
print()

for name, status, detail in results:
    print(f"{status:4}  {name}" + (f" ({detail})" if detail else ""))

if any(status == "FAIL" for _, status, _ in results):
    raise AssertionError("DEV output QA failed")
