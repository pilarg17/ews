# Retail EWS — latest preparation recipes (Current and Previous)

Six files are included. Four are newly commented in this batch; two were already commented and are re-included for convenience.

## Files

- `Loan_Exposure_Current_Previous_MAIN_CLEAN.py` — New; executable logic preserved.
- `cash_advance_data_Previous_MAIN_CLEAN.py` — New; executable logic preserved.
- `CC_Transactions_Neobanks_Previous_MAIN_CLEAN.py` — New; executable logic preserved.
- `Cash_Advance_CC_Limit_Previous_MAIN_CLEAN.py` — New; executable logic preserved.
- `RFO_PD_loan_Previous_MAIN_CLEAN.py` — Already delivered; reverified equivalent.
- `Neobank_Transfers_CC_Limit_MAIN_CLEAN.py` — Already delivered; reverified equivalent.

## Validation

All six scripts passed Python AST parsing and compilation.
ASTs and non-comment Python tokens of each delivered file were compared with its corresponding latest submitted source: equal in all six cases. No executable Python instructions or embedded SQL were changed.
These are static checks. Runtime success and output quality in Dataiku have not been tested.

## Not included

Salary_Daily_Entity and Salary_Daily_Entity_Previous are excluded because salary processing is currently out of scope.

## Important

The known Credit Insight temporal-availability incident for 22–23 October 2026 is not fixed or bypassed by these documentation-only edits. Do not treat a successful syntax check as evidence that the daily scenario will succeed.
