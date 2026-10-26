Retail EWS - previous snapshot preparation recipes

All recipes retain their existing calculations, source tables, dataset names and control checks.
The eight uploaded recipes were compared with their sources using Python AST and tokens.
The three short recipes were reproduced from the matching current-day versions, updating the names and messages exactly as provided in the user message.
The files pass Python syntax validation, but the recipes have not been executed against Dataiku sources.

Recipes:
vars_loan_Previous_MAIN_CLEAN.py
entity_level_merged_Previous_MAIN_CLEAN.py
vars_entity_level_merged_Previous_MAIN_CLEAN.py
GEC_Exposure_Current_Previous_MAIN_CLEAN.py
DPD_Arrears_Daily_Entity_Previous_MAIN_CLEAN.py
vars_loan_level_Previous_MAIN_CLEAN.py
Retail_Portfolio_Daily_Previous_MAIN_CLEAN.py
Entity_Information_Previous_MAIN_CLEAN.py
Customer_Balances_Previous_MAIN_CLEAN.py
vars_behavioural_daily_Previous_MAIN_CLEAN.py
vars_payment_level_Previous_MAIN_CLEAN.py

Checks to perform in Dataiku:
1. Confirm the previous reference date and its relationship to the current reference date.
2. Review row counts, duplicates, and source freshness on each output.
3. Compare Retail population, RRE / Other Retail membership and group exposure between dates.
4. Verify DPD / arrears, payment activity and card utilisation with source records.
5. Confirm runtime and results before production use.
