Retail EWS — current-date preparation recipes

This package contains eight separate Python recipes with short, client-friendly English comments.
Their existing data processing and SQL logic has not been intentionally altered.

Validation
- Seven complete attached recipes were compared with the uploaded sources using Python syntax trees and executable tokens. Both are identical.
- Cash_Deposits was transcribed from the code pasted in the conversation and checked for Python syntax.
- All eight files pass local Python compilation.
- None of these files has been run against Dataiku or the source databases.

The user also pasted the beginning of Salary_Daily_Entity. The code stopped inside resolve_column and therefore is not included as a complete recipe. It requires the complete original source to annotate safely.

The four uploaded copies of Neobank_Transfers_CC_Limit had identical content and have been included only once.

Files
- cash_advance_data_MAIN_CLEAN.py
- CC_Transactions_Neobanks_MAIN_CLEAN.py
- Cash_Advance_CC_Limit_MAIN_CLEAN.py
- RFO_PD_loan_MAIN_CLEAN.py
- Loan_Exposure_Current_MAIN_CLEAN.py
- Credit_Cards_MAIN_CLEAN.py
- Neobank_Transfers_CC_Limit_MAIN_CLEAN.py
- Cash_Deposits_MAIN_CLEAN.py

Business flow overview
- RFO_PD_loan: most recent reporting snapshot of loan exposure and risk attributes.
- Loan_Exposure_Current: classified mortgage, non-mortgage and Lombard exposure by entity.
- Credit_Cards: one contribution per card with a chosen credit limit and outstanding balance.
- cash_advance_data: mapped credit-card cash advances on the reference date.
- Cash_Advance_CC_Limit: 94% / 97% utilisation crossings associated with same-day cash advances.
- CC_Transactions_Neobanks: daily neobank top-up activity.
- Neobank_Transfers_CC_Limit: utilisation around neobank transfers on the reference and previous dates.
- Cash_Deposits: teller deposit count and amount by entity.

Notes for QA
- Several recipes depend on entity/customer mapping data from Credit Insight. In the current diagnostic, ENTITY_CUSTOMER_VW had zero records considered valid on 22-23 October 2026. Annotation does not correct the upstream data availability issue.
- Loan_Exposure_Current reads RFO exposure within the calendar month, and checks that usable exposure is available. Check the reporting month before running it for a new period.
- These files are for review and have not been validated in a live run.
