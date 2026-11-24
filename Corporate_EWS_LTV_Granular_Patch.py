# -*- coding: utf-8 -*-
"""Prepare a protected granular LTV backtest recipe from the existing script.

Run OFFLINE, not inside the Flow: python Corporate_EWS_LTV_Granular_Patch.py existing_recipe.py
Creates a NEW separate recipe code file without changing the original.
Does not execute Dataiku/Snowflake or certify LTV input/score coverage.
"""
from pathlib import Path
import re
import sys

SOURCE_NAME='results_Corporate_LTV'
NEW_NAME='results_Corporate_LTV_granular'  # technical comparison output; not client facing
PCTS=(85,90,95,100,105,110)
# Preserve ability to compare the historical 2/3-notch settings and add the
# partner's proposed 4-notch criterion as a distinct configuration.
RATING_NOTCHES=(2,3,4)


def build_grid():
    rows=[]
    for pct in PCTS:
        rows.append("('ltv_high_loan',%.2f,0)"%(pct/100.0))
    for pct in PCTS:
        for notch in RATING_NOTCHES:
            rows.append("('ltv_high_and_client_rating_downgrade',%.2f,%d)"%(pct/100.0,notch))
    return "GRID_RULES AS (\n SELECT COLUMN1::VARCHAR INDICATOR,COLUMN2::DOUBLE THRESHOLD_A,\n" + \
           "        COLUMN3::DOUBLE THRESHOLD_B FROM VALUES\n " + ",\n ".join(rows) + "\n),\nRAW_SIGNALS AS ("


def patch(text):
    if 'GRID_RULES AS (' not in text or 'RAW_SIGNALS AS (' not in text:
        raise RuntimeError('Expected original GRID_RULES and RAW_SIGNALS CTEs were not found')
    grid=re.compile(r'(?ms)^GRID_RULES AS \(\n.*?^\),\nRAW_SIGNALS AS \(')
    new,n=grid.subn(lambda _:build_grid(),text)
    if n!=1:raise RuntimeError('Found %s matching grids; require exactly one'%n)
    old_output="OUTPUT='"+SOURCE_NAME+"'"
    if new.count(old_output)!=1:
        raise RuntimeError('Expected exactly one original OUTPUT declaration')
    new=new.replace(old_output,"OUTPUT='"+NEW_NAME+"'",1)
    if ('m.LTV_VALUE>=g.THRESHOLD_A' not in new or
        'm.RATING_DOWNGRADE_3M>=g.THRESHOLD_B' not in new):
        raise RuntimeError('Unexpected signal logic: no safe patch possible')
    return new


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python Corporate_EWS_LTV_Granular_Patch.py existing_ltv_recipe.py')
    source=Path(sys.argv[1]);original=source.read_text(encoding='utf-8-sig')
    rewritten=patch(original)
    import ast
    ast.parse(rewritten)
    dest=source.with_name(source.stem+'_granular.py')
    if dest.exists():raise RuntimeError('Refusing to overwrite existing code: '+str(dest))
    dest.write_text(rewritten,encoding='utf-8')
    print('CREATED',dest)
    print('GRID',len(PCTS),'LTV-only x 6 portfolios plus',len(PCTS)*len(RATING_NOTCHES),'LTV+rating x 6 portfolios')
    print('REQUIRES: a NEW Dataiku output '+NEW_NAME+'; input LTV coverage and rating QA first.')
    print('NOT INCLUDED: ADC/IPRE slotting combination pending verified score/loan/date mapping.')

if __name__=='__main__':main()
