# -*- coding: utf-8 -*-
"""Align the existing recurring-payments backtest with complete monthly history."""

import ast

import dataiku

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
RECIPE = 'compute_results_Corporate_Recurring_Payments'
OUTPUT = 'results_Corporate_Recurring_Payments'
APPLY_CHANGES = False

project = dataiku.api_client().get_project(PROJECT)
settings = project.get_recipe(RECIPE).get_settings()
if settings.type != 'python':
    raise RuntimeError('Unexpected recipe type')

outputs = {
    str(item['ref']).split('.')[-1]
    for group in settings.get_recipe_outputs().values()
    for item in group.get('items', [])
}
if outputs != {OUTPUT}:
    raise RuntimeError('Unexpected backtest output')

code = settings.get_code()
old_condition = (
    "if source_info.empty or nonnegative_count(source_info, 'N_MONTHS_WITH_COMPLETE_SOURCE')<7 "
    "or nonnegative_count(source_info, 'N_PAYER_HASHES')==0:"
)
complete_condition = (
    "if source_info.empty or nonnegative_count(source_info, 'N_MONTHS')!=25 "
    "or nonnegative_count(source_info, 'N_MONTHS_WITH_COMPLETE_SOURCE')!=25 "
    "or nonnegative_count(source_info, 'N_PAYER_HASHES')==0:"
)

if complete_condition in code:
    print('The backtest already requires 25 complete months.')
elif code.count(old_condition) == 1:
    updated = code.replace(old_condition, complete_condition, 1)
    ast.parse(updated, filename=RECIPE)
    if APPLY_CHANGES:
        settings.set_code(updated)
        settings.save()
        print('The backtest completeness check has been updated.')
    else:
        print('Change available: require 25 complete months before the backtest.')
        print('Set APPLY_CHANGES=True after approval to save the updated recipe.')
else:
    raise RuntimeError('The current backtest has an unexpected completeness rule.')

print('The existing indicators, portfolio logic, target and cost formulas are preserved.')
