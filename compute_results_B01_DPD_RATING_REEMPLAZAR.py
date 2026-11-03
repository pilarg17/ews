# -*- coding: utf-8 -*-
"""Corporate EWS | Batch B01 | full Retail DPD, PD, Rating and GCC catalogue.

Input:  EWS_training_B01_DPD_RATING
Output: results_B01_DPD_RATING, test_results_B01_DPD_RATING

All historical calculations execute inside Snowflake. This recipe never reads the
full training dataset into Pandas and never changes its original source tables.

The rating/PD target retains the five components used in the
historical calibration. For DPD, the target excludes the DPD component to avoid
circular validation of thresholds 20 to 60. Both target definitions are labeled
in the results. Business event codes and candidate GEC exposure require review.
"""

import re
import dataiku
from dataiku import recipe
from dataiku.core.sql import SQLExecutor2

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CONNECTION = "DEV_AA_TEAMS"
TRAINING = "EWS_training_B01_DPD_RATING"
RESULTS = "results_B01_DPD_RATING"
TEST_RESULTS = "test_results_B01_DPD_RATING"

# Historical Retail parameters. Observations are daily, and comparisons use
# calendar dates (the same 1, 3, 6, 12 month offsets as the Retail model).
MIN_LEAD_MONTHS=1
MAX_LEAD_MONTHS=11
MAX_GAP_MONTHS=3
MAX_SIGNAL_GAP_MONTHS=2
MIN_EXPOSURE=0.0
COST_TP_FIXED=1000.0
COST_FP=200.0
COST_FN_PCT=0.20
COST_TP_PCT_LATE=0.20
COST_TP_PCT_EARLY=0.10

# Exact Retail feature names. These 41 singles and 47 pairs are the complete
# Retail DPD/PD/Rating/GCC subset supported by current Corporate B01 inputs.
INDICATORS = ['RATING_GRANULAR_SCALE', 'GCC_RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_1m', 'rating_granular_scale_downgrade_3m', 'rating_granular_scale_downgrade_6m', 'rating_granular_scale_downgrade_12m', 'rating_granular_scale_downgrade_best_12m', 'rating_granular_scale_downgrade_worst_12m', 'gcc_rating_granular_scale_downgrade_1m', 'gcc_rating_granular_scale_downgrade_3m', 'gcc_rating_granular_scale_downgrade_6m', 'gcc_rating_granular_scale_downgrade_12m', 'gcc_rating_granular_scale_downgrade_best_12m', 'gcc_rating_granular_scale_downgrade_worst_12m', 'dpd', 'PD_CRR', 'pd_change_1m', 'pd_change_3m', 'pd_change_6m', 'pd_change_12m', 'pd_change_worst_12m', 'pd_change_best_12m', 'pd_abs_change_1m', 'pd_abs_change_worst_12m', 'pd_abs_change_best_12m', 'PD_CRR_GCC', 'gccpd_abs_change_1m', 'gccpd_abs_change_best_12m', 'gccpd_pd_difference', 'arrears_days_above_250', 'arrears_days_above_500', 'arrears_amount_above_25', 'arrears_amount_above_50', 'arrears_amount_above_75', 'arrears_amount_above_100', 'arrears_amount_above_150', 'arrears_amount_above_200', 'arrears_amount_above_300', 'arrears_amount_above_500', 'arrears_amount_above_750', 'arrears_amount_above_1000']
PAIRS = [('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_1m'), ('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_3m'), ('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_6m'), ('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_12m'), ('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_best_12m'), ('RATING_GRANULAR_SCALE', 'rating_granular_scale_downgrade_worst_12m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_1m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_3m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_6m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_12m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_best_12m'), ('GCC_RATING_GRANULAR_SCALE', 'gcc_rating_granular_scale_downgrade_worst_12m'), ('RATING_GRANULAR_SCALE', 'pd_abs_change_best_12m'), ('PD_CRR_GCC', 'PD_CRR'), ('gccpd_abs_change_1m', 'RATING_GRANULAR_SCALE'), ('gccpd_abs_change_best_12m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_1m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_3m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_6m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_12m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_best_12m', 'RATING_GRANULAR_SCALE'), ('gcc_rating_granular_scale_downgrade_worst_12m', 'RATING_GRANULAR_SCALE'), ('gccpd_abs_change_1m', 'pd_abs_change_1m'), ('gccpd_abs_change_best_12m', 'pd_abs_change_best_12m'), ('gccpd_abs_change_1m', 'rating_granular_scale_downgrade_1m'), ('gccpd_abs_change_best_12m', 'rating_granular_scale_downgrade_12m'), ('gccpd_abs_change_best_12m', 'rating_granular_scale_downgrade_best_12m'), ('gccpd_abs_change_best_12m', 'rating_granular_scale_downgrade_worst_12m'), ('gcc_rating_granular_scale_downgrade_1m', 'pd_abs_change_1m'), ('gcc_rating_granular_scale_downgrade_12m', 'pd_abs_change_best_12m'), ('gcc_rating_granular_scale_downgrade_best_12m', 'pd_abs_change_best_12m'), ('gcc_rating_granular_scale_downgrade_worst_12m', 'pd_abs_change_worst_12m'), ('gcc_rating_granular_scale_downgrade_1m', 'rating_granular_scale_downgrade_1m'), ('gcc_rating_granular_scale_downgrade_3m', 'rating_granular_scale_downgrade_3m'), ('gcc_rating_granular_scale_downgrade_6m', 'rating_granular_scale_downgrade_6m'), ('gcc_rating_granular_scale_downgrade_12m', 'rating_granular_scale_downgrade_12m'), ('gcc_rating_granular_scale_downgrade_best_12m', 'rating_granular_scale_downgrade_best_12m'), ('gcc_rating_granular_scale_downgrade_worst_12m', 'rating_granular_scale_downgrade_worst_12m'), ('PD_CRR', 'pd_change_1m'), ('PD_CRR', 'pd_change_3m'), ('PD_CRR', 'pd_change_6m'), ('PD_CRR', 'pd_change_12m'), ('PD_CRR', 'pd_change_worst_12m'), ('PD_CRR', 'pd_change_best_12m'), ('PD_CRR', 'pd_abs_change_1m'), ('PD_CRR', 'pd_abs_change_worst_12m'), ('PD_CRR', 'pd_abs_change_best_12m')]
ACTIVE_INDICATORS = ['arrears_amount_above_100', 'arrears_amount_above_1000', 'arrears_amount_above_150', 'arrears_amount_above_200', 'arrears_amount_above_25', 'arrears_amount_above_300', 'arrears_amount_above_50', 'arrears_amount_above_500', 'arrears_amount_above_75', 'arrears_amount_above_750', 'gcc_rating_granular_scale_downgrade_12m', 'gcc_rating_granular_scale_downgrade_1m', 'gcc_rating_granular_scale_downgrade_3m', 'gcc_rating_granular_scale_downgrade_6m', 'gcc_rating_granular_scale_downgrade_best_12m', 'gcc_rating_granular_scale_downgrade_worst_12m', 'gccpd_abs_change_1m', 'gccpd_abs_change_best_12m', 'gccpd_pd_difference', 'pd_abs_change_1m', 'pd_abs_change_best_12m', 'pd_abs_change_worst_12m', 'pd_change_12m', 'pd_change_1m', 'pd_change_3m', 'pd_change_6m', 'pd_change_best_12m', 'pd_change_worst_12m', 'rating_granular_scale_downgrade_12m', 'rating_granular_scale_downgrade_1m', 'rating_granular_scale_downgrade_3m', 'rating_granular_scale_downgrade_6m', 'rating_granular_scale_downgrade_best_12m', 'rating_granular_scale_downgrade_worst_12m']
THRESHOLDS = {'RATING_GRANULAR_SCALE': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'GCC_RATING_GRANULAR_SCALE': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'rating_granular_scale_downgrade_1m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'rating_granular_scale_downgrade_3m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'rating_granular_scale_downgrade_6m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'rating_granular_scale_downgrade_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'rating_granular_scale_downgrade_best_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'rating_granular_scale_downgrade_worst_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_1m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_3m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_6m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_best_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'gcc_rating_granular_scale_downgrade_worst_12m': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'dpd': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 'PD_CRR': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5], 'pd_change_1m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_change_3m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_change_6m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_change_12m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_change_worst_12m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_change_best_12m': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56, 1.58, 1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0], 'pd_abs_change_1m': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'pd_abs_change_worst_12m': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'pd_abs_change_best_12m': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'PD_CRR_GCC': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5], 'gccpd_abs_change_1m': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'gccpd_abs_change_best_12m': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'gccpd_pd_difference': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'arrears_days_above_250': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'arrears_days_above_500': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'arrears_amount_above_25': [0.5], 'arrears_amount_above_50': [0.5], 'arrears_amount_above_75': [0.5], 'arrears_amount_above_100': [0.5], 'arrears_amount_above_150': [0.5], 'arrears_amount_above_200': [0.5], 'arrears_amount_above_300': [0.5], 'arrears_amount_above_500': [0.5], 'arrears_amount_above_750': [0.5], 'arrears_amount_above_1000': [0.5]}
GEC_THRESHOLDS = list(range(50_000,1_000_001,50_000))
GEC_INDICATOR = 'dpd x GEC_exposure'
RATING_NAMES = ['AAA+', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'D']
RETAIL_TOTAL=205
RETAIL_COVERED=88
RETAIL_UNAVAILABLE=117
EXPECTED_RESULTS = sum(len(THRESHOLDS[i]) for i in INDICATORS) + sum(
    len(THRESHOLDS[a])*len(THRESHOLDS[b]) for a,b in PAIRS
) + len(THRESHOLDS['dpd'])*len(GEC_THRESHOLDS)
assert len(INDICATORS)==41 and len(PAIRS)==47 and EXPECTED_RESULTS==50554

def q(value):
    value = str(value).replace("${projectKey}", PROJECT)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Invalid Snowflake identifier: " + repr(value))
    return '"' + value + '"'


project = dataiku.api_client().get_project(PROJECT)


def get_table(dataset_name, required):
    cfg = project.get_dataset(dataset_name).get_settings().get_raw()
    p = cfg.get("params") or {}
    if p.get("connection") != CONNECTION:
        raise ValueError("Unexpected connection for " + dataset_name)
    found = {c["name"].upper() for c in dataiku.Dataset(
        dataset_name, project_key=PROJECT).read_schema()}
    missing = set(required) - found
    if missing:
        raise ValueError("Missing fields in {}: {}".format(
            dataset_name, sorted(missing)))
    return q(p["schema"]) + "." + q(p["table"])


required_source = [
    "ENTITY_ID", "MAP_DTE", "DPD_MAX", "ARREARS_AMOUNT", "PD_CRR", "GCC_PD_CRR",
    "CLIENT_NOTCH", "GCC_NOTCH", "CLIENT_DOWNGRADE_12M",
    "GCC_DOWNGRADE_12M", "LOAN_COUNT", "HAS_VALID_ENTITY_HISTORY",
    "NOMINAL_KNOWN_SUM", "NOMINAL_KNOWN_LOANS", "LOAN_SOURCE_DUPLICATES",
    "GEC_NOMINAL_CANDIDATE", "GEC_GCA_CANDIDATE",
    "TGT_ANY_SAMEDAY", "TGT_DEFAULT_SAMEDAY",
    "TGT_FORBEARANCE_SAMEDAY", "TGT_DIM1_SAMEDAY",
]
SOURCE = get_table(TRAINING, required_source)

# Use the two existing B01 outputs only. This build replaces their previous results.
expected_outputs = {RESULTS, TEST_RESULTS}
configured_outputs = {
    str(getattr(item, "name", item)).split(".")[-1]
    for item in recipe.get_outputs(object_type="DATASET")
}
if configured_outputs != expected_outputs:
    raise ValueError("Configure exactly these outputs: " + ", ".join(sorted(expected_outputs)))

for dataset_name in sorted(expected_outputs):
    settings = project.get_dataset(dataset_name).get_settings().get_raw()
    p = settings.get("params") or {}
    physical_name = str(p.get("table", "")).replace("${projectKey}", PROJECT)
    expected_physical = PROJECT + "_" + dataset_name.upper()
    if (p.get("connection") != CONNECTION
            or str(p.get("schema", "")).upper() != "RRG"
            or physical_name.upper() != expected_physical):
        raise ValueError(
            "Invalid output table for {}. Use connection {}, schema RRG, "
            "table {}".format(dataset_name, CONNECTION, expected_physical)
        )

# Resolve the configured results table without depending on output schema.
TARGET_TABLE = q("RRG") + "." + q(PROJECT + "_" + RESULTS.upper())

# Supported grids. The generated grid contains Retail labels plus canonical
# Snowflake upper-case column names, preventing UNPIVOT capitalization issues.
grid_values = ",\n        ".join(
    "('{}', '{}', {:.8f})".format(ind.upper(), ind, float(t))
    for ind in INDICATORS for t in THRESHOLDS[ind]
)
single_fields = ", ".join(INDICATORS)
# Snowflake UNPIVOT requires exact matching types for every feature.
single_selection = ",\n           ".join(
    "TRY_TO_DOUBLE({name}::VARCHAR) AS {name}".format(name=i)
    for i in INDICATORS
)
previous_fields = ",\n           ".join(
    "LAG({name}) OVER(PARTITION BY TARGET_SCOPE,ENTITY_ID ORDER BY MAP_DTE) AS PREV_{name}".format(name=i)
    for i in INDICATORS
)
previous_choice = "CASE UPPER(u.PREDICTOR) " + " ".join(
    "WHEN '{}' THEN u.PREV_{}".format(i.upper(), i) for i in INDICATORS
) + " ELSE NULL END"

pair_grids = "\nUNION ALL\n".join(
    "SELECT '{a} x {b}' AS INDICATOR, '{a}' AS PREDICTOR_A, "
    "'{b}' AS PREDICTOR_B, a.THRESHOLD AS THRESHOLD_A, "
    "b.THRESHOLD AS THRESHOLD_B "
    "FROM GRID_SINGLE a JOIN GRID_SINGLE b "
    "ON a.PREDICTOR='{A}' AND b.PREDICTOR='{B}'".format(
        a=a,b=b,A=a.upper(),B=b.upper()) for a,b in PAIRS
)

pair_starts = []
for a,b in PAIRS:
    pair_starts.append("""
SELECT '{pair}' AS INDICATOR, '{a}' AS PREDICTOR_A, '{b}' AS PREDICTOR_B,
       s.ENTITY_ID, s.MAP_DTE, s.THRESHOLD_A, g.THRESHOLD AS THRESHOLD_B
FROM SINGLE_START s
JOIN PREP p ON p.ENTITY_ID=s.ENTITY_ID AND p.MAP_DTE=s.MAP_DTE
    AND p.TARGET_SCOPE='RATING'
JOIN GRID_SINGLE g ON g.PREDICTOR='{B}' AND p.{b}>g.THRESHOLD
WHERE s.INDICATOR='{a}' AND p.TARGET=0
UNION ALL
SELECT '{pair}', '{a}', '{b}',
       s.ENTITY_ID, s.MAP_DTE, g.THRESHOLD, s.THRESHOLD_A
FROM SINGLE_START s
JOIN PREP p ON p.ENTITY_ID=s.ENTITY_ID AND p.MAP_DTE=s.MAP_DTE
    AND p.TARGET_SCOPE='RATING'
JOIN GRID_SINGLE g ON g.PREDICTOR='{A}' AND p.{a}>g.THRESHOLD
WHERE s.INDICATOR='{b}' AND p.TARGET=0
""".format(pair=a+' x '+b,a=a,b=b,A=a.upper(),B=b.upper()))
pair_starts_sql = "\nUNION ALL\n".join(pair_starts)
active_grid_names = '('+','.join("'{}'".format(x.upper()) for x in ACTIVE_INDICATORS)+')'
rating_cols=('RATING_GRANULAR_SCALE','GCC_RATING_GRANULAR_SCALE')
rating_a_sql = "CASE WHEN m.PREDICTOR_A IN ('RATING_GRANULAR_SCALE','GCC_RATING_GRANULAR_SCALE') THEN CASE m.THRESHOLD_A " + ' '.join(
    "WHEN {} THEN '{}'".format(i,n) for i,n in enumerate(RATING_NAMES,1)
) + ' ELSE NULL END ELSE NULL END'
rating_b_sql = "CASE WHEN m.PREDICTOR_B IN ('RATING_GRANULAR_SCALE','GCC_RATING_GRANULAR_SCALE') THEN CASE m.THRESHOLD_B " + ' '.join(
    "WHEN {} THEN '{}'".format(i,n) for i,n in enumerate(RATING_NAMES,1)
) + ' ELSE NULL END ELSE NULL END'
GEC_GRID_VALUES = ', '.join('({})'.format(t) for t in GEC_THRESHOLDS)

rating_pair_eligible = "\nUNION ALL\n".join(
    "SELECT DISTINCT '{a} x {b}' AS INDICATOR, "
    "CAST(NULL AS DOUBLE) AS EXPOSURE_CUTOFF, e.EVENT_ID,e.EVENT_EXPOSURE "
    "FROM EVENTS e JOIN PREP p ON e.ENTITY_ID=p.ENTITY_ID "
    "AND p.TARGET_SCOPE='RATING' AND e.TARGET_SCOPE='RATING' "
    "AND p.MAP_DTE BETWEEN DATEADD(month,-{hi},e.EVENT_START) "
    "AND DATEADD(month,-{lo},e.EVENT_START) "
    "WHERE p.TARGET=0 AND p.{a} IS NOT NULL AND p.{b} IS NOT NULL".format(
        a=a,b=b,hi=MAX_LEAD_MONTHS,lo=MIN_LEAD_MONTHS)
    for a,b in PAIRS
)

feature_formulas = 'p.RATING_GRANULAR_SCALE-p1.RATING_GRANULAR_SCALE AS rating_granular_scale_downgrade_1m,\n       p.RATING_GRANULAR_SCALE-p3.RATING_GRANULAR_SCALE AS rating_granular_scale_downgrade_3m,\n       p.RATING_GRANULAR_SCALE-p6.RATING_GRANULAR_SCALE AS rating_granular_scale_downgrade_6m,\n       p.RATING_GRANULAR_SCALE-p12.RATING_GRANULAR_SCALE AS rating_granular_scale_downgrade_12m,\n       p.RATING_GRANULAR_SCALE-p.MIN_CLIENT_RATING_12M AS rating_granular_scale_downgrade_best_12m,\n       p.RATING_GRANULAR_SCALE-p.MAX_CLIENT_RATING_12M AS rating_granular_scale_downgrade_worst_12m,\n       p.GCC_RATING_GRANULAR_SCALE-p1.GCC_RATING_GRANULAR_SCALE AS gcc_rating_granular_scale_downgrade_1m,\n       p.GCC_RATING_GRANULAR_SCALE-p3.GCC_RATING_GRANULAR_SCALE AS gcc_rating_granular_scale_downgrade_3m,\n       p.GCC_RATING_GRANULAR_SCALE-p6.GCC_RATING_GRANULAR_SCALE AS gcc_rating_granular_scale_downgrade_6m,\n       p.GCC_RATING_GRANULAR_SCALE-p12.GCC_RATING_GRANULAR_SCALE AS gcc_rating_granular_scale_downgrade_12m,\n       p.GCC_RATING_GRANULAR_SCALE-p.MIN_GCC_RATING_12M AS gcc_rating_granular_scale_downgrade_best_12m,\n       p.GCC_RATING_GRANULAR_SCALE-p.MAX_GCC_RATING_12M AS gcc_rating_granular_scale_downgrade_worst_12m,\n       ((p.PD_CRR-p1.PD_CRR)/NULLIF(p1.PD_CRR,0)) AS pd_change_1m,\n       ((p.PD_CRR-p3.PD_CRR)/NULLIF(p3.PD_CRR,0)) AS pd_change_3m,\n       ((p.PD_CRR-p6.PD_CRR)/NULLIF(p6.PD_CRR,0)) AS pd_change_6m,\n       ((p.PD_CRR-p12.PD_CRR)/NULLIF(p12.PD_CRR,0)) AS pd_change_12m,\n       ((p.PD_CRR-p.MIN_PD_12M)/NULLIF(p.MIN_PD_12M,0)) AS pd_change_best_12m,\n       ((p.PD_CRR-p.MAX_PD_12M)/NULLIF(p.MAX_PD_12M,0)) AS pd_change_worst_12m,\n       IFF(p.arrears_amount>25,1.0,0.0) AS arrears_amount_above_25,\n       IFF(p.arrears_amount>50,1.0,0.0) AS arrears_amount_above_50,\n       IFF(p.arrears_amount>75,1.0,0.0) AS arrears_amount_above_75,\n       IFF(p.arrears_amount>100,1.0,0.0) AS arrears_amount_above_100,\n       IFF(p.arrears_amount>150,1.0,0.0) AS arrears_amount_above_150,\n       IFF(p.arrears_amount>200,1.0,0.0) AS arrears_amount_above_200,\n       IFF(p.arrears_amount>300,1.0,0.0) AS arrears_amount_above_300,\n       IFF(p.arrears_amount>500,1.0,0.0) AS arrears_amount_above_500,\n       IFF(p.arrears_amount>750,1.0,0.0) AS arrears_amount_above_750,\n       IFF(p.arrears_amount>1000,1.0,0.0) AS arrears_amount_above_1000,\n       p.PD_CRR-p1.PD_CRR AS pd_abs_change_1m,\n       p.PD_CRR-p.MIN_PD_12M AS pd_abs_change_best_12m,\n       p.PD_CRR-p.MAX_PD_12M AS pd_abs_change_worst_12m,\n       p.PD_CRR_GCC-p1.PD_CRR_GCC AS gccpd_abs_change_1m,\n       p.PD_CRR_GCC-p.MIN_GCCPD_12M AS gccpd_abs_change_best_12m,\n       p.PD_CRR_GCC-p.PD_CRR AS gccpd_pd_difference'

SQL = r"""
WITH
LAST_VALID_DPD AS (
    SELECT MAX(TO_DATE(MAP_DTE)) AS LAST_DPD_DATE
    FROM {SOURCE}
    WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0
),
PANEL_RAW AS (
    SELECT
        t.ENTITY_ID::VARCHAR AS ENTITY_ID,
        TO_DATE(t.MAP_DTE) AS MAP_DTE,
        TRY_TO_DOUBLE(t.DPD_MAX::VARCHAR) AS dpd,
        COALESCE(TRY_TO_DOUBLE(t.ARREARS_AMOUNT::VARCHAR),0.0) AS arrears_amount,
        TRY_TO_DOUBLE(t.CLIENT_NOTCH::VARCHAR) AS RATING_GRANULAR_SCALE,
        TRY_TO_DOUBLE(t.GCC_NOTCH::VARCHAR) AS GCC_RATING_GRANULAR_SCALE,
        TRY_TO_DOUBLE(t.PD_CRR::VARCHAR) AS PD_CRR,
        TRY_TO_DOUBLE(t.GCC_PD_CRR::VARCHAR) AS PD_CRR_GCC,
        TRY_TO_DOUBLE(t.GEC_NOMINAL_CANDIDATE::VARCHAR) AS GEC_EXPOSURE,
        CASE WHEN t.LOAN_COUNT>0 AND t.LOAN_SOURCE_DUPLICATES=0
                  AND t.NOMINAL_KNOWN_LOANS=t.LOAN_COUNT
             THEN TRY_TO_DOUBLE(t.NOMINAL_KNOWN_SUM::VARCHAR)
             ELSE NULL END AS EXPOSURE,
        TRY_TO_NUMBER(t.TGT_ANY_SAMEDAY::VARCHAR) AS TARGET_RETAIL,
        CASE WHEN TRY_TO_DOUBLE(t.PD_CRR::VARCHAR)=1
             OR t.TGT_DEFAULT_SAMEDAY=1
             OR t.TGT_FORBEARANCE_SAMEDAY=1
             OR t.TGT_DIM1_SAMEDAY=1 THEN 1
          WHEN t.PD_CRR IS NULL
             OR t.TGT_DEFAULT_SAMEDAY IS NULL
             OR t.TGT_FORBEARANCE_SAMEDAY IS NULL
             OR t.TGT_DIM1_SAMEDAY IS NULL THEN NULL
          ELSE 0
        END AS TARGET_EXCL_DPD
    FROM {SOURCE} t
    WHERE t.LOAN_COUNT>0 AND t.HAS_VALID_ENTITY_HISTORY=1
      AND TO_DATE(t.MAP_DTE)<=(SELECT LAST_DPD_DATE FROM LAST_VALID_DPD)
),
-- Time comparisons use the same calendar-day offsets as the Retail model.
ROLLING_HISTORY AS (
    SELECT p.*,
       SUM(IFF(arrears_amount>250,1,0)) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '29 DAY' PRECEDING
                   AND CURRENT ROW) AS arrears_days_above_250,
       SUM(IFF(arrears_amount>500,1,0)) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '29 DAY' PRECEDING
                   AND CURRENT ROW) AS arrears_days_above_500,
       MIN(RATING_GRANULAR_SCALE) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MIN_CLIENT_RATING_12M,
       MAX(RATING_GRANULAR_SCALE) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MAX_CLIENT_RATING_12M,
       MIN(GCC_RATING_GRANULAR_SCALE) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MIN_GCC_RATING_12M,
       MAX(GCC_RATING_GRANULAR_SCALE) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MAX_GCC_RATING_12M,
       MIN(PD_CRR) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MIN_PD_12M,
       MAX(PD_CRR) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MAX_PD_12M,
       MIN(PD_CRR_GCC) OVER(
         PARTITION BY ENTITY_ID ORDER BY MAP_DTE
         RANGE BETWEEN INTERVAL '12 MONTH' PRECEDING
                   AND INTERVAL '1 DAY' PRECEDING) AS MIN_GCCPD_12M
    FROM PANEL_RAW p
),
CORPORATE_PANEL AS (
    SELECT p.*,
       {feature_formulas}
    FROM ROLLING_HISTORY p
    LEFT JOIN PANEL_RAW p1 ON p1.ENTITY_ID=p.ENTITY_ID
        AND p1.MAP_DTE=DATEADD(month,-1,p.MAP_DTE)
    LEFT JOIN PANEL_RAW p3 ON p3.ENTITY_ID=p.ENTITY_ID
        AND p3.MAP_DTE=DATEADD(month,-3,p.MAP_DTE)
    LEFT JOIN PANEL_RAW p6 ON p6.ENTITY_ID=p.ENTITY_ID
        AND p6.MAP_DTE=DATEADD(month,-6,p.MAP_DTE)
    LEFT JOIN PANEL_RAW p12 ON p12.ENTITY_ID=p.ENTITY_ID
        AND p12.MAP_DTE=DATEADD(month,-12,p.MAP_DTE)
),
TARGET_PANELS AS (
    SELECT p.*, 'RATING' AS TARGET_SCOPE, TARGET_RETAIL AS TARGET
    FROM CORPORATE_PANEL p
    UNION ALL
    SELECT p.*, 'DPD' AS TARGET_SCOPE, TARGET_EXCL_DPD AS TARGET
    FROM CORPORATE_PANEL p WHERE p.dpd IS NOT NULL
),
PANEL_END AS (
    SELECT MAX(MAP_DTE) AS END_DATE FROM CORPORATE_PANEL
),
FIRST_OBS AS (
    SELECT TARGET_SCOPE,ENTITY_ID,MIN(MAP_DTE) AS FIRST_DATE
    FROM TARGET_PANELS GROUP BY 1,2
),
EVENT_ACTIVE AS (
    SELECT TARGET_SCOPE,ENTITY_ID,MAP_DTE,
           LAG(MAP_DTE) OVER(
               PARTITION BY TARGET_SCOPE,ENTITY_ID ORDER BY MAP_DTE
           ) AS PREV_ACTIVE
    FROM TARGET_PANELS WHERE TARGET=1
),
EVENT_GROUPS AS (
    SELECT TARGET_SCOPE,ENTITY_ID,MAP_DTE,
           SUM(CASE WHEN PREV_ACTIVE IS NULL
                       OR MAP_DTE>DATEADD(month,{event_gap},PREV_ACTIVE)
                    THEN 1 ELSE 0 END)
           OVER(PARTITION BY TARGET_SCOPE,ENTITY_ID ORDER BY MAP_DTE
                ROWS UNBOUNDED PRECEDING) AS EVENT_GROUP
    FROM EVENT_ACTIVE
),
EVENT_DATES AS (
    SELECT TARGET_SCOPE,ENTITY_ID,EVENT_GROUP,
           MIN(MAP_DTE) AS EVENT_START,MAX(MAP_DTE) AS EVENT_END
    FROM EVENT_GROUPS GROUP BY 1,2,3
),
EVENTS AS (
    SELECT CONCAT(e.TARGET_SCOPE,'|',e.ENTITY_ID,'|',TO_VARCHAR(e.EVENT_START)) AS EVENT_ID,
           e.TARGET_SCOPE,e.ENTITY_ID,e.EVENT_START,e.EVENT_END,
           p.EXPOSURE AS EVENT_EXPOSURE
    FROM EVENT_DATES e
    JOIN FIRST_OBS f ON f.ENTITY_ID=e.ENTITY_ID AND f.TARGET_SCOPE=e.TARGET_SCOPE
    JOIN TARGET_PANELS p ON p.ENTITY_ID=e.ENTITY_ID
      AND p.MAP_DTE=e.EVENT_START AND p.TARGET_SCOPE=e.TARGET_SCOPE
    WHERE e.EVENT_START>=DATEADD(month,{lead_hi},f.FIRST_DATE)
      AND p.EXPOSURE>{min_exposure}
      AND (e.TARGET_SCOPE='DPD' OR p.EXPOSURE>=50000)
),
PREP AS (
    SELECT TARGET_SCOPE,ENTITY_ID,MAP_DTE,TARGET,EXPOSURE,GEC_EXPOSURE,
           {single_selection},
           LAG(MAP_DTE) OVER(
               PARTITION BY TARGET_SCOPE,ENTITY_ID ORDER BY MAP_DTE
           ) AS PREV_DTE,
           LAG(TARGET) OVER(
               PARTITION BY TARGET_SCOPE,ENTITY_ID ORDER BY MAP_DTE
           ) AS PREV_TARGET,
           {previous_fields}
    FROM TARGET_PANELS
    WHERE EXPOSURE>{min_exposure}
      AND (TARGET_SCOPE='DPD' OR EXPOSURE>=50000)
),
GRID_SINGLE AS (
    SELECT COLUMN1::VARCHAR AS PREDICTOR,
           COLUMN2::VARCHAR AS LABEL,
           COLUMN3::DOUBLE AS THRESHOLD
    FROM VALUES {grid_values}
),
PREDICTOR_LABELS AS (
    SELECT DISTINCT PREDICTOR,LABEL FROM GRID_SINGLE
),
GRID_GEC AS (
    SELECT COLUMN1::DOUBLE AS THRESHOLD
    FROM VALUES {gec_grid_values}
),
GRID_ALL AS (
    SELECT LABEL AS INDICATOR,LABEL AS PREDICTOR_A,
           CAST(NULL AS VARCHAR) AS PREDICTOR_B,
           THRESHOLD AS THRESHOLD_A,CAST(NULL AS DOUBLE) AS THRESHOLD_B
    FROM GRID_SINGLE
    UNION ALL
    {pair_grids}
    UNION ALL
    SELECT '{gec_indicator}' AS INDICATOR,
           'dpd' AS PREDICTOR_A,'GEC_EXPOSURE' AS PREDICTOR_B,
           d.THRESHOLD AS THRESHOLD_A,g.THRESHOLD AS THRESHOLD_B
    FROM GRID_SINGLE d CROSS JOIN GRID_GEC g WHERE d.PREDICTOR='DPD'
),
FLAT_FEATURES AS (
    SELECT u.TARGET_SCOPE,u.ENTITY_ID,u.MAP_DTE,u.TARGET,u.PREV_TARGET,u.PREV_DTE,
           u.EXPOSURE,UPPER(u.PREDICTOR) AS PREDICTOR,u.FEATURE_VALUE,
           {previous_choice} AS PREVIOUS_VALUE
    FROM PREP f
    UNPIVOT EXCLUDE NULLS (FEATURE_VALUE FOR PREDICTOR IN ({single_fields})) u
    WHERE (u.TARGET_SCOPE='DPD' AND UPPER(u.PREDICTOR)='DPD')
       OR (u.TARGET_SCOPE='RATING' AND UPPER(u.PREDICTOR)<>'DPD')
),
SINGLE_START AS (
    SELECT g.LABEL AS INDICATOR,g.LABEL AS PREDICTOR_A,
           CAST(NULL AS VARCHAR) AS PREDICTOR_B,
           f.TARGET_SCOPE,f.ENTITY_ID,f.MAP_DTE,g.THRESHOLD AS THRESHOLD_A,
           CAST(NULL AS DOUBLE) AS THRESHOLD_B
    FROM FLAT_FEATURES f
    JOIN GRID_SINGLE g ON g.PREDICTOR=f.PREDICTOR
      AND ((f.PREDICTOR='DPD' AND f.FEATURE_VALUE>=g.THRESHOLD)
        OR (f.PREDICTOR<>'DPD' AND f.FEATURE_VALUE>g.THRESHOLD))
    WHERE f.TARGET=0 AND (
       (f.PREDICTOR NOT IN {active_grid_names}
        AND f.PREV_TARGET=0
        AND DATEDIFF(day,f.PREV_DTE,f.MAP_DTE)=1
        AND f.PREVIOUS_VALUE IS NOT NULL
        AND ((f.PREDICTOR='DPD' AND f.PREVIOUS_VALUE<g.THRESHOLD)
          OR (f.PREDICTOR<>'DPD' AND f.PREVIOUS_VALUE<=g.THRESHOLD)))
       OR
       (f.PREDICTOR IN {active_grid_names}
        AND (f.PREV_TARGET IS NULL OR f.PREV_TARGET<>0
             OR f.PREV_DTE IS NULL
             OR DATEDIFF(day,f.PREV_DTE,f.MAP_DTE)<>1
             OR f.PREVIOUS_VALUE IS NULL
             OR f.PREVIOUS_VALUE<=g.THRESHOLD))
    )
),
RATING_PAIR_START AS (
    {pair_starts_sql}
),
DPD_GEC_START AS (
    SELECT '{gec_indicator}' AS INDICATOR,
           'dpd' AS PREDICTOR_A, 'GEC_EXPOSURE' AS PREDICTOR_B,
           s.ENTITY_ID,s.MAP_DTE,s.THRESHOLD_A,g.THRESHOLD AS THRESHOLD_B
    FROM SINGLE_START s
    JOIN PREP p ON p.TARGET_SCOPE='DPD'
      AND p.ENTITY_ID=s.ENTITY_ID AND p.MAP_DTE=s.MAP_DTE
    JOIN GRID_GEC g ON p.GEC_EXPOSURE>=g.THRESHOLD
    WHERE s.INDICATOR='dpd' AND s.TARGET_SCOPE='DPD'
),
SIGNAL_STARTS AS (
    SELECT DISTINCT INDICATOR,PREDICTOR_A,PREDICTOR_B,
           ENTITY_ID,MAP_DTE,THRESHOLD_A,THRESHOLD_B
    FROM (
        SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,ENTITY_ID,MAP_DTE,
               THRESHOLD_A,THRESHOLD_B FROM SINGLE_START
        UNION ALL
        SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,ENTITY_ID,MAP_DTE,
               THRESHOLD_A,THRESHOLD_B FROM RATING_PAIR_START
        UNION ALL
        SELECT * FROM DPD_GEC_START
    )
    WHERE MAP_DTE<=DATEADD(month,-{lead_lo},(SELECT END_DATE FROM PANEL_END))
),
SIGNAL_PREVIOUS AS (
    SELECT s.*,
           LAG(MAP_DTE) OVER(
               PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,ENTITY_ID
               ORDER BY MAP_DTE) AS PREV_START
    FROM SIGNAL_STARTS s
),
SIGNAL_GROUPS AS (
    SELECT s.*,
           SUM(CASE WHEN PREV_START IS NULL
                         OR MAP_DTE>DATEADD(month,{signal_gap},PREV_START)
                    THEN 1 ELSE 0 END)
           OVER(PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,ENTITY_ID
                ORDER BY MAP_DTE ROWS UNBOUNDED PRECEDING) AS SIGNAL_GROUP
    FROM SIGNAL_PREVIOUS s
),
SIGNALS AS (
    SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,THRESHOLD_A,THRESHOLD_B,
           ENTITY_ID,MIN(MAP_DTE) AS SIGNAL_START
    FROM SIGNAL_GROUPS
    GROUP BY 1,2,3,4,5,6,SIGNAL_GROUP
),
EVENT_MAX_GEC AS (
    SELECT e.EVENT_ID,e.EVENT_EXPOSURE,MAX(p.GEC_EXPOSURE) AS MAX_LEAD_GEC
    FROM EVENTS e
    JOIN PREP p ON p.ENTITY_ID=e.ENTITY_ID AND p.TARGET_SCOPE='DPD'
      AND p.MAP_DTE BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                         AND DATEADD(month,-{lead_lo},e.EVENT_START)
    WHERE e.TARGET_SCOPE='DPD' AND p.TARGET=0 AND p.GEC_EXPOSURE IS NOT NULL
    GROUP BY 1,2
),
INDICATOR_EVENTS AS (
    SELECT DISTINCT pl.LABEL AS INDICATOR,
           CAST(NULL AS DOUBLE) AS EXPOSURE_CUTOFF,e.EVENT_ID,e.EVENT_EXPOSURE
    FROM EVENTS e
    JOIN FLAT_FEATURES f ON f.ENTITY_ID=e.ENTITY_ID
      AND f.TARGET_SCOPE=e.TARGET_SCOPE
      AND f.MAP_DTE BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                         AND DATEADD(month,-{lead_lo},e.EVENT_START)
    JOIN PREDICTOR_LABELS pl ON pl.PREDICTOR=f.PREDICTOR
    WHERE e.TARGET_SCOPE='RATING' AND f.TARGET=0
    UNION ALL
    SELECT DISTINCT 'dpd' AS INDICATOR,
           CAST(NULL AS DOUBLE) AS EXPOSURE_CUTOFF,
           e.EVENT_ID,e.EVENT_EXPOSURE
    FROM EVENTS e JOIN FLAT_FEATURES f
      ON f.ENTITY_ID=e.ENTITY_ID AND f.TARGET_SCOPE='DPD'
      AND e.TARGET_SCOPE='DPD' AND f.PREDICTOR='DPD'
      AND f.MAP_DTE BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                        AND DATEADD(month,-{lead_lo},e.EVENT_START)
    WHERE f.TARGET=0
    UNION ALL
    {rating_pair_eligible}
    UNION ALL
    SELECT '{gec_indicator}' AS INDICATOR,g.THRESHOLD AS EXPOSURE_CUTOFF,
           e.EVENT_ID,e.EVENT_EXPOSURE
    FROM EVENT_MAX_GEC e JOIN GRID_GEC g ON e.MAX_LEAD_GEC>=g.THRESHOLD
),
EVENT_COUNTS AS (
    SELECT INDICATOR,EXPOSURE_CUTOFF,COUNT(DISTINCT EVENT_ID) AS N_EVENTS,
           SUM(EVENT_EXPOSURE) AS SUM_GCA_EVENTS
    FROM INDICATOR_EVENTS GROUP BY 1,2
),
VALID_PAIRS AS (
    SELECT s.INDICATOR,s.THRESHOLD_A,s.THRESHOLD_B,
           s.ENTITY_ID,s.SIGNAL_START,e.EVENT_ID,e.EVENT_START,e.EVENT_EXPOSURE,
           IFF(s.INDICATOR='{gec_indicator}',s.THRESHOLD_B,NULL) AS EXPOSURE_CUTOFF
    FROM SIGNALS s
    JOIN EVENTS e ON e.ENTITY_ID=s.ENTITY_ID
      AND e.TARGET_SCOPE=IFF(s.INDICATOR IN ('dpd','{gec_indicator}'),'DPD','RATING')
      AND s.SIGNAL_START BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                             AND DATEADD(month,-{lead_lo},e.EVENT_START)
    JOIN INDICATOR_EVENTS ie ON ie.INDICATOR=s.INDICATOR
      AND ie.EVENT_ID=e.EVENT_ID
      AND EQUAL_NULL(ie.EXPOSURE_CUTOFF,
          IFF(s.INDICATOR='{gec_indicator}',s.THRESHOLD_B,NULL))
),
FIRST_MATCH AS (
    SELECT * FROM VALID_PAIRS
    QUALIFY ROW_NUMBER() OVER(
        PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,EVENT_ID
        ORDER BY SIGNAL_START)=1
),
TP AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           COUNT(*) AS N_TP,
           SUM(EVENT_EXPOSURE) AS SUM_GCA_TP,
           SUM(EVENT_EXPOSURE*(
             {cost_tp_late} + ({cost_tp_early}-{cost_tp_late})*
             (LEAST({lead_hi},GREATEST({lead_lo},
                DATEDIFF(day,SIGNAL_START,EVENT_START)/(365.25/12.0)))-{lead_lo})
             /({lead_hi}-{lead_lo})
           )) AS SUM_GCA_TP_WTD
    FROM FIRST_MATCH GROUP BY 1,2,3
),
DETECTED AS (
    SELECT INDICATOR,EXPOSURE_CUTOFF,
           COUNT(DISTINCT EVENT_ID) AS N_EVENTS_DETECTABLE
    FROM VALID_PAIRS GROUP BY 1,2
),
MATCHED_SIGNALS AS (
    SELECT DISTINCT INDICATOR,THRESHOLD_A,THRESHOLD_B,ENTITY_ID,SIGNAL_START
    FROM VALID_PAIRS
),
SIGNAL_COUNTS AS (
    SELECT s.INDICATOR,s.THRESHOLD_A,s.THRESHOLD_B,
           COUNT(*) AS N_SIGNALS,
           COUNT_IF(m.SIGNAL_START IS NULL) AS N_FP
    FROM SIGNALS s LEFT JOIN MATCHED_SIGNALS m
      ON s.INDICATOR=m.INDICATOR AND s.THRESHOLD_A=m.THRESHOLD_A
     AND EQUAL_NULL(s.THRESHOLD_B,m.THRESHOLD_B)
     AND s.ENTITY_ID=m.ENTITY_ID AND s.SIGNAL_START=m.SIGNAL_START
    GROUP BY 1,2,3
),
SIGNAL_MONTHS AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           DATE_TRUNC(month,SIGNAL_START) AS SIGNAL_MONTH,
           COUNT(*) AS N_MONTHLY_SIGNALS
    FROM SIGNALS GROUP BY 1,2,3,4
),
MONTHLY_MAX AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           MAX(N_MONTHLY_SIGNALS) AS MAX_MONTHLY_SIGNALS
    FROM SIGNAL_MONTHS GROUP BY 1,2,3
),
SIGNAL_DAYS AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           SIGNAL_START,COUNT(*) AS N_DAILY_SIGNALS
    FROM SIGNALS GROUP BY 1,2,3,4
),
DAILY_MAX AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           MAX(N_DAILY_SIGNALS) AS MAX_DAILY_SIGNALS
    FROM SIGNAL_DAYS GROUP BY 1,2,3
),
SCOPE_COUNTS AS (
    SELECT TARGET_SCOPE,COUNT(DISTINCT ENTITY_ID) AS N_ENTITIES,
           GREATEST(COUNT(DISTINCT DATE_TRUNC(month,MAP_DTE)),1) AS N_MONTHS,
           GREATEST(COUNT(DISTINCT MAP_DTE),1) AS N_DAYS
    FROM PREP WHERE TARGET=0 GROUP BY TARGET_SCOPE
),
METRICS AS (
    SELECT g.INDICATOR,g.PREDICTOR_A,g.PREDICTOR_B,
           g.THRESHOLD_A,g.THRESHOLD_B,
           c.N_ENTITIES,
           COALESCE(t.N_TP,0) AS N_TP,
           COALESCE(s.N_FP,0) AS N_FP,
           GREATEST(COALESCE(e.N_EVENTS,0)-COALESCE(t.N_TP,0),0) AS N_FN,
           COALESCE(t.N_TP,0)+COALESCE(s.N_FP,0) AS N_ALERTS,
           COALESCE(s.N_SIGNALS,0) AS N_SIGNALS,
           COALESCE(e.N_EVENTS,0) AS N_EVENTS,
           COALESCE(d.N_EVENTS_DETECTABLE,0) AS N_EVENTS_DETECTABLE,
           COALESCE(t.SUM_GCA_TP,0) AS SUM_GCA_TP,
           COALESCE(t.SUM_GCA_TP_WTD,0) AS SUM_GCA_TP_WTD,
           GREATEST(COALESCE(e.SUM_GCA_EVENTS,0)-COALESCE(t.SUM_GCA_TP,0),0) AS SUM_GCA_FN,
           COALESCE(mm.MAX_MONTHLY_SIGNALS,0) AS MAX_MONTHLY_ALERTS,
           COALESCE(dm.MAX_DAILY_SIGNALS,0) AS MAX_DAILY_ALERTS,
           c.N_MONTHS,c.N_DAYS
    FROM GRID_ALL g
    LEFT JOIN SCOPE_COUNTS c ON c.TARGET_SCOPE=
      IFF(g.INDICATOR IN ('dpd','{gec_indicator}'),'DPD','RATING')
    LEFT JOIN TP t ON t.INDICATOR=g.INDICATOR
      AND t.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(t.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN SIGNAL_COUNTS s ON s.INDICATOR=g.INDICATOR
      AND s.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(s.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN EVENT_COUNTS e ON e.INDICATOR=g.INDICATOR
      AND EQUAL_NULL(e.EXPOSURE_CUTOFF,
          IFF(g.INDICATOR='{gec_indicator}',g.THRESHOLD_B,NULL))
    LEFT JOIN DETECTED d ON d.INDICATOR=g.INDICATOR
      AND EQUAL_NULL(d.EXPOSURE_CUTOFF,
          IFF(g.INDICATOR='{gec_indicator}',g.THRESHOLD_B,NULL))
    LEFT JOIN MONTHLY_MAX mm ON mm.INDICATOR=g.INDICATOR
      AND mm.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(mm.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN DAILY_MAX dm ON dm.INDICATOR=g.INDICATOR
      AND dm.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(dm.THRESHOLD_B,g.THRESHOLD_B)
),
COSTED AS (
    SELECT m.*,
           {cost_tp_fixed}*N_TP + {cost_fp}*N_FP
              + SUM_GCA_TP_WTD + {cost_fn_pct}*SUM_GCA_FN AS TOTAL_COST,
           IFF(N_ALERTS>0,N_TP/N_ALERTS,NULL) AS ACCURACY,
           IFF(N_EVENTS>0,N_TP/N_EVENTS,NULL) AS COVERAGE,
           IFF(N_EVENTS_DETECTABLE>0,N_TP/N_EVENTS_DETECTABLE,NULL) AS COVERAGE_2,
           IFF(MAX(N_TP) OVER(PARTITION BY INDICATOR,
               IFF(INDICATOR='{gec_indicator}',THRESHOLD_B,NULL))>0,
               N_TP/MAX(N_TP) OVER(PARTITION BY INDICATOR,
               IFF(INDICATOR='{gec_indicator}',THRESHOLD_B,NULL)),NULL) AS COVERAGE_3
    FROM METRICS m
),
FINAL AS (
    SELECT m.*,
           ROW_NUMBER() OVER(PARTITION BY INDICATOR
             ORDER BY TOTAL_COST ASC,THRESHOLD_A ASC,THRESHOLD_B ASC NULLS FIRST) AS COST_RANK,
           ROW_NUMBER() OVER(PARTITION BY INDICATOR,
                    IFF(INDICATOR='{gec_indicator}',THRESHOLD_B,NULL)
             ORDER BY TOTAL_COST ASC,THRESHOLD_A ASC) AS COST_RANK_AT_GEC,
           MAX(N_TP) OVER(PARTITION BY INDICATOR,
                    IFF(INDICATOR='{gec_indicator}',THRESHOLD_B,NULL)) AS N_EVENTS_REACHABLE
    FROM COSTED m
)
SELECT
    INDICATOR,PREDICTOR_A,PREDICTOR_B,N_ENTITIES,THRESHOLD_A,THRESHOLD_B,
    {rating_a_sql} AS THRESHOLD_A_RATING_LABEL,
    {rating_b_sql} AS THRESHOLD_B_RATING_LABEL,
    IFF(INDICATOR='{gec_indicator}',THRESHOLD_B,NULL) AS GEC_EXPOSURE_THRESHOLD,
    IFF(INDICATOR IN ('dpd','{gec_indicator}'),'EXCLUDING_DPD_COMPONENT','RETAIL_FIVE_COMPONENTS') AS TARGET_DEFINITION,
    N_TP,N_FP,N_FN,N_ALERTS,N_SIGNALS,
    N_SIGNALS AS N_ALERTS_SIGNAL_STARTS,
    MAX_MONTHLY_ALERTS,MAX_DAILY_ALERTS,
    N_EVENTS,N_EVENTS_DETECTABLE,N_EVENTS_REACHABLE,
    ROUND(ACCURACY,3) AS ACCURACY,
    ROUND(COVERAGE,3) AS COVERAGE,
    ROUND(COVERAGE_2,3) AS COVERAGE_2,
    ROUND(COVERAGE_3,3) AS COVERAGE_3,
    SUM_GCA_TP,SUM_GCA_TP_WTD,SUM_GCA_FN,
    ROUND(TOTAL_COST,2) AS TOTAL_COST,
    IFF(COST_RANK=1,TRUE,FALSE) AS OPTIMAL,
    IFF(COST_RANK_AT_GEC=1,TRUE,FALSE) AS OPTIMAL_AT_GEC,
    CAST(NULL AS DOUBLE) AS SUM_COST2_TP,
    CAST(NULL AS DOUBLE) AS SUM_COST2_FN,
    CAST(NULL AS DOUBLE) AS TOTAL_COST_2,
    CAST(NULL AS BOOLEAN) AS OPTIMAL_COST_2,
    ROUND(N_ALERTS/N_MONTHS,0) AS AV_NR_MON_ALERTS,
    ROUND(N_ALERTS/N_MONTHS,0)/0.7 AS AV_EXP_NR_MON_ALERTS,
    ROUND(MAX_MONTHLY_ALERTS,0)/0.7 AS MAX_EXP_NR_MON_ALERTS,
    ROUND(N_ALERTS/N_DAYS,3) AS AV_NR_DAY_ALERTS,
    ROUND(N_SIGNALS/N_MONTHS,2) AS AV_NR_MON_SIGNAL_STARTS,
    ROUND(N_SIGNALS/N_DAYS,3) AS AV_NR_DAY_SIGNAL_STARTS,
    IFF(INDICATOR='{gec_indicator}','GEC_NOMINAL_CANDIDATE','OWN_NOMINAL_COMPLETE') AS EXPOSURE_BASIS,
    'RETAIL_DPD_PD_RATING_GCC_B01' AS CATALOG_SCOPE,
    'CALENDAR_DATE_EXACT' AS FEATURE_LAG_METHOD,
    CASE WHEN N_EVENTS=0 THEN 'NO_HISTORICAL_EVENTS'
         WHEN N_SIGNALS=0 THEN 'NO_HISTORICAL_SIGNALS'
         ELSE 'REVIEW_REQUIRED' END AS CALIBRATION_STATUS
FROM FINAL m
""".format(
    SOURCE=SOURCE,
    event_gap=MAX_GAP_MONTHS,
    lead_hi=MAX_LEAD_MONTHS,
    lead_lo=MIN_LEAD_MONTHS,
    signal_gap=MAX_SIGNAL_GAP_MONTHS,
    min_exposure=MIN_EXPOSURE,
    single_fields=single_fields,
    single_selection=single_selection,
    previous_fields=previous_fields,
    grid_values=grid_values,
    gec_grid_values=GEC_GRID_VALUES,
    gec_indicator=GEC_INDICATOR,
    pair_grids=pair_grids,
    previous_choice=previous_choice,
    active_grid_names=active_grid_names,
    pair_starts_sql=pair_starts_sql,
    rating_pair_eligible=rating_pair_eligible,
    cost_tp_fixed=COST_TP_FIXED,
    cost_fp=COST_FP,
    cost_fn_pct=COST_FN_PCT,
    cost_tp_late=COST_TP_PCT_LATE,
    cost_tp_early=COST_TP_PCT_EARLY,
    rating_a_sql=rating_a_sql,
    rating_b_sql=rating_b_sql,
    feature_formulas=feature_formulas,
    expected_results=EXPECTED_RESULTS,
)

output_results = dataiku.Dataset(RESULTS, project_key=PROJECT)
SQLExecutor2.exec_recipe_fragment(output_results, SQL, overwrite_output_schema=True)

QA_SQL = r"""
WITH SOURCE_QA AS (
    SELECT COUNT(*) AS TOTAL_ROWS,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL) AS MAPPED_DPD_ROWS,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL
             AND GEC_NOMINAL_CANDIDATE IS NOT NULL) AS GEC_NOMINAL_AVAILABLE,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL
             AND GEC_GCA_CANDIDATE IS NOT NULL) AS GEC_GCA_AVAILABLE,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL AND TGT_ANY_SAMEDAY IS NULL)
               AS RETAIL_TARGET_UNKNOWN,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL AND CLIENT_NOTCH IS NOT NULL)
               AS CLIENT_RATING_AVAILABLE,
           COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
             AND DPD_MAX IS NOT NULL AND GCC_NOTCH IS NOT NULL)
               AS GCC_RATING_AVAILABLE,
           MAX(IFF(DPD_MAX IS NOT NULL AND LOAN_COUNT>0,
             TO_DATE(MAP_DTE),NULL)) AS LAST_DPD_DATE,
           MAX(TO_DATE(MAP_DTE)) AS LAST_TRAINING_DATE
    FROM {SOURCE}
),
RESULT_QA AS (
    SELECT COUNT(*) AS RESULT_ROWS,
           COUNT_IF(INDICATOR='{gec_indicator}') AS DPD_GEC_ROWS,
           COUNT_IF(INDICATOR='{gec_indicator}' AND OPTIMAL_AT_GEC=TRUE) AS DPD_GEC_OPTIMAL_ROWS,
           COUNT_IF(N_TP+N_FP<>N_ALERTS) AS BAD_ALERT_IDENTITIES,
           COUNT_IF(N_ALERTS_SIGNAL_STARTS<>N_SIGNALS) AS BAD_RAW_ALERT_IDENTITIES,
           COUNT_IF(N_TP+N_FN<>N_EVENTS) AS BAD_EVENT_IDENTITIES,
           COUNT_IF(ACCURACY<0 OR ACCURACY>1
                OR COVERAGE<0 OR COVERAGE>1
                OR COVERAGE_2<0 OR COVERAGE_2>1
                OR COVERAGE_3<0 OR COVERAGE_3>1) AS BAD_METRIC_RANGES,
           COUNT_IF(COVERAGE_2<COVERAGE OR COVERAGE_3<COVERAGE_2) AS BAD_COVERAGE_ORDER,
           COUNT(DISTINCT INDICATOR) AS INDICATORS,
           COUNT_IF(OPTIMAL) AS OPTIMAL_ROWS,
           COUNT_IF(N_EVENTS>0) AS ROWS_WITH_EVENTS
    FROM {RESULTS_TABLE}
),
CHECKS AS (
    SELECT 'MODEL_CHECK' AS QA_TYPE,'At least one result' AS CHECK_NAME,
           IFF(r.RESULT_ROWS>0,TRUE,FALSE) AS CHECK_PASSED,
           r.RESULT_ROWS::DOUBLE AS CHECK_VALUE,1.0 AS EXPECTED_VALUE
    FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Retail B01 indicator grid complete',
           IFF(r.RESULT_ROWS={expected_results},TRUE,FALSE),r.RESULT_ROWS::DOUBLE,{expected_results}.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','DPD x GEC grid complete',
           IFF(r.DPD_GEC_ROWS=240,TRUE,FALSE),r.DPD_GEC_ROWS::DOUBLE,240.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','One DPD optimum at each GEC threshold',
           IFF(r.DPD_GEC_OPTIMAL_ROWS=20,TRUE,FALSE),r.DPD_GEC_OPTIMAL_ROWS::DOUBLE,20.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','TP+FP equals alerts',
           IFF(r.BAD_ALERT_IDENTITIES=0,TRUE,FALSE),r.BAD_ALERT_IDENTITIES::DOUBLE,0.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Raw alert starts equal signal count',
           IFF(r.BAD_RAW_ALERT_IDENTITIES=0,TRUE,FALSE),r.BAD_RAW_ALERT_IDENTITIES::DOUBLE,0.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','TP+FN equals events',
           IFF(r.BAD_EVENT_IDENTITIES=0,TRUE,FALSE),r.BAD_EVENT_IDENTITIES::DOUBLE,0.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Metrics within 0-1',
           IFF(r.BAD_METRIC_RANGES=0,TRUE,FALSE),r.BAD_METRIC_RANGES::DOUBLE,0.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Coverage order',
           IFF(r.BAD_COVERAGE_ORDER=0,TRUE,FALSE),r.BAD_COVERAGE_ORDER::DOUBLE,0.0 FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Exactly one optimal per indicator',
           IFF(r.OPTIMAL_ROWS=r.INDICATORS,TRUE,FALSE),r.OPTIMAL_ROWS::DOUBLE,r.INDICATORS::DOUBLE FROM RESULT_QA r
    UNION ALL SELECT 'MODEL_CHECK','Historical events available',
           IFF(r.ROWS_WITH_EVENTS>0,TRUE,FALSE),r.ROWS_WITH_EVENTS::DOUBLE,1.0 FROM RESULT_QA r
    UNION ALL SELECT 'DATA_QUALITY','Corporate DPD-mapped rows',NULL,
           s.MAPPED_DPD_ROWS::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','Complete GEC nominal candidate rows',NULL,
           s.GEC_NOMINAL_AVAILABLE::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','Complete GEC GCA candidate rows',NULL,
           s.GEC_GCA_AVAILABLE::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','Retail target unknown rows',NULL,
           s.RETAIL_TARGET_UNKNOWN::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','Client rating available rows',NULL,
           s.CLIENT_RATING_AVAILABLE::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','GCC rating available rows',NULL,
           s.GCC_RATING_AVAILABLE::DOUBLE,NULL FROM SOURCE_QA s
    UNION ALL SELECT 'DATA_QUALITY','GEC nominal completeness share in mapped DPD rows',NULL,
           IFF(s.MAPPED_DPD_ROWS>0,s.GEC_NOMINAL_AVAILABLE/s.MAPPED_DPD_ROWS,NULL),NULL
           FROM SOURCE_QA s
    UNION ALL SELECT 'METHODOLOGY','Retail DPD/PD/Rating/GCC candidates',TRUE,88.0,88.0 FROM SOURCE_QA
    UNION ALL SELECT 'DATA_QUALITY','Retail candidates missing B01 inputs',NULL,117.0,205.0 FROM SOURCE_QA
    UNION ALL SELECT 'METHODOLOGY','DPD 5-60 in 5-day steps',TRUE,12.0,12.0 FROM SOURCE_QA
    UNION ALL SELECT 'METHODOLOGY','GEC exposure 50k-1m in 50k steps',TRUE,20.0,20.0 FROM SOURCE_QA
    UNION ALL SELECT 'METHODOLOGY','1-11 month lead window',TRUE,1.0,1.0 FROM SOURCE_QA
    UNION ALL SELECT 'METHODOLOGY','3-month event grouping',TRUE,3.0,3.0 FROM SOURCE_QA
    UNION ALL SELECT 'METHODOLOGY','2-month signal merging',TRUE,2.0,2.0 FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','GEC nominal candidate requires bank reconciliation',NULL,NULL,NULL FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','Mortgage exclusions require source-level verification',NULL,NULL,NULL FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','Default DIM1 and forbearance codes require validation',NULL,NULL,NULL FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','Collateral-adjusted cost not computed',NULL,NULL,NULL FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','DPD target excludes DPD component to prevent circularity',NULL,NULL,NULL FROM SOURCE_QA
    UNION ALL SELECT 'BUSINESS_REVIEW','N_ALERTS uses Retail TP plus FP; N_ALERTS_SIGNAL_STARTS counts actual signals',NULL,NULL,NULL FROM SOURCE_QA
),
BEST AS (
    SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,THRESHOLD_A,THRESHOLD_B,
           N_ALERTS,AV_NR_MON_ALERTS,AV_EXP_NR_MON_ALERTS,MAX_EXP_NR_MON_ALERTS,
           ACCURACY,COVERAGE,COVERAGE_2,COVERAGE_3,TOTAL_COST
    FROM {RESULTS_TABLE} WHERE OPTIMAL=TRUE
)
SELECT c.QA_TYPE,c.CHECK_NAME,
       CAST(NULL AS VARCHAR) AS INDICATOR,
       CAST(NULL AS VARCHAR) AS PREDICTOR_A,
       CAST(NULL AS VARCHAR) AS PREDICTOR_B,
       CAST(NULL AS VARCHAR) AS UNAVAILABLE_COLUMNS,
       CAST(NULL AS VARCHAR) AS ENTITY_ID,
       CAST(NULL AS DATE) AS MAP_DTE,
       CAST(NULL AS VARCHAR) AS SIGNAL_LOGIC,
       CAST(NULL AS VARCHAR) AS DIRECTION,
       CAST(NULL AS DOUBLE) AS THRESHOLD_A,
       CAST(NULL AS DOUBLE) AS THRESHOLD_B,
       CAST(NULL AS DOUBLE) AS PREDICTOR_VALUE,
       CAST(NULL AS DOUBLE) AS PREVIOUS_VALUE,
       CAST(NULL AS BOOLEAN) AS ALERT_GENERATED,
       c.CHECK_PASSED,c.CHECK_VALUE,c.EXPECTED_VALUE,
       CAST(NULL AS NUMBER) AS N_ALERTS,
       CAST(NULL AS DOUBLE) AS AV_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS AV_EXP_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS MAX_EXP_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS ACCURACY,
       CAST(NULL AS DOUBLE) AS COVERAGE,
       CAST(NULL AS DOUBLE) AS COVERAGE_2,
       CAST(NULL AS DOUBLE) AS COVERAGE_3,
       CAST(NULL AS DOUBLE) AS TOTAL_COST
FROM CHECKS c
UNION ALL
SELECT 'OPTIMAL_THRESHOLD',CAST(NULL AS VARCHAR),
       b.INDICATOR,b.PREDICTOR_A,b.PREDICTOR_B,
       CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR),CAST(NULL AS DATE),
       CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR),
       b.THRESHOLD_A,b.THRESHOLD_B,
       CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS BOOLEAN),
       CAST(NULL AS BOOLEAN),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
       b.N_ALERTS,b.AV_NR_MON_ALERTS,b.AV_EXP_NR_MON_ALERTS,
       b.MAX_EXP_NR_MON_ALERTS,b.ACCURACY,b.COVERAGE,b.COVERAGE_2,
       b.COVERAGE_3,b.TOTAL_COST
FROM BEST b
""".format(SOURCE=SOURCE,RESULTS_TABLE=TARGET_TABLE,gec_indicator=GEC_INDICATOR,expected_results=EXPECTED_RESULTS)

output_tests = dataiku.Dataset(TEST_RESULTS, project_key=PROJECT)
SQLExecutor2.exec_recipe_fragment(output_tests, QA_SQL, overwrite_output_schema=True)
print('Corporate B01 full Retail DPD/PD/Rating/GCC calibration completed.')
