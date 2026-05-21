#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_groups_qe.py
"""
import os
import sys
import argparse
import math
from pathlib import Path
import base64
import io
from datetime import datetime
import pingouin as pg
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from statsmodels.stats.multitest import multipletests

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_WIDE = "./metrics_wide_by_timestamp_fillna.csv"
OUT_ROOT = "./qe_outputs/qe_compare_outputs"
ALPHA = 0.05

DEFAULT_METRICS = [
    "sbert_jap_en_ai", "sbert_jap_en_fansub", "sbert_jap_en_official",
    "sbert_jap_sp_ai", "sbert_jap_sp_fansub", "sbert_jap_sp_official",
    "comet_jap_en_ai", "comet_jap_en_fansub", "comet_jap_en_official",
    "comet_jap_sp_ai", "comet_jap_sp_fansub", "comet_jap_sp_official",
]
sbert_pairs = [
    ('sbert_jap_en_ai', 'sbert_jap_en_fansub'), ('sbert_jap_en_fansub', 'sbert_jap_en_official'), ('sbert_jap_en_ai', 'sbert_jap_en_official'),
    ('sbert_jap_sp_ai', 'sbert_jap_sp_fansub'), ('sbert_jap_sp_fansub', 'sbert_jap_sp_official'), ('sbert_jap_sp_ai', 'sbert_jap_sp_official')
]

comet_pairs = [
    ('comet_jap_en_ai', 'comet_jap_en_fansub'), ('comet_jap_en_fansub', 'comet_jap_en_official'), ('comet_jap_en_ai', 'comet_jap_en_official'),
    ('comet_jap_sp_ai', 'comet_jap_sp_fansub'), ('comet_jap_sp_fansub', 'comet_jap_sp_official'), ('comet_jap_sp_ai', 'comet_jap_sp_official')
]

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def run_paired_analysis(df, metric_name, pairs):
    print(f"\n=== Running Analysis for {metric_name} ===")
    results = []

    for col1, col2 in pairs:
        # Step A: Isolate pairs and drop missing data just for these two columns
        pair_df = df[[col1, col2]].dropna()
        n_pairs = len(pair_df)

        # Fallback handling for empty pairs
        if n_pairs == 0:
            results.append({"Pair": f"{col1} vs {col2}", "Valid Pairs (N)": 0, "Test Used": "Failed (No Data)"})
            continue

        # Step B: Calculate differences for the normality check
        x = pair_df[col1].values
        y = pair_df[col2].values
        diff = x - y

        # Step B: Shapiro-Wilk test on differences
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        is_normal = shapiro_p > 0.05

        # Step C: Parametric calculations (Your Cohen's d formula)
        t_stat, t_p = stats.ttest_rel(x, y)
        md = np.nanmean(diff)
        sd = np.nanstd(diff, ddof=1)
        cohens_d = md / sd if sd and not np.isnan(sd) else np.nan
        sem = stats.sem(diff)
        ci = stats.t.interval(
            0.95,
            len(diff)-1,
            loc=np.mean(diff),
            scale=sem
        )

        # Step D: Non-parametric calculations (Your Wilcoxon formula with error handling)
        w_stat, wilcox_p, wilcox_r, wilcox_error = np.nan, np.nan, np.nan, None
        try:
            # Using your requested parameters: zero_method='wilcox', correction=False
            w_stat, wilcox_p = stats.wilcoxon(x, y, zero_method='wilcox', correction=False)
            # Approximate z from w using normal approximation formula
            mean_w = n_pairs * (n_pairs + 1) / 4.0
            sd_w = math.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24.0)
            z = (w_stat - mean_w) / sd_w if sd_w > 0 else 0.0
            wilcox_r = abs(z) / math.sqrt(n_pairs) if n_pairs > 0 else np.nan
        except Exception as e:
            wilcox_error = str(e)
        # Wilcox with pingouin
        # Index(['W_val', 'alternative', 'p_val', 'RBC', 'CLES'], dtype='object')
        try:
            wilcox_pg = pg.wilcoxon(x, y)
            wilcox_w_pg = wilcox_pg["W_val"].iloc[0]
            wilcox_alt_pg = wilcox_pg["alternative"].iloc[0]
            wilcox_p_pg = wilcox_pg["p_val"].iloc[0]
            wilcox_rbc_pg = wilcox_pg["RBC"].iloc[0]
            wilcox_cles_pg = wilcox_pg["CLES"].iloc[0]
            wilcox_error_pg = None
        except Exception as e:
            wilcox_w_pg = np.nan
            wilcox_alt_pg = np.nan
            wilcox_p_pg = np.nan
            wilcox_rbc_pg = np.nan
            wilcox_cles_pg = np.nan
            wilcox_error_pg = str(e)
        
        # Step E: Make an overarching choice for the "Primary Test" based on normality
        if is_normal:
            test_name = "Paired T-Test"
            final_p = t_p
        else:
            test_name = "Wilcoxon Signed-Rank"
            final_p = wilcox_p

        results.append({
            "Pair": f"{col1} vs {col2}",
            "Valid Pairs (N)": n_pairs,
            "Normality Test": "Normal" if is_normal else "Non-Normal",
            "Shapiro p-val": shapiro_p,
            "Primary Test": test_name,

            # Parametric Outputs
            "T-Stat": t_stat,
            "T-Test P-Value": t_p,
            "Mean Diff": md,
            "SD Diff": sd,
            "Cohen's d": cohens_d,

            # Non-Parametric Outputs
            "W-Stat": w_stat,
            "W-Stat pg": wilcox_w_pg,
            "Wilcox P-Value": wilcox_p,
            "Wilcox P-Value pg": wilcox_p_pg,
            "Wilcox alternative pg": wilcox_alt_pg,
            "Wilcox r (Effect Size)": wilcox_r,
            "Wilcox RBC pg": wilcox_rbc_pg,
            "Wilcox CLES pg": wilcox_cles_pg,
            "Wilcox Error": wilcox_error,
            "Wilcox Error pg": wilcox_error_pg,

            # Choice for global correction
            "RAW P-Value": final_p
        })

    # Step F: Group results and apply Holm-Bonferroni correction on the chosen tests
    res_df = pd.DataFrame(results)
    if not res_df.empty and "RAW P-Value" in res_df.columns:
        # Filter out completely failed rows if any exist before correcting p-values
        valid_mask = res_df['RAW P-Value'].notna()
        if valid_mask.any():
            res_df.loc[valid_mask, 'Adjusted P-Value'] = multipletests(res_df.loc[valid_mask, 'RAW P-Value'], method='holm')[1]
            res_df['Significant?'] = res_df['Adjusted P-Value'] < ALPHA

    return res_df

def pipeline():
    # Read wide dataset
    wide = pd.read_csv(DEFAULT_WIDE, engine="python", encoding="utf-8")
    #print(wide.columns)
    for g in wide.columns:
      if g in DEFAULT_METRICS: print(f'{g} in DEFAULT_METRICS')
    #-----------------------------------------#
    # Delete invalid rows // THIS IS OPTIONAL #
    #-----------------------------------------#
    sbert_results = run_paired_analysis(wide, "SBERT", sbert_pairs)
    comet_results = run_paired_analysis(wide, "COMET", comet_pairs)
    pair_df = pd.concat([pd.DataFrame(sbert_results), pd.DataFrame(comet_results)], ignore_index=True)
    stats_folder = ensure_dir(os.path.join(OUT_ROOT, "stats"))
    pair_csv = os.path.join(stats_folder, f"pairwise_tests.csv")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    print("[OK] Saved pairwise test results:", pair_csv)

pipeline()
