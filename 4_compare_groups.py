#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_groups_qe.py
Local pipeline to compare translation quality between groups (official, fansub, ai)
Default input:
 - .\data\metrics_wide_strict_allmetrics.csv  (wide file already created)
If not present, the script will attempt to load .\data\metrics_all_groups.csv and pivot to wide.
Default output:
 - .\qe_compare_outputs\ (CSVs, plots, HTML report)

Dependencies:
pip install pandas numpy scipy statsmodels matplotlib seaborn jinja2

Usage:
python compare_groups_qe.py --input .\data\metrics_wide_strict_allmetrics.csv
"""
import os
import sys
import argparse
import math
from pathlib import Path
import base64
import io
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from statsmodels.stats.multitest import multipletests

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_WIDE = "./metrics_wide_by_timestamp_fillna.csv"
FALLBACK_RAW = ""
OUT_ROOT = "./qe_outputs/qe_compare_outputs"
ALPHA = 0.05

# default metrics to analyze (must match column base names in the wide file)
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

def es_label_r(r):
    try:
        v = abs(float(r))
    except:
        return "NA"
    if v < 0.1: return "negligible (<0.1)"
    if v < 0.3: return "small (0.1-0.3)"
    if v < 0.5: return "medium (0.3-0.5)"
    return "large (>=0.5)"

def run_paired_analysis(df, metric_name, pairs):
    print(f"\n=== Running Analysis for {metric_name} ===")
    results = []

    for col1, col2 in pairs:
        #Isolate pairs and drop missing data just for these two columns
        pair_df = df[[col1, col2]].dropna()
        n_pairs = len(pair_df)

        #Calculate differences for the normality check
        x = pair_df[col1].values
        y = pair_df[col2].values
        diff = x - y

        #Shapiro-Wilk test on differences
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        is_normal = shapiro_p > 0.05

        #Parametric calculations
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

        #Non-parametric calculations
        w_stat, wilcoxon_p, wilcoxon_r = [np.nan] * 3
        RBC, CLES = np.nan, np.nan
        boot_ci_low, boot_ci_high = np.nan, np.nan
        wilcoxon_error = None
        effect_size_label_RBC = "NA"
        effect_size_label_wilcoxon_r = "NA"
        try:
            w_stat, wilcoxon_p = stats.wilcoxon(x, y, zero_method='wilcox', correction=False)
            # remove zeros
            d_diff = diff[diff != 0]
            n_nonzero = len(d_diff)

            # mean, sd, z, r
            mean_w = n_nonzero * (n_nonzero + 1) / 4.0
            sd_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
            z = (w_stat - mean_w) / sd_w if sd_w > 0 else 0.0
            wilcoxon_r = z / math.sqrt(n_nonzero) if n_nonzero > 0 else np.nan

            # rank-biserial correlation
            abs_d = np.abs(d_diff)
            ranks = stats.rankdata(abs_d)
            W_plus = ranks[d_diff > 0].sum()
            W_minus = ranks[d_diff < 0].sum()
            RBC = (W_plus - W_minus) / (W_plus + W_minus)

            # Common Language Effect Size
            CLES = np.mean(diff > 0) + 0.5 * np.mean(diff == 0)

            # Effect size label
            effect_size_label_RBC = es_label_r(RBC)
            effect_size_label_wilcoxon_r = es_label_r(wilcoxon_r)

            # Bootstrap CI
            boot_ci_low, boot_ci_high = np.nan, np.nan
            try:
                rng = np.random.default_rng(42)
                diffs = diff[~np.isnan(diff)]
                if len(diffs) > 0: 
                  boot_means = np.empty(2000)
                  for i in range(2000):
                    idx = rng.integers(0, len(diffs), len(diffs))
                    boot_means[i] = diffs[idx].mean()
                boot_ci_low = np.percentile(boot_means, 2.5)
                boot_ci_high = np.percentile(boot_means, 97.5)
            except Exception as e:
                print("Bootstrap error:", e)        
        except Exception as e:
            wilcoxon_error = str(e)                          

        # Primary Test based on normality
        if is_normal:
            test_name = "Paired T-Test"
            final_p = t_p
        else:
            test_name = "Wilcoxon Signed-Rank"
            final_p = wilcoxon_p

        results.append({
            # General
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
            "CI Lower": ci[0],
            "CI Upper": ci[1],

            # Non-Parametric Outputs
            "W-Stat": w_stat,
            "wilcoxon P-Value": wilcoxon_p,
            "wilcoxon r": wilcoxon_r,
            "r (Effect Size)": effect_size_label_wilcoxon_r,
            "wilcoxon RBC": RBC,
            "RBC (Effect Size)": effect_size_label_RBC,
            "wilcoxonCLES": CLES,           
            "bootstrap_ci_lower": boot_ci_low,
            "bootstrap_ci_upper": boot_ci_high,
            "wilcoxon Error": wilcoxon_error,

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
  # Combine all result tables
  all_results = pd.concat([sbert_results, comet_results], ignore_index=True)
  
  # Export CSV
  ensure_dir(OUT_ROOT)
  timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
  stats_folder = ensure_dir(os.path.join(OUT_ROOT, "stats"))
  pair_csv = os.path.join(stats_folder, f"pairwise_tests_{timestamp}.csv")
  all_results.to_csv(pair_csv, index=False, encoding="utf-8")
  print("[OK] Saved pairwise test results:", pair_csv)

def descriptive_stats_by_group(wide_subset, metric, groups):
    rows = []
    for g in groups:
        col = f"{metric}_{g}"
        if col not in wide_subset.columns:
            rows.append({"group":g,"metric":metric,"n":0,"mean":np.nan,"sd":np.nan,"median":np.nan})
            continue
        series = wide_subset[col].dropna().astype(float)
        rows.append({
            "group": g,
            "metric": metric,
            "n": int(series.count()),
            "mean": float(series.mean()) if len(series)>0 else np.nan,
            "sd": float(series.std(ddof=1)) if len(series)>1 else np.nan,
            "median": float(series.median()) if len(series)>0 else np.nan
        })
    return pd.DataFrame(rows)

# ---------------- Pipeline Core ----------------
pipeline()
