#2_make_wide_matrix
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_wide_matrix.py
Generates 'wide' matrices by time_stamp with columns <metric>__<group>
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_INPUT = "./metrics_all_groups.csv"
OUT_DIR = "./qe_outputs"

# Default list of metrics to pivot (adjust if necessary)
DEFAULT_METRICS = [
    "sbert_jap_en","sbert_jap_sp",
    "comet_jap_en","comet_jap_sp",
]

def find_metric(df, dm):
    cols = df.columns.tolist()
    m = {}
    for d in dm:
        if d in cols:
            m[d] = d
            continue
    return m

def pivot_to_wide(df, t, g, m):
    pivots, metrics = [], []
    for a, b in m.items():
        metrics.append(b)
        tmp = df[[t, g, b]].copy()
        pivot = tmp.pivot(index=t, columns=g, values=b)
        pivot.columns = [f"{b}_{x}" for x in pivot.columns]
        pivots.append(pivot)
    wide = pd.concat(pivots, axis=1).reset_index().rename(columns={"index": t})
    return wide, metrics

def fill_na(df):
    print(df.shape)
    mask_en = df.iloc[:, 3].fillna("").str.strip() == ""    
    affected_en = df.loc[mask_en, [
      "time_stamp",
      "group",
      "sub_jap",
      "sub_trad_en",
      "sbert_jap_en",
      "comet_jap_en"
    ]]
    df.loc[mask_en, ["sbert_jap_en","comet_jap_en"]] = np.nan
    print("\n=== EN rows affected ===")
    print(affected_en)
    print(f"\nTotal EN affected rows: {len(affected_en)}")
    mask_sp = df.iloc[:, 4].fillna("").str.strip() == ""
    affected_sp = df.loc[mask_sp, [
        "time_stamp",
        "group",
        "sub_jap",
        "sub_trad_sp",
        "sbert_jap_sp",
        "comet_jap_sp"
    ]]
    print("\n=== SP rows affected ===")
    print(affected_sp)
    print(f"\nTotal SP affected rows: {len(affected_sp)}")    
    df.loc[mask_sp, ["sbert_jap_sp","comet_jap_sp"]] = np.nan
    return df

def main():
    df = fill_na(pd.read_csv(Path(DEFAULT_INPUT), engine="python", on_bad_lines="warn", encoding="utf-8"))  
    c = df.columns.str.lower().tolist()
    groups = sorted(df[c[1]].dropna().unique().tolist())
    metric_map = find_metric(df, DEFAULT_METRICS)
    wide, metrics = pivot_to_wide(df, c[0], c[1], metric_map)
    out = Path(OUT_DIR) / "metrics_wide_by_timestamp_fillna.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False, encoding="utf-8")
    print(f"[OK] Saved wide all -> {out} (rows: {len(wide)})")
    # summary
    summary = {
        "input_rows": len(df),
        "distinct_time_stamps": wide["time_stamp"].nunique() if "time_stamp" in wide.columns else wide.iloc[:,0].nunique(),
        "groups_detected": groups,
        "metrics_used_actual": metrics,
        "wide_all": str(out),
    }
    print(summary)
main()