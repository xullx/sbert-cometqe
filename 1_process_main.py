#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs:
 - ./qe_outputs/group__{group}/metrics_group_{group}.csv
 - ./qe_outputs/metrics_all_groups.csv

Dependencies:
    pip install pandas numpy sentence-transformers unbabel-comet
"""

import os
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# --------- CONFIG ----------
DATASET_PATH = "./dataset.csv"   # <-- LOCAL path to CSV
OUT_ROOT = "./qe_outputs"

# Feature toggles
SBERT_ENABLE = True
COMET_QE_ENABLE = True

# SBERT / COMET settings
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH_SIZE = 64
COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"
COMET_QE_BATCH_SIZE = 1 #Smaller = less work. (8) crashes. (1) is working.

# --- SBERT load (optional) ---
_HAS_SBERT = False
_sbert_model = None
if SBERT_ENABLE:
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL)
        _HAS_SBERT = True
        print("[INFO] SBERT loaded:", SBERT_MODEL)
    except Exception as e:
        print("[WARN] SBERT not available:", e)
        _HAS_SBERT = False

# --- COMET-QE load (optional) ---
_HAS_COMET = False
_comet_obj = None
if COMET_QE_ENABLE:
    try:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model(COMET_QE_MODEL)
        _comet_obj = load_from_checkpoint(ckpt)
        _HAS_COMET = True
        print("[INFO] COMET-QE model ready:", COMET_QE_MODEL)
    except Exception as e:
        print("[WARN] COMET-QE not available or download failed:", e)
        _HAS_COMET = False

# ---- Helpers ----
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def sbert_batch(src_texts, hyp_texts, batch_size=SBERT_BATCH_SIZE):
    if not _HAS_SBERT:
        return [np.nan]*len(src_texts)
    import torch
    model = _sbert_model
    model.max_seq_length = 512
    src_embs, hyp_embs = [], []
    for i in range(0, len(src_texts), batch_size):
        chunk = src_texts[i:i+batch_size]
        emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
        src_embs.append(emb)
    for i in range(0, len(hyp_texts), batch_size):
        chunk = hyp_texts[i:i+batch_size]
        emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
        hyp_embs.append(emb)
    src_all = torch.cat(src_embs, dim=0)
    hyp_all = torch.cat(hyp_embs, dim=0)
    sims = (src_all * hyp_all).sum(dim=1).cpu().numpy()
    return sims.tolist()

def cometqe_batch(src_texts, hyp_texts, batch_size=COMET_QE_BATCH_SIZE):
    if not _HAS_COMET: return [np.nan] * len(src_texts)
    data = [{"src": s or "", "mt": t or ""} for s,t in zip(src_texts, hyp_texts)]
    try:
        out = _comet_obj.predict(data, batch_size=batch_size, gpus=0, progress_bar=True)
        if isinstance(out, dict) and "scores" in out:
            return [float(x) for x in out["scores"]]
        if isinstance(out, list):
            return [float(x.get("score", np.nan)) if isinstance(x, dict) else float(np.nan) for x in out]
    except Exception as e:
        print("[WARN] COMET predict error:", e)
    return [np.nan] * len(src_texts)

def process_group(df_group, group_name, out_root):
    ensure_dir(out_root)
    gdf = df_group.copy().reset_index(drop=True)

    srcs = gdf["sub_jap"].fillna("").astype(str).tolist()
    hyp_en = gdf["sub_trad_en"].fillna("").astype(str).tolist() if "sub_trad_en" in gdf.columns else ["" for _ in srcs]
    hyp_sp = gdf["sub_trad_sp"].fillna("").astype(str).tolist() if "sub_trad_sp" in gdf.columns else ["" for _ in srcs]

    if _HAS_SBERT:
        try:
            print(f"[INFO] Computing SBERT for group {group_name} (EN)")
            gdf["sbert_jap_en"] = sbert_batch(srcs, hyp_en)
            print(f"[INFO] Computing SBERT for group {group_name} (SP)")
            gdf["sbert_jap_sp"] = sbert_batch(srcs, hyp_sp)
        except Exception as e:
            print("[WARN] SBERT failed:", e)
            gdf["sbert_jap_en"] = np.nan; gdf["sbert_jap_sp"] = np.nan
    else:
        gdf["sbert_jap_en"] = np.nan; gdf["sbert_jap_sp"] = np.nan

    if _HAS_COMET:
        try:
            print(f"[INFO] Computing COMET-QE for group {group_name} (EN)")
            gdf["comet_jap_en"] = cometqe_batch(srcs, hyp_en)
            print(f"[INFO] Computing COMET-QE for group {group_name} (SP)")
            gdf["comet_jap_sp"] = cometqe_batch(srcs, hyp_sp)
        except Exception as e:
            print("[WARN] COMET failed:", e)
            gdf["comet_jap_en"] = np.nan; gdf["comet_jap_sp"] = np.nan
    else:
        gdf["comet_jap_en"] = np.nan; gdf["comet_jap_sp"] = np.nan

    out_csv = os.path.join(out_root, f"metrics_group_{group_name}.csv")
    gdf.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved metrics CSV -> {out_csv}")
    return out_csv, gdf

def main():
    ensure_dir(OUT_ROOT)
    print("[INFO] Reading dataset:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH, engine="python", on_bad_lines="warn")
    groups = sorted(df["group"].dropna().unique())
    all_frames = []
    for g in groups:
        group_dir = os.path.join(OUT_ROOT, f"group__{g}")
        ensure_dir(group_dir)
        subdf = df[df["group"] == g].copy()
        print(f"[INFO] Processing group={g} ({len(subdf)} rows)")
        csv_path, gdf = process_group(subdf, g, group_dir)
        all_frames.append(gdf)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_csv = os.path.join(OUT_ROOT, "metrics_all_groups.csv")
        combined.to_csv(combined_csv, index=False, encoding="utf-8")
        print("[OK] Saved combined CSV:", combined_csv)
    else:
        print("[WARN] No data processed.")
try:
    main()
except Exception as e:
    print("ERROR:", e)