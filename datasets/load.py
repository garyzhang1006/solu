#!/usr/bin/env python3
"""
Standardized loader for the Solu datasets.

Each public dataset stores its SMILES and target under different column names
(and AqSolDB is tab-separated). This module reads MANIFEST.json and exposes every
dataset as a clean DataFrame with two columns: `smiles` and `target`.

    from load import load, list_datasets, load_solubility
    df = load("aqsoldb")          # -> columns: smiles, target  (target = logS)
    big = load_solubility()       # ESOL + AqSolDB unioned, de-duplicated by SMILES

CLI:
    python load.py                # print a summary table of all datasets
    python load.py aqsoldb        # print head + stats for one dataset
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "MANIFEST.json").read_text())
BY_KEY = {m["file"].replace(".csv", ""): m for m in MANIFEST}

# datasets whose target is aqueous solubility (logS), safe to union for training
SOLUBILITY_KEYS = ["esol", "aqsoldb"]


def list_datasets():
    """Return the manifest (list of dataset metadata dicts)."""
    return MANIFEST


def load(key):
    """Load one dataset by key (filename without .csv) as DataFrame[smiles, target]."""
    if key not in BY_KEY:
        raise KeyError(f"unknown dataset {key!r}; options: {sorted(BY_KEY)}")
    m = BY_KEY[key]
    df = pd.read_csv(HERE / m["file"], sep=m["sep"])
    out = pd.DataFrame({
        "smiles": df[m["smiles"]].astype(str).str.strip(),
        "target": pd.to_numeric(df[m["target"]], errors="coerce"),
    }).dropna()
    return out.reset_index(drop=True)


def load_solubility(keys=None, dedup=True):
    """Union the aqueous-solubility datasets into one DataFrame[smiles, target=logS]."""
    keys = keys or SOLUBILITY_KEYS
    frames = [load(k) for k in keys]
    df = pd.concat(frames, ignore_index=True)
    if dedup:
        df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    return df


def _summary():
    print(f"{'key':14s} {'dataset':18s} {'rows':>6s}  {'target range':>16s}  property")
    print("-" * 90)
    for m in MANIFEST:
        key = m["file"].replace(".csv", "")
        df = load(key)
        rng = f"[{df['target'].min():.2f}, {df['target'].max():.2f}]"
        print(f"{key:14s} {m['name']:18s} {len(df):6d}  {rng:>16s}  {m['property']}")
    big = load_solubility()
    print("-" * 90)
    print(f"{'(union logS)':14s} {'ESOL+AqSolDB':18s} {len(big):6d}  "
          f"[{big['target'].min():.2f}, {big['target'].max():.2f}]  de-duplicated by SMILES")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1]
        df = load(key)
        print(f"{key}: {len(df)} rows")
        print(df.head(8).to_string(index=False))
        print(f"\ntarget: mean={df['target'].mean():.3f}  min={df['target'].min():.2f}  max={df['target'].max():.2f}")
    else:
        _summary()
