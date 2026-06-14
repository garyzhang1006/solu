#!/usr/bin/env python3
"""
Build web/grounding.json -- the reference corpus the AI tutor cites.

The tutor (Sonnet 4.6) does not invent solubility numbers. For every molecule a
user predicts, the browser finds the structurally nearest REAL molecules in this
corpus -- each with a published, measured logS -- and hands them to the model as
evidence. So the tutor explains a prediction by pointing at actual lab data.

"Train the tutor on more datasets" = grow this corpus. Add more keys to
SOLUBILITY_KEYS (any logS dataset in datasets/MANIFEST.json) and re-run:

    python tutor/build_grounding.py

The descriptor vector is computed with the SAME rdkit routine model/train.py uses,
in the SAME feature_order as web/artifacts.json -- so the browser can measure
distance in the model's own (scaler-standardized) feature space.
"""
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Crippen, Descriptors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
DATASETS = ROOT / "datasets"

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(DATASETS))
from load import load  # datasets/load.py -> DataFrame[smiles, target]

# logS (log10 aqueous solubility, mol/L) datasets, safe to union.
# Grow this list to "train the tutor on more datasets".
SOLUBILITY_KEYS = ["esol", "aqsoldb"]

# feature_order must match web/artifacts.json exactly.
FEATURE_ORDER = ["clogp", "amw", "tpsa", "hbd", "hba", "rotb", "aromrings", "fcsp3", "rings"]

# How many molecules to keep. Stratified across the logS range so the nearest-
# neighbour search has anchors everywhere, not just near the dataset mean.
TARGET_N = 1400
N_BINS = 28  # logS buckets; ~50 molecules each


def descriptors(mol):
    """Exact feature vector model/train.py trains on (RDKit-JS mirrors these C++ calls)."""
    return [
        round(Crippen.MolLogP(mol), 3),                    # clogp
        round(Descriptors.MolWt(mol), 2),                  # amw
        round(rdMolDescriptors.CalcTPSA(mol), 2),          # tpsa
        rdMolDescriptors.CalcNumHBD(mol),                  # hbd
        rdMolDescriptors.CalcNumHBA(mol),                  # hba
        rdMolDescriptors.CalcNumRotatableBonds(mol),       # rotb
        rdMolDescriptors.CalcNumAromaticRings(mol),        # aromrings
        round(rdMolDescriptors.CalcFractionCSP3(mol), 3),  # fcsp3
        rdMolDescriptors.CalcNumRings(mol),                # rings
    ]


def featurize(df, source):
    """SMILES + measured logS -> {smiles, logS, source, x[9]}; drop unparseable."""
    out, dropped = [], 0
    for smiles, logS in zip(df["smiles"], df["target"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            dropped += 1
            continue
        out.append({
            "smiles": Chem.MolToSmiles(mol),  # canonical, so the browser can de-dupe
            "logS": round(float(logS), 2),
            "source": source,
            "x": descriptors(mol),
        })
    print(f"  {source:10s} featurized {len(out)}, dropped {dropped} unparseable")
    return out


def stratified_sample(records, target_n, n_bins):
    """Keep an even spread across the logS range so neighbours exist everywhere."""
    if len(records) <= target_n:
        return records
    lo = min(r["logS"] for r in records)
    hi = max(r["logS"] for r in records)
    span = (hi - lo) or 1.0
    buckets = [[] for _ in range(n_bins)]
    for r in records:
        i = min(n_bins - 1, int((r["logS"] - lo) / span * n_bins))
        buckets[i].append(r)
    per = max(1, target_n // n_bins)
    kept = []
    for b in buckets:
        # even stride through each bucket (deterministic, no RNG)
        if len(b) <= per:
            kept.extend(b)
        else:
            step = len(b) / per
            kept.extend(b[int(i * step)] for i in range(per))
    return kept


def main():
    print(f"Building grounding corpus from: {', '.join(SOLUBILITY_KEYS)}")
    records, seen = [], set()
    for key in SOLUBILITY_KEYS:
        for r in featurize(load(key), key):
            if r["smiles"] in seen:
                continue
            seen.add(r["smiles"])
            records.append(r)
    print(f"  union (de-duped by SMILES): {len(records)} molecules")

    sample = stratified_sample(records, TARGET_N, N_BINS)
    sample.sort(key=lambda r: r["logS"])

    payload = {
        "feature_order": FEATURE_ORDER,
        "datasets": SOLUBILITY_KEYS,
        "n": len(sample),
        "molecules": sample,
    }
    out = WEB / "grounding.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(sample)} molecules, {kb:.0f} KB)")
    print(f"  logS range [{sample[0]['logS']:.2f}, {sample[-1]['logS']:.2f}]")


if __name__ == "__main__":
    main()
