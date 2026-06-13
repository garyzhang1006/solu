"""
Solu - AI Chemistry Tutor
Train aqueous-solubility (logS) regressors on the ESOL / Delaney dataset
(MoleculeNet; Delaney 2004, J. Chem. Inf. Comput. Sci.).

Descriptors are computed with rdkit.Chem.rdMolDescriptors / Crippen -- the SAME
C++ routines RDKit-JS (MinimalLib WASM) exposes via get_descriptors(), so the
feature vector computed in the browser at inference time matches training exactly.

Outputs (into ../web/ for the standalone app, ../model/ for the repo):
  web/model.onnx           best tree model, ONNX (run via onnxruntime-web)
  web/artifacts.json       feature spec, linear surrogate, scaler, dataset stats,
                           metrics, example-molecule library
  web/parity_fixture.json  per-molecule descriptors + predictions for JS parity test
  model/metrics.txt        human-readable metrics
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Crippen, Descriptors

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
DATA = ROOT / "data"
WEB.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---- Feature spec -----------------------------------------------------------
# (key shown to user, RDKit-JS get_descriptors() key, human label, unit, teaching note)
FEATURES = [
    ("clogp",      "CrippenClogP",       "LogP (lipophilicity)", "",      "Higher = more 'oily' / less water-friendly. Pushes solubility DOWN."),
    ("amw",        "amw",                "Molecular weight",     "g/mol", "Bigger molecules are generally harder to dissolve."),
    ("tpsa",       "tpsa",               "Polar surface area",   "A^2",   "More polar surface = more water contact. Pushes solubility UP."),
    ("hbd",        "NumHBD",             "H-bond donors",        "",      "Groups that donate H-bonds to water (e.g. -OH, -NH). Raise solubility."),
    ("hba",        "NumHBA",             "H-bond acceptors",     "",      "Atoms that accept H-bonds from water (e.g. O, N). Raise solubility."),
    ("rotb",       "NumRotatableBonds",  "Rotatable bonds",      "",      "Flexibility; weakly linked to solubility."),
    ("aromrings",  "NumAromaticRings",   "Aromatic rings",       "",      "Flat hydrophobic rings (like benzene). Tend to lower solubility."),
    ("fcsp3",      "FractionCSP3",       "Fraction sp3 C",       "",      "Share of 3D (saturated) carbons; more 3D often dissolves better."),
    ("rings",      "NumRings",           "Ring count",           "",      "Total rings in the structure."),
]
FEAT_KEYS = [f[0] for f in FEATURES]
JS_KEYS = [f[1] for f in FEATURES]


def descriptors(mol):
    """Compute the feature vector with the exact RDKit routines RDKit-JS mirrors."""
    return [
        Crippen.MolLogP(mol),                        # CrippenClogP
        Descriptors.MolWt(mol),                      # amw (average mol weight)
        rdMolDescriptors.CalcTPSA(mol),              # tpsa (default: no S/P)
        rdMolDescriptors.CalcNumHBD(mol),            # NumHBD
        rdMolDescriptors.CalcNumHBA(mol),            # NumHBA
        rdMolDescriptors.CalcNumRotatableBonds(mol), # NumRotatableBonds (default/strict)
        rdMolDescriptors.CalcNumAromaticRings(mol),  # NumAromaticRings
        rdMolDescriptors.CalcFractionCSP3(mol),      # FractionCSP3
        rdMolDescriptors.CalcNumRings(mol),          # NumRings
    ]


def build_dataset():
    df = pd.read_csv(DATA / "delaney-processed.csv")
    target_col = "measured log solubility in mols per litre"
    X, y, names = [], [], []
    dropped = 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            dropped += 1
            continue
        X.append(descriptors(mol))
        y.append(float(row[target_col]))
        names.append(str(row["Compound ID"]))
    print(f"parsed {len(X)} molecules, dropped {dropped} unparseable")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), names


def evaluate(model, Xtr, Xte, ytr, yte, cv_X, cv_y):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae = float(mean_absolute_error(yte, pred))
    r2 = float(r2_score(yte, pred))
    cv = cross_val_score(model, cv_X, cv_y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                         scoring="r2")
    return {"rmse": rmse, "mae": mae, "r2": r2, "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std())}


def main():
    X, y, names = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results = {}

    # Linear (standardized) -- the interpretable Delaney-style surrogate
    scaler = StandardScaler().fit(Xtr)
    lin = LinearRegression()
    lin_metrics = evaluate(lin, scaler.transform(Xtr), scaler.transform(Xte), ytr, yte,
                           scaler.transform(X), y)
    results["linear"] = lin_metrics

    rf = RandomForestRegressor(n_estimators=400, max_depth=18, min_samples_leaf=2,
                               n_jobs=-1, random_state=RANDOM_STATE)
    results["random_forest"] = evaluate(rf, Xtr, Xte, ytr, yte, X, y)

    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                    subsample=0.9, random_state=RANDOM_STATE)
    results["gradient_boosting"] = evaluate(gbm, Xtr, Xte, ytr, yte, X, y)

    print("\n=== model comparison (held-out 20% test) ===")
    for k, m in results.items():
        print(f"{k:18s} RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
              f"CV-R2={m['cv_r2_mean']:.3f}+/-{m['cv_r2_std']:.3f}")

    # pick best tree model by test R2
    best_name = max(["random_forest", "gradient_boosting"], key=lambda k: results[k]["r2"])
    best_model = rf if best_name == "random_forest" else gbm
    # refit best on ALL data for shipping
    best_model.fit(X, y)
    print(f"\nbest tree model: {best_name} -> shipped as model.onnx (refit on full data)")

    # refit linear surrogate on ALL data for the explanation panel
    full_scaler = StandardScaler().fit(X)
    lin_full = LinearRegression().fit(full_scaler.transform(X), y)

    # ---- export ONNX ----
    from skl2onnx import to_onnx
    onx = to_onnx(best_model, X[:1].astype(np.float32), target_opset=18)
    (WEB / "model.onnx").write_bytes(onx.SerializeToString())
    print(f"wrote web/model.onnx ({(WEB / 'model.onnx').stat().st_size} bytes)")

    # ---- example molecule library (curated for teaching) ----
    library = [
        ("Water", "O"), ("Ethanol", "CCO"),
        ("Table sugar (sucrose)",
         "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"),
        ("Glucose", "OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Benzene", "c1ccccc1"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("DDT", "ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1"),
        ("Cholesterol", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),
        ("Acetic acid", "CC(=O)O"),
        ("Vitamin C", "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"),
    ]
    examples = []
    for name, smi in library:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"  WARN: example '{name}' did not parse, skipping")
            continue
        examples.append({"name": name, "smiles": Chem.MolToSmiles(m)})

    # ---- dataset feature stats (for gauges / context) ----
    stats = {}
    for i, key in enumerate(FEAT_KEYS):
        col = X[:, i]
        stats[key] = {"min": float(col.min()), "max": float(col.max()),
                      "mean": float(col.mean()), "std": float(col.std())}
    y_stats = {"min": float(y.min()), "max": float(y.max()),
               "mean": float(y.mean()), "std": float(y.std())}

    artifacts = {
        "dataset": {"name": "ESOL / Delaney (MoleculeNet)", "n": int(len(X)),
                    "target": "log10 aqueous solubility (mol/L)",
                    "citation": "Delaney, J.S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. J. Chem. Inf. Comput. Sci. 2004."},
        "features": [{"key": k, "js_key": jk, "label": lab, "unit": u, "note": note}
                     for (k, jk, lab, u, note) in FEATURES],
        "feature_order": FEAT_KEYS,
        "js_keys": JS_KEYS,
        "scaler": {"mean": full_scaler.mean_.tolist(), "scale": full_scaler.scale_.tolist()},
        "linear": {"coef": lin_full.coef_.tolist(), "intercept": float(lin_full.intercept_)},
        "feature_stats": stats,
        "target_stats": y_stats,
        "metrics": results,
        "best_model": best_name,
        "examples": examples,
    }
    (WEB / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    print("wrote web/artifacts.json")

    # ---- parity fixture: descriptors + predictions for JS-side verification ----
    fixture_smiles = [smi for _, smi in library[:12]]
    fixture = []
    sess = None
    in_name = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession((WEB / "model.onnx").as_posix())
        in_name = sess.get_inputs()[0].name
    except Exception as e:
        print("onnxruntime check skipped:", e)
    for smi in fixture_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        d = descriptors(m)
        rec = {"smiles": Chem.MolToSmiles(m),
               "descriptors": {k: d[i] for i, k in enumerate(FEAT_KEYS)},
               "js_descriptors": {jk: d[i] for i, jk in enumerate(JS_KEYS)}}
        if sess is not None:
            onnx_pred = float(sess.run(None, {in_name: np.asarray([d], dtype=np.float32)})[0].ravel()[0])
            rec["onnx_logS"] = onnx_pred
        rec["sklearn_logS"] = float(best_model.predict([d])[0])
        z = (np.asarray(d) - full_scaler.mean_) / full_scaler.scale_
        rec["linear_logS"] = float(lin_full.intercept_ + np.dot(z, lin_full.coef_))
        fixture.append(rec)
    (WEB / "parity_fixture.json").write_text(json.dumps(fixture, indent=2))
    print(f"wrote web/parity_fixture.json ({len(fixture)} molecules)")

    # ---- human-readable metrics ----
    lines = ["Solu - solubility model metrics (held-out 20% test)\n"]
    for k, m in results.items():
        lines.append(f"{k:18s} RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
                     f"CV-R2={m['cv_r2_mean']:.3f}+/-{m['cv_r2_std']:.3f}")
    lines.append(f"\nshipped model: {best_name}")
    lines.append(f"dataset: ESOL/Delaney, n={len(X)}")
    (HERE / "metrics.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
