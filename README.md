# Solu — *See why molecules dissolve* 💧

An AI chemistry tutor that predicts a molecule's **water solubility** from its structure and **teaches the structure–property reasoning behind the answer** — running 100% in your browser, with no server, no API, and no LLM.

Built for **DSH Hacks V1** (theme: *AI × STEM Education*).

> Type or pick any molecule → RDKit-JS draws it and computes its descriptors → a gradient-boosting model (trained on a published chemistry dataset, exported to ONNX) predicts log-solubility client-side → an interpretable linear model explains *which properties* pushed the answer up or down.

---


## What it does

| Feature | Description |
|---|---|
| **Live prediction** | Enter a SMILES string (or pick from 14 example molecules); get log S (mol/L), a mg/L value, and a qualitative class. |
| **Structure drawing** | RDKit-JS renders the 2D structure in-browser. |
| **"Why?" panel** | Per-property contributions (LogP, polar surface area, H-bond donors/acceptors, …) from an interpretable linear surrogate, relative to a typical molecule. |
| **Molecular fingerprint** | The 9 computed descriptors with where each sits across the dataset. |
| **Quiz mode** | Guess soluble / insoluble before the model reveals — active learning. |
| **Lessons** | Plain-English explanations of polarity, hydrogen bonding, and lipophilicity. |

## Model & results

Nine interpretable RDKit descriptors (the Delaney-ESOL feature family): Crippen LogP, molecular weight, topological polar surface area, H-bond donors, H-bond acceptors, rotatable bonds, aromatic rings, fraction sp³ carbon, ring count.

Held-out 20% test set (and 5-fold CV):

| Model | RMSE (log units) | MAE | R² | CV-R² |
|---|---|---|---|---|
| Linear regression (classic Delaney-style) | 1.038 | 0.781 | 0.772 | 0.787 |
| Random forest | 0.799 | 0.549 | 0.865 | 0.890 |
| **Gradient boosting (shipped)** | **0.746** | **0.538** | **0.882** | **0.893** |

The shipped gradient-boosting model improves RMSE from **1.04 → 0.75** over the textbook linear ESOL equation on the same split — a real, measured gain.

## Correctness: the train/inference parity gates

The model trains in **Python (RDKit + scikit-learn)** but runs in the **browser (RDKit-JS + ONNX Runtime Web)**. If the two RDKit builds compute descriptors differently, browser predictions silently go wrong. We caught and fixed exactly this (the `NumHBA` definition changed between RDKit versions), then locked it down with three automated gates:

1. **`model/parity_test.cjs`** — RDKit-JS vs RDKit-Python descriptors agree to `1e-5` (rounding only). *Requires pinning Python `rdkit==2025.3.4` to match RDKit-JS `2025.03.4`.*
2. **ONNX vs scikit-learn** — predictions agree to `2e-6` (checked in `train.py`'s fixture).
3. **`model/e2e_test.cjs`** — the full browser chain (RDKit-JS → ONNX Runtime Web) reproduces the Python prediction **bit-for-bit (diff = 0.0)**.

Browser predictions are therefore provably identical to the trained model.

## Run it

**Web app (recommended):**
```bash
cd web
python3 -m http.server 8910
# open http://localhost:8910
```

**Single-file build** — `dist/solu.html` has the model + data embedded; serve it anywhere (or open locally) and it runs with only the RDKit/ONNX CDN scripts.

## Reproduce the model

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -L -o data/delaney-processed.csv \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
python model/train.py            # trains, compares models, exports web/model.onnx + artifacts
npm install                      # @rdkit/rdkit + onnxruntime-web (for the gates)
node model/parity_test.cjs       # descriptor parity gate
node model/e2e_test.cjs          # end-to-end browser-chain gate
python model/build_standalone.py # bundle dist/solu.html
```

## Project layout

```
solu/
├── web/
│   ├── index.html          # the app (RDKit-JS + ONNX Runtime Web)
│   ├── model.onnx          # trained gradient-boosting model
│   ├── artifacts.json      # feature spec, scaler, linear surrogate, metrics, examples
│   └── parity_fixture.json # per-molecule descriptors + predictions for the gates
├── dist/solu.html          # single-file portable build
├── model/
│   ├── train.py            # data → descriptors → train → export
│   ├── parity_test.cjs     # RDKit-JS vs RDKit-Python
│   ├── e2e_test.cjs        # RDKit-JS + ONNX Runtime Web vs Python
│   └── build_standalone.py # bundler
├── data/delaney-processed.csv
├── docs/                   # one-page description + demo script
├── requirements.txt
└── README.md
```

## Tech

RDKit & RDKit-JS (cheminformatics), scikit-learn (gradient boosting), skl2onnx + ONNX Runtime Web (portable inference), vanilla HTML/CSS/JS (no framework).

## Credits & citation

- Delaney, J. S. *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.* J. Chem. Inf. Comput. Sci. **44**, 1000–1005 (2004).
- Wu, Z. et al. *MoleculeNet: A Benchmark for Molecular Machine Learning.* Chem. Sci. **9**, 513–530 (2018).
- RDKit: Open-source cheminformatics (rdkit.org).

Predictions are educational model estimates, not laboratory values.

## License

MIT — see `LICENSE`.
