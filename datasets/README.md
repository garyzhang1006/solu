# Datasets

Public datasets for training / benchmarking the Solu water-solubility model. Downloaded from the original academic sources (see `download.sh` to re-fetch). Aqueous-solubility records across the primary sets: **~11,110**.

## Aqueous solubility (primary target: log S, mol/L)

| File | Dataset | Rows | SMILES col | Target col | logS range | Source |
|---|---|---|---|---|---|---|
| `esol.csv` | ESOL / Delaney | 1,128 | `smiles` | `measured log solubility in mols per litre` | -11.6 to 1.58 | [link](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) |
| `aqsoldb.csv` | AqSolDB | 9,982 | `SMILES` | `Solubility` | -13.17 to 2.14 | [link](https://dataverse.harvard.edu/api/access/datafile/3407241) |

## Related molecular properties (different target, useful for transfer/extension)

| File | Dataset | Rows | SMILES col | Target col | Property |
|---|---|---|---|---|---|
| `freesolv.csv` | FreeSolv (SAMPL) | 642 | `smiles` | `expt` | experimental hydration free energy (kcal/mol) [related property] |
| `lipophilicity.csv` | Lipophilicity | 4,200 | `smiles` | `exp` | experimental octanol/water logD at pH 7.4 [related property] |

## Notes

- **AqSolDB** is tab-separated and is itself curated from 9 source datasets (incl. Huuskonen, ESOL, PHYSPROP). It is the largest single aqueous-solubility set here (~10k compounds) and the best target for a stronger model.
- The shipped model currently trains on **ESOL** (see `../model/train.py`). To retrain on AqSolDB or a union, use `load.py` to get standardized `(smiles, logS)` frames.
- `load.py` exposes every dataset as a clean `(smiles, target)` DataFrame.

## Citations

- **ESOL / Delaney** — Delaney, J.S. ESOL. J. Chem. Inf. Comput. Sci. 44, 1000-1005 (2004). License: Public (MoleculeNet).
- **AqSolDB** — Sorkun, M.C., Khetan, A., Er, S. AqSolDB. Sci. Data 6, 143 (2019). License: CC0 1.0 (Harvard Dataverse).
- **FreeSolv (SAMPL)** — Mobley, D.L., Guthrie, J.P. FreeSolv. J. Comput. Aided Mol. Des. 28, 711-720 (2014). License: Public (MoleculeNet).
- **Lipophilicity** — Wu, Z. et al. MoleculeNet. Chem. Sci. 9, 513-530 (2018). License: Public (MoleculeNet).
