# Yale Chemistry Project — NMR property prediction

Predict molecular properties (`gap_ev`, logP, HOMO/LUMO) from 1H/13C NMR spectra.
The core idea is an **NMF dictionary representation**: learn a shared basis of
spectral "motifs" from a large unlabeled NMR corpus, project any molecule's
spectra into that basis to get interpretable codes, then train a regressor on
the codes. The codes map back to recognizable chemistry (via SHAP + motif
lookup), so predictions are explainable.

## Layout

| Path | What it is |
|------|-----------|
| `nmrlib/` | Shared library: dataset registry, feature definitions, metrics, model space. Import from both notebooks and scripts. |
| `*.ipynb` | Thin notebooks (see below) — config + `nmrlib` calls + plots. |
| Pipeline scripts (repo root) | Command-line stages of the full workflow (see the pipeline map). |
| `Visualization/` | Standalone plotting / interpretation scripts. |
| `Datasets/` | Local data home (gitignored). Loaders look here first. |
| `figures/` | All generated plots (gitignored contents aside from committed keepers). |
| `references/` | Papers backing the descriptors (An 2014, Voutchkova 2010, …). |
| `docker/` | Containerization for running the tuning sweep on the cloud. |

## Notebooks

1. **`create_features_nmr.ipynb`** — load a raw-spectra dataset, attach every
   feature block (An 2014 1H descriptors, 13C bins, peak stats, NMF codes) via
   `nmrlib.featurize`, save a featurized pickle.
2. **`feature_comparison.ipynb`** — cross-validate each named feature set against
   a target to see which representation carries the most signal.
3. **`ml_workflow.ipynb`** — grid-search models on one feature set, evaluate on a
   held-out test set, interpret with SHAP, visualize the top NMF motifs.

All three are **thin drivers**: switch datasets/targets in the single config
cell. Feature-set columns and metrics come from `nmrlib`, so nothing drifts
between notebooks.

## `nmrlib` at a glance

```python
from nmrlib import load_dataset, featurize, feature_sets, compare_feature_sets

df = load_dataset("alberts_10k")          # registry name or path; normalizes
                                          # qchem_gap_ev->gap_ev, canonical_smiles->smiles
df = featurize(df, dictionary_path=...)   # attach all NMR feature blocks
sets = feature_sets(df)                   # named column groups, derived from df
```

- **`nmrlib.data`** — `DATASETS` registry (looks in `Datasets/` then `~/Downloads`),
  `load_dataset` (alias normalization + duplicate-column dedupe).
- **`nmrlib.features`** — `feature_sets`, `nmf_cols` (NMF code count derived from
  the frame, never hardcoded), `featurize`.
- **`nmrlib.metrics`** — `regression_metrics`.
- **`nmrlib.models`** — `default_models`, `grid_search_space`, `compare_feature_sets`.

## Pipeline map (command-line stages)

The scripts form a layered pipeline. Foundations first, then the workflow:

**Foundations** (feature builders, imported everywhere):
- `create_nmr_dictionary_features.py` — soft-peak matrices + NMF dictionary learner
- `an2014_nmr.py` — An et al. (2014) 1H QSDAR descriptors
- `add_c_nmr_bins.py` — 13C integration bins + broadness

**Workflow:**
1. `tune_representation.py` — Optuna sweep for the NMF representation
   (held-out `gap_ev` R² primary; reconstruction / logP / functional-group F1 guardrails).
2. `analyze_sweep.py` — read the sweep study, produce the Pareto front, importances, plots.
3. `train_gap_model.py` — train the final HistGradientBoosting model from the best sweep trial.
4. `train_gap_from_dict.py` — full pipeline: fit the tuned NMF dictionary on a large
   *unlabeled* corpus, transform a labeled set, HPO-tune the booster, report honest test R².
5. `build_dict_codes.py` — apply a tuned dictionary to a new dataset (append codes + errors).
6. `cross_dataset_gap.py` — transfer test: fit representation on dataset A, evaluate on B.
7. `benchmark_functional_groups.py` — micro-F1 functional-group benchmark (Alberts setup).

Run the tuning sweep on the cloud with the assets in [`docker/README.md`](docker/README.md).

## Datasets

Data lives in `Datasets/` (gitignored — pickles exceed GitHub limits). Reference
sets by short name; `load_dataset` resolves `Datasets/<file>` first, then
`~/Downloads`. See `DATASETS` in [`nmrlib/data.py`](nmrlib/data.py) for the
current registry.

## Setup

```bash
uv sync            # install dependencies into .venv
uv run jupyter lab # or open a notebook in the IDE
```
