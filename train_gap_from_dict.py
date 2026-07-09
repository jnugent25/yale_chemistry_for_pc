"""Train an HPO-tuned HistGradientBoosting gap_ev model on NMF codes, where the
NMF dictionary is learned on a large *unlabeled* NMR corpus.

Pipeline:
  1. Fit the tuned NMF dictionary (e.g. the RF-optimal sweep config) on a large
     DICT corpus (ids_nmr_100k), after removing molecules that also appear in the
     labeled set — so the labeled eval stays out-of-sample for the dictionary.
  2. Transform the labeled set (alberts, with gap_ev) into that code space.
  3. HPO-tune a HistGradientBoosting regressor on the codes (train/val), report
     honest held-out test R², and save the model, predictions, and NMF artifact.

Mirrors create_features_nmr.ipynb: fit a shared NMF dictionary, transform new
data into it, and attach reconstruction-error features.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from create_nmr_dictionary_features import _invert_transform, _make_grid
from cross_dataset_gap import canonical_smiles
from tune_representation import assemble_x, build_raw_matrices
from train_gap_model import config_from_params, load_working_set, make_nmf
from nmrlib.models import make_hgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit NMF dictionary on a big NMR corpus, train HPO-HGB gap_ev on a labeled set.")
    p.add_argument("--dict-data", type=Path, default=Path("/Users/jacknugent/Downloads/ids_nmr_100k.pkl"),
                   help="Large unlabeled NMR corpus the NMF dictionary is fit on.")
    p.add_argument("--label-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"),
                   help="Labeled set (with gap_ev) to transform and train on.")
    p.add_argument("--params-json", type=Path, required=True,
                   help="NMF hyperparameters JSON (e.g. the RF-optimal trial).")
    p.add_argument("--sample", type=int, default=12000, help="Cap on labeled molecules.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-trials", type=int, default=40, help="Optuna trials for the HGB.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=Path("/Users/jacknugent/Downloads/gap_from_100k_dict"))
    return p.parse_args()


def load_dict_corpus(path: Path) -> pd.DataFrame:
    """Load an unlabeled NMR corpus; keep rows with valid spectra + SMILES."""
    df = pd.read_pickle(path)
    if "smiles" not in df.columns:
        for alias in ("canonical_smiles", "SMILES"):
            if alias in df.columns:
                df = df.rename(columns={alias: "smiles"})
                break

    def _has_peaks(v: object) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        try:
            return len(v) > 0
        except TypeError:
            return False

    ok = (df["smiles"].notna()
          & df["h_nmr_peaks"].map(_has_peaks)
          & df["c_nmr_peaks"].map(_has_peaks))
    return df[ok].reset_index(drop=True)


def build_x(df: pd.DataFrame, cfg, h_step: float, c_step: float, free_df: bool = False):
    """Assemble the representation matrix; optionally free the (huge) df early."""
    h_grid = _make_grid(0.0, 12.0, h_step)
    c_grid = _make_grid(0.0, 220.0, c_step)
    h, c = build_raw_matrices(df, h_grid, c_grid, cfg)
    if free_df:
        del df
        gc.collect()
    x = assemble_x(h, c, cfg)
    return h, c, x


def tune_hgb(x_tr, g_tr, x_va, g_va, n_trials, seed) -> dict:
    """Optuna HPO maximizing val R² (codes fixed; NMF already applied)."""
    def objective(t: optuna.Trial) -> float:
        params = dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=t.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100, log=True),
            l2_regularization=t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=t.suggest_float("max_features", 0.3, 1.0),
        )
        m = make_hgb(params, seed)
        m.fit(x_tr, g_tr)
        return float(r2_score(g_va, m.predict(x_va)))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for pth in (args.dict_data, args.label_data, args.params_json):
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_from_params(json.loads(args.params_json.read_text()))
    print(f"NMF config: n_components={cfg.n_components}, solver={cfg.solver}, "
          f"transform={cfg.intensity_transform}, multiplicity={cfg.h_multiplicity}")

    # ---- 1. Fit NMF dictionary on the big corpus, minus label overlap ----
    print(f"\nLoading dict corpus {args.dict_data.name} ...")
    df_dict = load_dict_corpus(args.dict_data)
    print(f"  {len(df_dict)} valid molecules")

    print("Loading label set + computing canonical SMILES for overlap removal ...")
    df_lab, gap = load_working_set(args.label_data, args.sample, args.seed)
    lab_canon = set(canonical_smiles(df_lab["smiles"]).dropna())
    dict_canon = canonical_smiles(df_dict["smiles"])
    keep = ~dict_canon.isin(lab_canon)
    n_overlap = int((~keep).sum())
    df_dict = df_dict[keep.values].reset_index(drop=True)
    print(f"  removed {n_overlap} corpus molecules overlapping the label set; "
          f"{len(df_dict)} remain for the dictionary")

    print(f"Building dictionary matrix + fitting NMF (max_iter={args.max_iter}) ...")
    _, _, x_dict = build_x(df_dict, cfg, args.h_step_ppm, args.c_step_ppm, free_df=True)
    del df_dict
    gc.collect()
    print(f"  dictionary matrix {x_dict.shape}")
    nmf = make_nmf(cfg, args.max_iter)
    nmf.fit(x_dict)
    comps = nmf.components_
    del x_dict
    gc.collect()

    # ---- 2. Transform the labeled set into the code space ----
    print(f"\nTransforming label set into the code space ...")
    h_lab, c_lab, x_lab = build_x(df_lab, cfg, args.h_step_ppm, args.c_step_ppm)
    W = nmf.transform(x_lab).astype(np.float64)
    del x_lab
    gc.collect()
    print(f"  {W.shape[0]} labeled molecules -> {W.shape[1]} codes")

    # Reconstruction-error features (mirrors the notebook).
    h_width = h_lab.shape[1]
    h_comps = comps[:, :h_width] / cfg.h_modality_weight
    c_comps = comps[:, h_width:] / cfg.c_modality_weight
    h_err = np.linalg.norm(h_lab - _invert_transform(W @ h_comps, cfg.intensity_transform), axis=1)
    c_err = np.linalg.norm(c_lab - _invert_transform(W @ c_comps, cfg.intensity_transform), axis=1)
    recon = np.column_stack([h_err, c_err, h_err + c_err])
    Wp = np.hstack([W, recon])   # codes + 3 recon-error features
    del h_lab, c_lab
    gc.collect()

    # ---- 3. HPO-tuned HGB on the codes ----
    m_ok = ~np.isnan(gap)
    Wp, gap = Wp[m_ok], gap[m_ok]
    idx = np.arange(len(Wp))
    trv, te = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    tr, va = train_test_split(trv, test_size=args.val_frac / (1 - args.test_frac), random_state=args.seed)
    print(f"\nLabel split: {len(tr)} train / {len(va)} val / {len(te)} test")

    print(f"HPO-tuning HGB ({args.n_trials} trials) ...")
    best = tune_hgb(Wp[tr], gap[tr], Wp[va], gap[va], args.n_trials, args.seed)
    print("Best HGB params:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    model = make_hgb(best, args.seed)
    model.fit(Wp[np.concatenate([tr, va])], gap[np.concatenate([tr, va])])  # train+val for final fit
    pred = model.predict(Wp[te])
    test_r2 = float(r2_score(gap[te], pred))
    test_rmse = float(root_mean_squared_error(gap[te], pred))
    train_r2 = float(r2_score(gap[tr], model.predict(Wp[tr])))
    print(f"\n=== HPO-tuned HGB (100k-dictionary codes) ===")
    print(f"  train R² = {train_r2:.4f}")
    print(f"  test  R² = {test_r2:.4f}   RMSE = {test_rmse:.4f} eV   (overfit gap {train_r2 - test_r2:.4f})")

    # ---- Save artifacts ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gap[te], pred, s=18, alpha=0.4, edgecolor="none")
    lo, hi = min(gap[te].min(), pred.min()), max(gap[te].max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual gap_ev (eV)"); ax.set_ylabel("predicted gap_ev (eV)")
    ax.set_title(f"HPO HGB on 100k-dictionary codes\nR²={test_r2:.3f} RMSE={test_rmse:.3f} eV")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "gap_actual_vs_pred.png", dpi=150); plt.close(fig)

    code_cols = [f"nmr_dict_code_{i:03d}" for i in range(W.shape[1])]
    artifact = {
        "config": asdict(cfg), "nmf_model": nmf,
        "h_grid_ppm": _make_grid(0.0, 12.0, args.h_step_ppm),
        "c_grid_ppm": _make_grid(0.0, 220.0, args.c_step_ppm),
        "h_components": h_comps, "c_components": c_comps,
        "feature_columns": code_cols,
    }
    with open(args.out_dir / "nmf_dictionary_100k.pkl", "wb") as f:
        pickle.dump(artifact, f)
    with open(args.out_dir / "hgb_gap_model.pkl", "wb") as f:
        pickle.dump(model, f)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"test_r2": test_r2, "test_rmse": test_rmse, "train_r2": train_r2,
         "best_hgb_params": best, "n_dict_molecules": int(len(keep) - n_overlap),
         "n_components": int(W.shape[1])}, indent=2))
    print(f"\nWrote model, NMF dictionary, plot, and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
