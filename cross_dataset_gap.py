"""Cross-dataset gap_ev transfer test.

Fit the tuned NMF representation on a TRAIN dataset (e.g. alberts), transform a
separate TEST dataset (e.g. ids_nmr) into the same code space, remove molecules
that appear in both (by canonical SMILES) so the test is truly held out, then
train a regressor on the train codes and evaluate on the test codes. This is the
strongest generalization check: the model must predict gap_ev for a different
dataset it has never seen, using a representation fit only on the train set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display (works over SSH)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from train_gap_model import (
    build_representation,
    config_from_params,
    load_working_set,
    make_nmf,
    select_trial,
)
from nmrlib.models import make_hgb_preset, make_linear, make_rf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset gap_ev transfer: fit NMF on train, test on a different dataset.")
    p.add_argument("--train-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--test-data", type=Path, default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_all_features.pkl"))
    p.add_argument("--storage", type=Path,
                   default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.db"))
    p.add_argument("--study-name", type=str, default="nmf_repr")
    p.add_argument("--trial", type=int, default=None, help="Representation trial; default = best sweep gap_ev R².")
    p.add_argument("--params-json", type=Path, default=None,
                   help="JSON file of NMF hyperparameters to use directly, instead of pulling "
                        "a trial from --storage (lets you run without the study db).")
    p.add_argument("--sample", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=Path("/Users/jacknugent/Downloads/cross_dataset_gap"))
    return p.parse_args()


def canonical_smiles(series: pd.Series) -> pd.Series:
    from rdkit import Chem

    def canon(s: object) -> str | None:
        try:
            m = Chem.MolFromSmiles(str(s))
            return Chem.MolToSmiles(m) if m is not None else None
        except Exception:
            return None

    return series.map(canon)


def evaluate_transfer(model, w_tr, g_tr, w_te, g_te) -> dict:
    model.fit(w_tr, g_tr)
    pred = model.predict(w_te)
    return {
        "test_r2": float(r2_score(g_te, pred)),
        "test_rmse": float(root_mean_squared_error(g_te, pred)),
    }


def report(label: str, m: dict) -> None:
    print(f"  {label:<34} test R² = {m['test_r2']:.4f}   RMSE = {m['test_rmse']:.4f} eV")


def main() -> None:
    args = parse_args()
    required = [args.train_data, args.test_data]
    if args.params_json is None:
        required.append(args.storage)
    for pth in required:
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.params_json is not None:
        params = json.loads(args.params_json.read_text())
        cfg = config_from_params(params)
        print(f"Representation from {args.params_json.name}: n_components={cfg.n_components}, "
              f"solver={cfg.solver}, transform={cfg.intensity_transform}")
    else:
        trial = select_trial(args.storage, args.study_name, args.trial)
        cfg = config_from_params(trial.params)
        print(f"Representation: trial {trial.number} (sweep gap_ev R²={trial.values[0]:.4f}), "
              f"n_components={cfg.n_components}")

    # Train dataset.
    print(f"\nTrain: {args.train_data}")
    df_tr, gap_tr = load_working_set(args.train_data, args.sample, args.seed)
    print(f"  {len(df_tr)} molecules")

    # Test dataset.
    print(f"Test:  {args.test_data}")
    df_te, gap_te = load_working_set(args.test_data, args.sample, args.seed)
    print(f"  {len(df_te)} molecules (before overlap removal)")

    # Remove molecules shared with the train set (by canonical SMILES).
    tr_canon = set(canonical_smiles(df_tr["smiles"]).dropna())
    te_canon = canonical_smiles(df_te["smiles"])
    keep = ~te_canon.isin(tr_canon)
    n_overlap = int((~keep).sum())
    df_te = df_te[keep.values].reset_index(drop=True)
    gap_te = gap_te[keep.values]
    print(f"  removed {n_overlap} molecules overlapping the train set; {len(df_te)} test molecules remain")

    # Build the representation on both with identical grids/config, fit NMF on
    # train only, transform test into the same code space.
    x_tr = build_representation(df_tr, cfg, args.h_step_ppm, args.c_step_ppm)
    x_te = build_representation(df_te, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"\nRepresentation: train {x_tr.shape}, test {x_te.shape}")
    print(f"Fitting NMF on train (max_iter={args.max_iter}) ...")
    nmf = make_nmf(cfg, args.max_iter)
    # float64 codes: ElasticNetCV's Gram precompute fails its precision check on
    # float32 at this scale (the codes are small, so this is cheap).
    w_tr = nmf.fit_transform(x_tr).astype(np.float64)
    w_te = nmf.transform(x_te).astype(np.float64)

    # Drop any NaN gap targets defensively.
    m_tr = ~np.isnan(gap_tr)
    m_te = ~np.isnan(gap_te)
    w_tr, g_tr = w_tr[m_tr], gap_tr[m_tr]
    w_te, g_te = w_te[m_te], gap_te[m_te]

    print(f"\n=== Cross-dataset gap_ev transfer (train→test, no overlap) ===")
    linear = evaluate_transfer(make_linear(args.seed), w_tr, g_tr, w_te, g_te)
    rf = evaluate_transfer(make_rf(args.seed), w_tr, g_tr, w_te, g_te)
    hgb = evaluate_transfer(make_hgb_preset(args.seed), w_tr, g_tr, w_te, g_te)
    report("ElasticNet (linear)", linear)
    report("Random forest (regularized)", rf)
    report("HGB preset (depth 4, 200 trees)", hgb)

    # Actual-vs-predicted for the RF (usually the strongest here).
    rf_model = make_rf(args.seed).fit(w_tr, g_tr)
    pred = rf_model.predict(w_te)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g_te, pred, s=18, alpha=0.4, edgecolor="none")
    lo, hi = min(g_te.min(), pred.min()), max(g_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xlabel("actual gap_ev (eV)")
    ax.set_ylabel("predicted gap_ev (eV)")
    ax.set_title(f"Cross-dataset RF transfer\nR²={rf['test_r2']:.3f}  RMSE={rf['test_rmse']:.3f} eV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(args.out_dir / "cross_dataset_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"actual_gap_ev": g_te, "predicted_gap_ev": pred}).to_csv(
        args.out_dir / "cross_dataset_predictions.csv", index=False
    )
    print(f"\nWrote plot + predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
