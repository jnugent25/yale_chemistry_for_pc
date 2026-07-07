"""Train a HistGradientBoosting regressor for gap_ev on the NMF representation
chosen by the tuning sweep.

By default it picks the trial with the best held-out gap_ev R² from the Optuna
study, rebuilds that trial's NMF representation, fits NMF on the training split
ONLY (so the held-out codes are honest), and trains a HistGradientBoostingRegressor
on the codes to predict gap_ev. Reports held-out R²/RMSE/MAE and writes an
actual-vs-predicted plot. Optionally runs k-fold CV that refits NMF each fold.

The sweep tuned the representation with a linear ElasticNet probe; this swaps in
a gradient-boosted model for the final fit, which typically lifts R² since it can
exploit nonlinear structure in the codes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display (works over SSH)
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from create_nmr_dictionary_features import _make_grid
from tune_representation import SweepConfig, assemble_x, build_raw_matrices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HistGradientBoosting for gap_ev on the tuned NMF representation.")
    p.add_argument("--raw-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--storage", type=Path,
                   default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.db"),
                   help="SQLite study from tune_representation.py, to pull the winning config.")
    p.add_argument("--study-name", type=str, default="nmf_repr")
    p.add_argument("--trial", type=int, default=None,
                   help="Trial number to use; default = best held-out gap_ev R².")
    p.add_argument("--sample", type=int, default=12000)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--cv", type=int, default=0,
                   help="If >1, run k-fold CV (refitting NMF per fold) instead of a single split.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=Path("/Users/jacknugent/Downloads/gap_model"))
    return p.parse_args()


def config_from_params(params: dict) -> SweepConfig:
    """Rebuild the SweepConfig for a trial's hyperparameters (C weight pinned to 1)."""
    return SweepConfig(
        name="selected",
        n_components=params["n_components"],
        h_sigma_ppm=params["h_sigma_ppm"],
        c_sigma_ppm=params["c_sigma_ppm"],
        h_width_scale=params["h_width_scale"],
        c_width_scale=params["c_width_scale"],
        h_multiplicity=params["h_multiplicity"],
        use_peak_width=params["use_peak_width"],
        intensity_transform=params["intensity_transform"],
        h_modality_weight=params["h_modality_weight"],
        c_modality_weight=1.0,
        solver=params["solver"],
        beta_loss=params.get("beta_loss", "frobenius"),
        alpha_W=params["alpha_W"],
        alpha_H=params["alpha_H"],
        l1_ratio=params["l1_ratio"],
    )


def select_trial(storage: Path, study_name: str, trial_number: int | None) -> optuna.trial.FrozenTrial:
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage}")
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        raise SystemExit("No completed trials in the study.")
    if trial_number is not None:
        match = [t for t in completed if t.number == trial_number]
        if not match:
            raise SystemExit(f"Trial {trial_number} not found among completed trials.")
        return match[0]
    # Objective 0 is gap_ev R² (see tune_representation.py's objective return).
    return max(completed, key=lambda t: t.values[0])


def load_working_set(raw_data: Path, sample: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the dataset, normalize aliases, drop rows missing spectra/gap, sample.

    Mirrors the loading/filtering in tune_representation.py so the representation
    is built on the same rows the sweep saw.
    """
    df = pd.read_pickle(raw_data)
    if "smiles" not in df.columns:
        for alias in ("canonical_smiles", "SMILES"):
            if alias in df.columns:
                df = df.rename(columns={alias: "smiles"})
                break
    if "gap_ev" not in df.columns and "qchem_gap_ev" in df.columns:
        df = df.rename(columns={"qchem_gap_ev": "gap_ev"})

    def _has_peaks(v: object) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        try:
            return len(v) > 0
        except TypeError:
            return False

    complete = (
        df["gap_ev"].notna()
        & df["h_nmr_peaks"].map(_has_peaks)
        & df["c_nmr_peaks"].map(_has_peaks)
    )
    df = df[complete]
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    df = df.reset_index(drop=True)
    gap = pd.to_numeric(df["gap_ev"], errors="coerce").to_numpy(dtype=float)
    return df, gap


def build_representation(df: pd.DataFrame, cfg: SweepConfig, h_step: float, c_step: float) -> np.ndarray:
    h_grid = _make_grid(0.0, 12.0, h_step)
    c_grid = _make_grid(0.0, 220.0, c_step)
    h, c = build_raw_matrices(df, h_grid, c_grid, cfg)
    return assemble_x(h, c, cfg)


def make_nmf(cfg: SweepConfig, max_iter: int) -> NMF:
    return NMF(
        n_components=cfg.n_components,
        init="nndsvda",
        solver=cfg.solver,
        beta_loss=cfg.beta_loss,
        alpha_W=cfg.alpha_W,
        alpha_H=cfg.alpha_H,
        l1_ratio=cfg.l1_ratio,
        max_iter=max_iter,
        random_state=0,
    )


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def run_holdout(x: np.ndarray, gap: np.ndarray, cfg: SweepConfig, args) -> tuple[dict, np.ndarray, np.ndarray]:
    idx = np.arange(len(x))
    tr, te = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    nmf = make_nmf(cfg, args.max_iter)
    w_tr = nmf.fit_transform(x[tr])   # NMF fit on train only -> no leakage
    w_te = nmf.transform(x[te])
    model = make_model(args.seed)
    model.fit(w_tr, gap[tr])
    pred = model.predict(w_te)
    return _metrics(gap[te], pred), gap[te], pred


def run_cv(x: np.ndarray, gap: np.ndarray, cfg: SweepConfig, args) -> tuple[dict, np.ndarray, np.ndarray]:
    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    y_true_all, y_pred_all, fold_r2 = [], [], []
    for k, (tr, te) in enumerate(kf.split(x), 1):
        nmf = make_nmf(cfg, args.max_iter)
        w_tr = nmf.fit_transform(x[tr])   # refit NMF each fold -> honest CV
        w_te = nmf.transform(x[te])
        model = make_model(args.seed)
        model.fit(w_tr, gap[tr])
        pred = model.predict(w_te)
        fold_r2.append(r2_score(gap[te], pred))
        y_true_all.append(gap[te])
        y_pred_all.append(pred)
        print(f"  fold {k}/{args.cv}: R²={fold_r2[-1]:.4f}")
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    m = _metrics(y_true, y_pred)
    m["r2_fold_mean"] = float(np.mean(fold_r2))
    m["r2_fold_std"] = float(np.std(fold_r2))
    return m, y_true, y_pred


def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.4, edgecolor="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xlabel("actual gap_ev (eV)")
    ax.set_ylabel("predicted gap_ev (eV)")
    ax.set_title(f"HistGradientBoosting on NMF codes\nR²={metrics['r2']:.3f}  "
                 f"RMSE={metrics['rmse']:.3f} eV  MAE={metrics['mae']:.3f} eV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "gap_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.storage.exists():
        raise SystemExit(f"Storage not found: {args.storage}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trial = select_trial(args.storage, args.study_name, args.trial)
    cfg = config_from_params(trial.params)
    print(f"Using trial {trial.number}: sweep gap_ev R²={trial.values[0]:.4f}, logP R²={trial.values[1]:.4f}")
    print("Representation config:")
    for k, v in sorted(trial.params.items()):
        print(f"    {k}: {v}")

    print(f"\nLoading {args.raw_data} ...")
    df, gap = load_working_set(args.raw_data, args.sample, args.seed)
    print(f"Working set: {len(df)} molecules")
    x = build_representation(df, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"Representation matrix: {x.shape}")

    if args.cv and args.cv > 1:
        print(f"\n{args.cv}-fold CV (NMF refit per fold):")
        metrics, y_true, y_pred = run_cv(x, gap, cfg, args)
    else:
        print(f"\nSingle held-out split (test_frac={args.test_frac}):")
        metrics, y_true, y_pred = run_holdout(x, gap, cfg, args)

    print("\n=== HistGradientBoosting gap_ev performance ===")
    print(f"  R²   = {metrics['r2']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f} eV")
    print(f"  MAE  = {metrics['mae']:.4f} eV")
    if "r2_fold_mean" in metrics:
        print(f"  per-fold R² = {metrics['r2_fold_mean']:.4f} ± {metrics['r2_fold_std']:.4f}")

    plot_actual_vs_pred(y_true, y_pred, metrics, args.out_dir)
    pd.DataFrame({"actual_gap_ev": y_true, "predicted_gap_ev": y_pred}).to_csv(
        args.out_dir / "gap_predictions.csv", index=False
    )
    print(f"\nWrote plot + predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
