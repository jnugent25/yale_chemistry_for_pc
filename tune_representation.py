"""Tune the NMF representation for held-out gap_ev R^2 (primary), with
held-out reconstruction, functional-group micro-F1, and logP R^2 as guardrails.

Design choices:
  - NMF is fit on TRAIN molecules only; codes for VAL are obtained via transform.
    This avoids leakage and gives an honest *held-out* reconstruction error.
   
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from create_nmr_dictionary_features import (
    _apply_transform,
    _invert_transform,
    _make_grid,
    build_h_multichannel_matrix,
    build_soft_peak_matrix,
)
from benchmark_functional_groups import label_matrix


@dataclass(frozen=True)
class SweepConfig:
    name: str
    n_components: int
    h_sigma_ppm: float
    c_sigma_ppm: float
    h_width_scale: float = 0.25
    c_width_scale: float = 1.0
    h_multiplicity: bool = False
    h_mult_classes: tuple = ("s", "d", "t", "q", "m")
    use_peak_width: bool = True
    intensity_transform: str = "none"
    h_modality_weight: float = 1.0
    c_modality_weight: float = 1.0
    solver: str = "cd"
    beta_loss: str = "frobenius"
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0




def raster_key(cfg: SweepConfig) -> tuple:
    """Cache key covering only the parameters that affect Gaussian rasterization.

    intensity_transform and modality weights are cheap elementwise ops applied
    per-trial, so they stay out of the key to maximize cache hits.
    """
    return (
        cfg.h_sigma_ppm,
        cfg.c_sigma_ppm,
        cfg.h_width_scale,
        cfg.c_width_scale,
        cfg.h_multiplicity,
        cfg.h_mult_classes if cfg.h_multiplicity else None,
        cfg.use_peak_width,
    )


def build_raw_matrices(
    df: pd.DataFrame,
    h_grid: np.ndarray,
    c_grid: np.ndarray,
    cfg: SweepConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.h_multiplicity:
        h = build_h_multichannel_matrix(
            df["h_nmr_peaks"], h_grid, cfg.h_sigma_ppm, cfg.h_mult_classes,
            width_scale=cfg.h_width_scale, use_peak_width=cfg.use_peak_width,
        )
    else:
        h = build_soft_peak_matrix(
            df["h_nmr_peaks"], h_grid, cfg.h_sigma_ppm, "h",
            width_scale=cfg.h_width_scale, use_peak_width=cfg.use_peak_width,
        )
    c = build_soft_peak_matrix(
        df["c_nmr_peaks"], c_grid, cfg.c_sigma_ppm, "c",
        width_scale=cfg.c_width_scale, use_peak_width=cfg.use_peak_width,
    )
    return h.astype(np.float32, copy=False), c.astype(np.float32, copy=False)


def assemble_x(h: np.ndarray, c: np.ndarray, cfg: SweepConfig) -> np.ndarray:
    h_transformed = _apply_transform(h, cfg.intensity_transform)
    c_transformed = _apply_transform(c, cfg.intensity_transform)
    return np.concatenate([
        cfg.h_modality_weight * h_transformed,
        cfg.c_modality_weight * c_transformed,
    ], axis=1)


def relative_reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean per-row ||x - x_hat|| / ||x|| over rows with positive norm."""
    num = np.linalg.norm(x - x_hat, axis=1)
    den = np.linalg.norm(x, axis=1)
    mask = den > 0
    return float(np.mean(num[mask] / den[mask]))


def fit_predict_per_label_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Per-label elasticnet logistic-regression probe (linear analogue of the RF probe)."""
    predictions = np.zeros((len(x_test), y_train.shape[1]), dtype=np.int8)
    scaler = StandardScaler().fit(x_train)
    x_tr = scaler.transform(x_train)
    x_te = scaler.transform(x_test)
    for label_idx in range(y_train.shape[1]):
        col = y_train[:, label_idx]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        clf = LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-3,
            random_state=random_state,
        )
        # This is a guardrail metric, not the optimization target, so a label
        # that just misses the strict tolerance is fine — quiet the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(x_tr, col)
        predictions[:, label_idx] = clf.predict(x_te)
    return predictions


def elasticnet_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> float:
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=50,
            cv=3,
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    # Downstream target/guardrail regressor on the NMF codes; a near-converged
    # elastic-net fit is fine for comparing R² across trials — quiet the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)
    return float(r2_score(y_test, model.predict(x_test)))


def logp_targets(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen

    vals = np.full(len(smiles), np.nan)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            vals[i] = Crippen.MolLogP(mol)
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune NMF representation (gap_ev-primary, recon+FG+logP guardrails).")
    p.add_argument("--raw-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--sample", type=int, default=12000)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3245)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials to run.")
    p.add_argument("--out", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.raw_data} ...")
    df = pd.read_pickle(args.raw_data)
    # Normalize column aliases (e.g. the alberts qchem-merged dataset).
    if "smiles" not in df.columns:
        for alias in ("canonical_smiles", "SMILES"):
            if alias in df.columns:
                df = df.rename(columns={alias: "smiles"})
                break
    if "gap_ev" not in df.columns and "qchem_gap_ev" in df.columns:
        df = df.rename(columns={"qchem_gap_ev": "gap_ev"})
    keep = [c for c in ["smiles", "molecular_formula", "h_nmr_peaks", "c_nmr_peaks", "gap_ev", "logP"] if c in df.columns]
    df = df[keep]

    # Drop rows missing anything the run needs: both spectra (NMF inputs),
    # gap_ev (target), and smiles (FG/logP guardrails). Peaks are numpy arrays,
    # so check for a non-empty sequence rather than relying on isna.
    def _has_peaks(v: object) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        try:
            return len(v) > 0
        except TypeError:
            return False

    n_before = len(df)
    complete = (
        df["smiles"].notna()
        & df["gap_ev"].notna()
        & df["h_nmr_peaks"].map(_has_peaks)
        & df["c_nmr_peaks"].map(_has_peaks)
    )
    df = df[complete]
    print(f"Dropped {n_before - len(df)} of {n_before} rows missing required data.")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"Working set: {len(df)} molecules")

    # Labels / targets (computed once, aligned to df rows)
    y_fg, valid = label_matrix(df["smiles"])
    # Prefer the dataset's own logP column (stored as strings); fall back to the
    # Crippen estimate from SMILES only if the column is absent.
    if "logP" in df.columns:
        logp = pd.to_numeric(df["logP"], errors="coerce").to_numpy(dtype=float)
    else:
        logp = logp_targets(df["smiles"])
    gap = df["gap_ev"].values if "gap_ev" in df.columns else np.full(len(df), np.nan)

    h_grid = _make_grid(0.0, 12.0, args.h_step_ppm)
    c_grid = _make_grid(0.0, 220.0, args.c_step_ppm)

    # Train/val split (shared across all configs)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    n_val = int(len(df) * args.val_frac)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val\n")

    # Small LRU over raw rasterized matrices (~hundreds of MB each at float32).
    matrix_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
    max_cache_entries = 3
    results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        # Integer (not categorical) so TPE sees the ordering; recon improves with
        # more atoms but downstream regression overfits past a point.
        nc = trial.suggest_int("n_components", 10, 120, step=5)
        # Gaussian broadening. Lower bounds are held at ~1 grid bin (H=0.01 ppm,
        # C=0.25 ppm) so peaks are never narrower than the raster can resolve.
        h_sigma = trial.suggest_float("h_sigma_ppm", 0.01, 0.10, step=0.01)
        c_sigma = trial.suggest_float("c_sigma_ppm", 0.25, 2.0, step=0.25)
        # Per-peak physical-width scaling (only used when use_peak_width=True).
        h_width_scale = trial.suggest_float("h_width_scale", 0.1, 1.0, step=0.1)
        c_width_scale = trial.suggest_float("c_width_scale", 0.5, 2.0, step=0.5)
        h_multiplicity = trial.suggest_categorical("h_multiplicity", [True, False])
        use_peak_width = trial.suggest_categorical("use_peak_width", [True, False])
        intensity_transform = trial.suggest_categorical("intensity_transform", ["none", "sqrt", "cbrt", "log1p", "arcsinh"])
        # H/C balance: C is pinned to 1.0, so this is the H:C ratio. Log-scaled and
        # spanning both sides of 1.0 because H and C blocks have unequal total norm.
        h_modality_weight = trial.suggest_float("h_modality_weight", 0.1, 10.0, log=True)

        solver = trial.suggest_categorical("solver", ["cd", "mu"])
        # KL/IS divergence needs the mu solver; cd is frobenius-only.
        if solver == "mu":
            beta_loss = trial.suggest_categorical("beta_loss", ["frobenius", "kullback-leibler"])
        else:
            beta_loss = "frobenius"
        # Sparsity is searched for BOTH solvers. alpha_W sparsifies the codes
        # (feature selection for the gap_ev regressor), alpha_H sparsifies the
        # dictionary (interpretable spectral motifs). sklearn scales these by
        # matrix size, so the effective penalty is large; the upper bound is
        # capped at 1e-2 to stay below the dictionary-collapse regime, and the
        # 1e-5 lower bound is effectively dense. l1_ratio is the shared L1/L2 mix.
        alpha_W = trial.suggest_float("alpha_W", 1e-5, 1e-2, log=True)
        alpha_H = trial.suggest_float("alpha_H", 1e-5, 1e-2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        cfg = SweepConfig(
            name=f"trial_{trial.number}",
            n_components=nc,
            h_sigma_ppm=h_sigma,
            c_sigma_ppm=c_sigma,
            h_width_scale=h_width_scale,
            c_width_scale=c_width_scale,
            h_multiplicity=h_multiplicity,
            use_peak_width=use_peak_width,
            intensity_transform=intensity_transform,
            h_modality_weight=h_modality_weight,
            c_modality_weight=1.0,
            solver=solver,
            beta_loss=beta_loss,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
        )

        key = raster_key(cfg)
        if key in matrix_cache:
            matrix_cache.move_to_end(key)
        else:
            print(f"[build] fingerprints for raster={key} ...", flush=True)
            matrix_cache[key] = build_raw_matrices(df, h_grid, c_grid, cfg)
            while len(matrix_cache) > max_cache_entries:
                matrix_cache.popitem(last=False)
        h, c = matrix_cache[key]
        x = assemble_x(h, c, cfg)
        x_tr, x_va = x[train_idx], x[val_idx]
        h_va, c_va = h[val_idx], c[val_idx]

        print(f"[fit ] {cfg.name}: NMF n_components={cfg.n_components} (train only) ...", flush=True)
        nmf = NMF(
            n_components=cfg.n_components,
            init="nndsvda",
            solver=cfg.solver,
            beta_loss=cfg.beta_loss,
            alpha_W=cfg.alpha_W,
            alpha_H=cfg.alpha_H,
            l1_ratio=cfg.l1_ratio,
            max_iter=args.max_iter,
            random_state=0,
        )
        try:
            w_tr = nmf.fit_transform(x_tr)
            w_va = nmf.transform(x_va)
        except ValueError as exc:
            # Over-regularization can collapse the dictionary to all zeros;
            # sklearn then raises on transform. Treat as a bad trial, not a crash.
            raise optuna.TrialPruned(f"degenerate NMF fit: {exc}") from exc
        comps = nmf.components_

        # Held-out reconstruction (primary metric, evaluated in original intensity space)
        h_width = h.shape[1]
        h_comps = comps[:, :h_width] / cfg.h_modality_weight
        c_comps = comps[:, h_width:] / cfg.c_modality_weight

        recon_h_va = _invert_transform(w_va @ h_comps, cfg.intensity_transform)
        recon_c_va = _invert_transform(w_va @ c_comps, cfg.intensity_transform)

        recon_va = np.concatenate([recon_h_va, recon_c_va], axis=1)
        original_va = np.concatenate([h_va, c_va], axis=1)

        rel_err = relative_reconstruction_error(original_va, recon_va)

        # Guardrail 1: functional-group micro-F1 on val
        v_tr = valid[train_idx]
        v_va = valid[val_idx]
        pred = fit_predict_per_label_logreg(
            w_tr[v_tr], y_fg[train_idx][v_tr], w_va[v_va], random_state=0,
        )
        micro = float(f1_score(y_fg[val_idx][v_va], pred, average="micro", zero_division=0))
        macro = float(f1_score(y_fg[val_idx][v_va], pred, average="macro", zero_division=0))

        # Target 2: logP R^2 on val (dataset logP)
        lp_tr_mask = ~np.isnan(logp[train_idx])
        lp_va_mask = ~np.isnan(logp[val_idx])
        logp_r2 = elasticnet_r2(
            w_tr[lp_tr_mask], logp[train_idx][lp_tr_mask],
            w_va[lp_va_mask], logp[val_idx][lp_va_mask], random_state=0,
        )

        # Primary target: gap_ev R^2 on val
        gap_tr_mask = ~np.isnan(gap[train_idx])
        gap_va_mask = ~np.isnan(gap[val_idx])
        if gap_tr_mask.any() and gap_va_mask.any():
            gap_r2 = elasticnet_r2(
                w_tr[gap_tr_mask], gap[train_idx][gap_tr_mask],
                w_va[gap_va_mask], gap[val_idx][gap_va_mask], random_state=0,
            )
        else:
            gap_r2 = float("nan")

        row = {
            "name": cfg.name,
            "n_components": cfg.n_components,
            "h_sigma": cfg.h_sigma_ppm,
            "c_sigma": cfg.c_sigma_ppm,
            "val_rel_recon_err": rel_err,
            "fg_micro_f1": micro,
            "fg_macro_f1": macro,
            "logp_r2": logp_r2,
            "gap_ev_r2": gap_r2,
        }
        results.append(row)
        # Persist after every trial so a crash/interrupt never loses progress.
        args.out.write_text(json.dumps(results, indent=2))
        print(
            f"  -> recon_err={rel_err:.4f} | FG micro={micro:.4f} macro={macro:.4f}"
            f" | logP R²={logp_r2:.4f} | gap_ev R²={gap_r2:.4f}\n",
            flush=True,
        )

        trial.set_user_attr("val_rel_recon_err", rel_err)
        trial.set_user_attr("fg_micro_f1", micro)
        trial.set_user_attr("fg_macro_f1", macro)

        # Both objectives must be finite to place the trial on the Pareto front.
        if np.isnan(gap_r2) or np.isnan(logp_r2):
            raise optuna.TrialPruned("gap_ev or logP target unavailable for this split")
        return gap_r2, logp_r2

    # Multi-objective: jointly maximize held-out gap_ev R² and logP R².
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials)

    args.out.write_text(json.dumps(results, indent=2))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)

    print("\n=== Optuna Representation Tuning (multi-objective) ===")
    if not completed:
        print(f"No trials completed ({n_pruned} pruned) — try weaker regularization "
              f"(lower alpha_H/alpha_W upper bound) or a larger --sample.")
        print(f"\nWrote: {args.out}")
        return
    print(f"Completed {len(completed)} trials ({n_pruned} pruned).")

    # No single "best" in multi-objective — report the Pareto front (the set of
    # trials not dominated on both gap_ev R² and logP R²).
    pareto = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    print(f"\nPareto front ({len(pareto)} non-dominated trials), by gap_ev R²:")
    print(f"{'trial':<10}{'gap R²↑':>10}{'logP R²↑':>10}")
    print("-" * 30)
    for t in pareto:
        print(f"{'trial_'+str(t.number):<10}{t.values[0]:>10.4f}{t.values[1]:>10.4f}")
    print("\nPareto-front params:")
    for t in pareto:
        print(f"  trial_{t.number} (gap R²={t.values[0]:.4f}, logP R²={t.values[1]:.4f}):")
        for k, v in t.params.items():
            print(f"      {k}: {v}")

    print("\nAll Trials (in order of completion):")
    print(f"{'trial':<10}{'nc':>5}{'hσ':>7}{'cσ':>7}{'recon↓':>10}{'FG micro↑':>11}{'FG macro↑':>11}{'logP R²↑':>10}{'gap R²↑':>10}")
    print("-" * 87)
    for r in results:
        print(
            f"{r['name']:<10}{r['n_components']:>5}{r['h_sigma']:>7.2f}{r['c_sigma']:>7.2f}"
            f"{r['val_rel_recon_err']:>10.4f}{r['fg_micro_f1']:>11.4f}{r['fg_macro_f1']:>11.4f}{r['logp_r2']:>10.4f}{r['gap_ev_r2']:>10.4f}"
        )
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
