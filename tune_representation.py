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


def rf_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> float:
    """Random-forest probe: tunes the representation for the nonlinear model we
    actually use downstream, instead of a linear decoder. Settings match the
    final RF (train_gap_model.make_rf)."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=500, max_features=0.33, min_samples_leaf=10,
        n_jobs=-1, random_state=random_state,
    )
    model.fit(x_train, y_train)
    return float(r2_score(y_test, model.predict(x_test)))


def probe_r2(probe: str, *args) -> float:
    """Dispatch the regression probe used for the gap/logP objectives."""
    if probe == "elasticnet":
        return elasticnet_r2(*args)
    if probe == "rf":
        return rf_r2(*args)
    raise ValueError(f"Unknown probe: {probe}")


def logp_targets(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen

    vals = np.full(len(smiles), np.nan)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            vals[i] = Crippen.MolLogP(mol)
    return vals


def _trial_row(t: optuna.trial.FrozenTrial) -> dict:
    """Flatten one completed trial into the JSON row schema, pulling from the
    persisted study (params + user_attrs + objective values) rather than any
    in-memory state, so the dump is correct even after a resume."""
    return {
        "name": f"trial_{t.number}",
        "n_components": t.params.get("n_components"),
        "h_sigma": t.params.get("h_sigma_ppm"),
        "c_sigma": t.params.get("c_sigma_ppm"),
        "val_rel_recon_err": t.user_attrs.get("val_rel_recon_err"),
        "fg_micro_f1": t.user_attrs.get("fg_micro_f1"),
        "fg_macro_f1": t.user_attrs.get("fg_macro_f1"),
        "logp_r2": t.values[1] if t.values is not None else None,
        "gap_ev_r2": t.values[0] if t.values is not None else None,
    }


def dump_results(study: optuna.Study, path: Path) -> None:
    """Write every COMPLETE trial in the study to the JSON output path."""
    rows = [
        _trial_row(t)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    path.write_text(json.dumps(rows, indent=2))


# Representation losses that need the torch engine's geomloss OT/MMD path (and a
# SpectralGeometry). Everything else (frobenius/kl) is pointwise.
OT_FAMILIES = {"sinkhorn", "sinkhorn_unbalanced", "energy", "gaussian", "laplacian"}


def _suggest_repr_loss(trial: optuna.Trial, args: argparse.Namespace) -> str:
    """Pick the representation loss for this trial. sklearn is always frobenius/CD;
    torch searches over --losses (a categorical only when more than one is given)."""
    if args.engine == "sklearn":
        return "frobenius"
    if len(args.losses) == 1:
        return args.losses[0]
    return trial.suggest_categorical("repr_loss", args.losses)


def build_torch_engine(trial, repr_loss, cfg, h_grid, c_grid, args):
    """Construct a TorchNMF (Adam, GPU) for `repr_loss`, suggesting the loss-specific
    knobs. Imported lazily so sklearn runs never pull in torch."""
    from nmrlib.torch_nmf import TorchNMF, TorchNMFConfig, SpectralGeometry

    # Optimizer / iteration budget: fixed from CLI unless --search-optim, in which
    # case they join the search. lr range is conditioned on the optimizer (LBFGS
    # wants a much larger step than Adam).
    if args.search_optim:
        optimizer = trial.suggest_categorical("optimizer", ["adam", "lbfgs"])
        lr = (trial.suggest_float("lr", 0.1, 1.0, log=True) if optimizer == "lbfgs"
              else trial.suggest_float("lr", 1e-3, 0.2, log=True))
        n_iter = trial.suggest_int("n_iter", 100, 600, step=50)
    else:
        optimizer, lr, n_iter = "adam", args.torch_lr, args.torch_n_iter

    common = dict(
        n_components=cfg.n_components, alpha_W=cfg.alpha_W, alpha_H=cfg.alpha_H,
        l1_ratio=cfg.l1_ratio, optimizer=optimizer, lr=lr,
        n_iter=n_iter, transform_n_iter=n_iter,
        device=args.torch_device, dtype="float32",
    )
    if repr_loss in ("frobenius", "kl"):
        return TorchNMF(TorchNMFConfig(loss=repr_loss, **common), geometry=None)

    # geomloss OT / MMD family. h_multiplicity is off for these (see objective), so
    # the H block is single-channel and its width == len(h_grid).
    ot_loss = "sinkhorn" if repr_loss.startswith("sinkhorn") else repr_loss
    reach = trial.suggest_float("reach", 0.05, 0.5, log=True) if repr_loss == "sinkhorn_unbalanced" else None
    p = trial.suggest_categorical("sinkhorn_p", [1, 2]) if ot_loss == "sinkhorn" else 2
    # Sinkhorn annealing ratio (accuracy/speed); only sinkhorn uses it.
    scaling = trial.suggest_float("sinkhorn_scaling", 0.5, 0.9) if ot_loss == "sinkhorn" else 0.7
    h_blur = c_blur = None
    if ot_loss in ("sinkhorn", "gaussian", "laplacian"):
        # Normalized-span units: H peaks are much narrower than C, so search them apart.
        h_blur = trial.suggest_float("h_blur", 0.005, 0.1, log=True)
        c_blur = trial.suggest_float("c_blur", 0.02, 0.3, log=True)
    # Shape-only (unit mass) vs keep-magnitude — only searchable where both are valid:
    # unbalanced OT or an MMD loss. Balanced sinkhorn stays auto (=normalized), since
    # balanced OT on unequal-mass rows is ill-posed.
    mass_normalize = None
    if repr_loss == "sinkhorn_unbalanced" or ot_loss in ("energy", "gaussian", "laplacian"):
        mass_normalize = trial.suggest_categorical("mass_normalize", [True, False])
    geom = SpectralGeometry(
        h_coords=h_grid, c_coords=c_grid,
        h_modality_weight=cfg.h_modality_weight, c_modality_weight=cfg.c_modality_weight,
    )
    tcfg = TorchNMFConfig(loss="geomloss", ot_loss=ot_loss, reach=reach, sinkhorn_p=p,
                          sinkhorn_scaling=scaling, mass_normalize=mass_normalize,
                          h_blur=h_blur, c_blur=c_blur, **common)
    return TorchNMF(tcfg, geometry=geom)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune NMF representation (gap_ev-primary, recon+FG+logP guardrails).")
    p.add_argument("--raw-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--sample", type=int, default=12000)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3245)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--n-trials", type=int, default=50,
                   help="Target TOTAL number of trials in the study; on resume, only the "
                        "remaining trials are run.")
    p.add_argument("--probe", choices=["elasticnet", "rf"], default="elasticnet",
                   help="Regression probe for the gap/logP objectives. 'rf' tunes the "
                        "representation for the nonlinear model used downstream.")
    p.add_argument("--engine", choices=["sklearn", "torch"], default="sklearn",
                   help="NMF engine. sklearn = coordinate descent on CPU (frobenius, no mu). "
                        "torch = Adam gradient descent on GPU; enables the geomloss OT losses "
                        "and is much faster on large data.")
    p.add_argument("--losses", nargs="+", default=["frobenius"],
                   choices=["frobenius", "kl", "sinkhorn", "sinkhorn_unbalanced", "energy", "gaussian", "laplacian"],
                   help="(torch engine) representation loss(es) to search. Give several to let "
                        "the sweep tune which loss maximizes downstream R². Ignored for sklearn.")
    p.add_argument("--torch-device", default=None, help="torch device (default: auto cuda>mps>cpu).")
    p.add_argument("--torch-n-iter", type=int, default=300, help="Adam iterations per torch NMF fit (when not searched).")
    p.add_argument("--torch-lr", type=float, default=0.05, help="Adam learning rate (torch engine, when not searched).")
    p.add_argument("--search-optim", action="store_true",
                   help="(torch engine) also tune the optimizer (adam/lbfgs), learning rate, "
                        "and iteration count per trial instead of fixing them from --torch-lr/--torch-n-iter.")
    # Defaults for these are derived from --probe (below) so elasticnet and rf runs
    # never share a study file / name — keeps the two searches fully separate.
    p.add_argument("--out", type=Path, default=None,
                   help="Results JSON. Default: alberts_gap_repr_sweep_<probe>.json")
    p.add_argument("--storage", type=Path, default=None,
                   help="SQLite file backing the Optuna study for resumability. "
                        "Defaults to <out>.db next to --out.")
    p.add_argument("--study-name", type=str, default=None,
                   help="Optuna study name; reused to resume. Default: nmf_repr_<probe>.")
    args = p.parse_args()
    # Fold engine into the default out/study name so sklearn and torch studies never
    # collide (sklearn keeps the original names for continuity with existing .db files).
    tag = args.probe if args.engine == "sklearn" else f"{args.probe}_torch"
    if args.out is None:
        args.out = Path(f"/Users/jacknugent/Downloads/alberts_gap_repr_sweep_{tag}.json")
    if args.study_name is None:
        args.study_name = f"nmf_repr_{tag}"
    return args


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

    def objective(trial: optuna.Trial) -> float:
        # Representation loss / engine. sklearn -> CD frobenius (no mu); torch -> Adam
        # on GPU for any loss, including the geomloss OT families.
        repr_loss = _suggest_repr_loss(trial, args)
        is_ot = repr_loss in OT_FAMILIES
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
        # OT losses treat each block as a distribution over its ppm grid; the
        # multichannel (per-multiplicity) H layout has no single ppm axis, so it's
        # disabled for OT. Pointwise losses (frobenius/kl) keep the choice.
        h_multiplicity = False if is_ot else trial.suggest_categorical("h_multiplicity", [True, False])
        use_peak_width = trial.suggest_categorical("use_peak_width", [True, False])
        intensity_transform = trial.suggest_categorical("intensity_transform", ["none", "sqrt", "cbrt", "log1p", "arcsinh"])
        # H/C balance: C is pinned to 1.0, so this is the H:C ratio. Log-scaled and
        # spanning both sides of 1.0 because H and C blocks have unequal total norm.
        h_modality_weight = trial.suggest_float("h_modality_weight", 0.1, 10.0, log=True)

        # Solver is pinned to sklearn's coordinate descent (mu handles L1 sparsity
        # poorly). The torch engine ignores these fields (it uses Adam); they only
        # feed the sklearn path, where cd is frobenius-only.
        solver = "cd"
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

        print(f"[fit ] {cfg.name}: {args.engine}/{repr_loss} n_components={cfg.n_components} (train only) ...", flush=True)
        try:
            if args.engine == "sklearn":
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
                w_tr = nmf.fit_transform(x_tr)
                w_va = nmf.transform(x_va)
                comps = nmf.components_
            else:
                engine = build_torch_engine(trial, repr_loss, cfg, h_grid, c_grid, args)
                w_tr = engine.fit_transform(x_tr)
                w_va = engine.transform(x_va)
                comps = engine.components_
        except (ValueError, FloatingPointError) as exc:
            # Over-regularization can collapse the dictionary (sklearn raises on
            # transform); OT under-conditioning can go non-finite. Bad trial, not a crash.
            raise optuna.TrialPruned(f"degenerate NMF fit: {exc}") from exc

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
        logp_r2 = probe_r2(
            args.probe,
            w_tr[lp_tr_mask], logp[train_idx][lp_tr_mask],
            w_va[lp_va_mask], logp[val_idx][lp_va_mask], 0,
        )

        # Primary target: gap_ev R^2 on val
        gap_tr_mask = ~np.isnan(gap[train_idx])
        gap_va_mask = ~np.isnan(gap[val_idx])
        if gap_tr_mask.any() and gap_va_mask.any():
            gap_r2 = probe_r2(
                args.probe,
                w_tr[gap_tr_mask], gap[train_idx][gap_tr_mask],
                w_va[gap_va_mask], gap[val_idx][gap_va_mask], 0,
            )
        else:
            gap_r2 = float("nan")

        print(
            f"  -> recon_err={rel_err:.4f} | FG micro={micro:.4f} macro={macro:.4f}"
            f" | logP R²={logp_r2:.4f} | gap_ev R²={gap_r2:.4f}\n",
            flush=True,
        )

        # Persisted on the trial (in SQLite) so the JSON dump survives a resume.
        trial.set_user_attr("val_rel_recon_err", rel_err)
        trial.set_user_attr("fg_micro_f1", micro)
        trial.set_user_attr("fg_macro_f1", macro)
        trial.set_user_attr("repr_loss", repr_loss)

        # Both objectives must be finite to place the trial on the Pareto front.
        if np.isnan(gap_r2) or np.isnan(logp_r2):
            raise optuna.TrialPruned("gap_ev or logP target unavailable for this split")
        return gap_r2, logp_r2

    # SQLite-backed, resumable study. Re-running the same command reloads the
    # study and only runs the trials still missing to reach the --n-trials target,
    # so an interruption (WSL shutdown, reboot, crash) never loses progress.
    storage_path = args.storage or args.out.with_suffix(".db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    # Multi-objective: jointly maximize held-out gap_ev R² and logP R².
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    already = len(study.trials)
    remaining = max(0, args.n_trials - already)
    print(f"Engine: {args.engine}  |  Losses: {args.losses if args.engine == 'torch' else ['frobenius']}")
    print(f"Probe: {args.probe}  |  Study '{args.study_name}' @ {storage_path}")
    print(f"  {already} existing trials; running {remaining} more to reach {args.n_trials}.\n")

    # Rewrite the JSON after every finished trial (from the persisted study).
    def _save_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        dump_results(study, args.out)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[_save_callback])

    dump_results(study, args.out)

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

    print("\nAll Trials (by trial number):")
    print(f"{'trial':<10}{'nc':>5}{'hσ':>7}{'cσ':>7}{'recon↓':>10}{'FG micro↑':>11}{'FG macro↑':>11}{'logP R²↑':>10}{'gap R²↑':>10}")
    print("-" * 87)
    for r in sorted((_trial_row(t) for t in completed), key=lambda r: int(r["name"].split("_")[1])):
        print(
            f"{r['name']:<10}{r['n_components']:>5}{r['h_sigma']:>7.2f}{r['c_sigma']:>7.2f}"
            f"{r['val_rel_recon_err']:>10.4f}{r['fg_micro_f1']:>11.4f}{r['fg_macro_f1']:>11.4f}{r['logp_r2']:>10.4f}{r['gap_ev_r2']:>10.4f}"
        )
    print(f"\nWrote: {args.out}")
    print(f"Resume/extend with the same command (raise --n-trials to add more).")


if __name__ == "__main__":
    main()
