"""Fit the PyTorch/OT NMF engine on a dataset with a fixed settings JSON and
write the codes out. The torch counterpart of build_dict_codes.py.

build_dict_codes.py uses sklearn (frobenius / KL only). This uses
nmrlib.torch_nmf.TorchNMF, so it can run the optimal-transport losses the sweep
tunes — sinkhorn, sinkhorn_unbalanced, energy, gaussian, laplacian — which
sklearn cannot express. Takes a params JSON (e.g. a winning sweep trial's
hyperparameters, including repr_loss + the OT knobs) rather than individual
flags, so the representation matches exactly what was tuned.

Intended for CUDA: the geomloss `online` (keops) backend keeps OT memory at
O(N) instead of the O(N^2) dense cost matrix, which is the only way large grids
fit. On CPU/MPS it falls back to `tensorized` and will be slow / memory-heavy.

Fit-and-transform the same dataset (default), or fit on --dataset and apply the
learned dictionary to --apply-dataset (the pretrained-dictionary pattern).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from create_nmr_dictionary_features import _invert_transform, _make_grid
from nmrlib.data import REPO_ROOT, load_dataset, resolve_dataset
from train_gap_model import config_from_params
from tune_representation import SweepConfig, assemble_x, build_raw_matrices

# OT / MMD losses that need the geomloss engine + a SpectralGeometry.
OT_FAMILIES = {"sinkhorn", "sinkhorn_unbalanced", "energy", "gaussian", "laplacian"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build torch/OT NMF codes for a dataset from a settings JSON.")
    p.add_argument("--dataset", type=str, required=True,
                   help="Registry short name (see nmrlib.describe_datasets()) or a path. "
                        "The NMF dictionary is fit on this set.")
    p.add_argument("--apply-dataset", type=str, default=None,
                   help="If given, fit on --dataset and transform this one instead "
                        "(kept row-aligned). Default: transform --dataset itself.")
    p.add_argument("--params-json", type=Path, required=True,
                   help="JSON of the tuned hyperparameters (repr_loss + NMF/OT knobs), "
                        "e.g. a winning sweep trial exported to JSON.")
    p.add_argument("--sample", type=int, default=0, help="Cap on fit-set rows (0 = no cap).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)

    # Torch engine runtime (not part of the representation config).
    p.add_argument("--device", default=None, help="torch device (default: auto cuda>mps>cpu).")
    p.add_argument("--n-iter", type=int, default=400, help="Adam iterations for the fit.")
    p.add_argument("--transform-n-iter", type=int, default=200,
                   help="Adam iterations for transform (W only, H frozen).")
    p.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--sinkhorn-backend", choices=["auto", "tensorized", "online", "multiscale"],
                   default="auto",
                   help="geomloss backend. 'auto' -> 'online' (keops) on CUDA, needs a CUDA "
                        "toolkit (nvcc). Pass 'tensorized' to force pure torch (no compiler, "
                        "much more memory — small grids only).")
    p.add_argument("--verbose", action="store_true", help="Print per-step loss during the fit.")

    p.add_argument("--prefix", type=str, default="nmr_dict_code_")
    p.add_argument("--out", type=Path, default=None,
                   help="Output pickle. Default: Datasets/<target>_torch_nmf_codes.pkl")
    p.add_argument("--save-dictionary", action=argparse.BooleanOptionalAction, default=True,
                   help="Also save the learned dictionary artifact (h/c component atoms + "
                        "grids + config) so the NMF visualizers (plot_nmf_components, "
                        "plot_nmf_reconstruction, interpret_nmr_motifs) can plot the OT atoms. "
                        "On by default; pass --no-save-dictionary to skip.")
    p.add_argument("--dict-out", type=Path, default=None,
                   help="Dictionary artifact path. Default: alongside --out as "
                        "<target>_torch_nmf_dictionary.pkl")
    return p.parse_args()


def save_dictionary(path: Path, engine, params: dict, cfg: SweepConfig,
                    h_grid: np.ndarray, c_grid: np.ndarray, h_width: int,
                    code_cols: list[str], repr_loss: str) -> None:
    """Dump a dictionary bundle in the same format the sklearn pipeline uses
    (nmf_dictionary_*.pkl), so the existing NMF visualizers work on the OT atoms.

    Splits the learned dictionary ``engine.components_`` back into per-block H/C
    atoms in physical intensity scale (undoing the modality weights), matching how
    build_dict_codes.py / create_nmr_dictionary_features.py store their bundles.
    """
    from dataclasses import asdict

    comps = engine.components_
    h_comps = comps[:, :h_width] / cfg.h_modality_weight
    c_comps = comps[:, h_width:] / cfg.c_modality_weight
    # config: SweepConfig fields the visualizers read (h_sigma_ppm, width scales,
    # use_peak_width, intensity_transform) plus the raw OT params for provenance.
    config = {**asdict(cfg), **params, "repr_loss": repr_loss}
    bundle = {
        "config": config,
        "engine": "torch_nmf",
        "repr_loss": repr_loss,
        "h_components": h_comps,
        "c_components": c_comps,
        "h_grid_ppm": h_grid,
        "c_grid_ppm": c_grid,
        "feature_columns": code_cols,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(bundle, path)
    print(f"Wrote dictionary artifact ({h_comps.shape[0]} atoms, "
          f"H{h_comps.shape} C{c_comps.shape}) to:\n  {path}")


def build_engine(params: dict, cfg: SweepConfig, h_grid: np.ndarray, c_grid: np.ndarray,
                 args: argparse.Namespace):
    """Construct a TorchNMF from the tuned params. Mirrors tune_representation's
    build_torch_engine, but reads fixed values from `params` instead of suggesting
    them from an Optuna trial."""
    from nmrlib.torch_nmf import TorchNMF, TorchNMFConfig, SpectralGeometry

    repr_loss = params.get("repr_loss", "frobenius")
    common = dict(
        n_components=cfg.n_components, alpha_W=cfg.alpha_W, alpha_H=cfg.alpha_H,
        l1_ratio=cfg.l1_ratio, optimizer=args.optimizer, lr=args.lr,
        n_iter=args.n_iter, transform_n_iter=args.transform_n_iter,
        device=args.device, dtype="float32", verbose=args.verbose,
    )
    if repr_loss in ("frobenius", "kl"):
        return TorchNMF(TorchNMFConfig(loss=repr_loss, **common), geometry=None), repr_loss

    if repr_loss not in OT_FAMILIES:
        raise SystemExit(f"Unknown repr_loss {repr_loss!r}; expected one of "
                         f"{{frobenius, kl}} | {sorted(OT_FAMILIES)}.")

    ot_loss = "sinkhorn" if repr_loss.startswith("sinkhorn") else repr_loss
    # reach => unbalanced OT (only meaningful for sinkhorn_unbalanced).
    reach = params.get("reach") if repr_loss == "sinkhorn_unbalanced" else None
    tcfg = TorchNMFConfig(
        loss="geomloss", ot_loss=ot_loss,
        reach=reach,
        sinkhorn_p=params.get("sinkhorn_p", 2),
        sinkhorn_scaling=params.get("sinkhorn_scaling", 0.7),
        mass_normalize=params.get("mass_normalize", None),
        h_blur=params.get("h_blur"), c_blur=params.get("c_blur"),
        blur=params.get("blur", 0.05),
        sinkhorn_backend=args.sinkhorn_backend,
        **common,
    )
    geom = SpectralGeometry(
        h_coords=h_grid, c_coords=c_grid,
        h_modality_weight=cfg.h_modality_weight, c_modality_weight=cfg.c_modality_weight,
    )
    return TorchNMF(tcfg, geometry=geom), repr_loss


def _has_peaks(v: object) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    try:
        return len(v) > 0
    except TypeError:
        return False


def load_with_spectra(name_or_path: str, sample: int, seed: int) -> pd.DataFrame:
    """Load via the nmrlib registry, drop rows missing either spectrum, optionally
    subsample. No target filtering — ids_nmr_* sets are unlabeled."""
    df = load_dataset(name_or_path)
    complete = df["h_nmr_peaks"].map(_has_peaks) & df["c_nmr_peaks"].map(_has_peaks)
    n_before = len(df)
    df = df[complete].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} of {n_before} rows missing NMR spectra.")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    return df


def build_matrices(df: pd.DataFrame, cfg: SweepConfig, h_step: float, c_step: float):
    h_grid = _make_grid(0.0, 12.0, h_step)
    c_grid = _make_grid(0.0, 220.0, c_step)
    h, c = build_raw_matrices(df, h_grid, c_grid, cfg)
    x = assemble_x(h, c, cfg)
    return h, c, x, h_grid, c_grid


def main() -> None:
    args = parse_args()
    if not args.params_json.exists():
        raise SystemExit(f"--params-json not found: {args.params_json}")
    params = json.loads(args.params_json.read_text())
    cfg = config_from_params(params)

    print(f"Fitting torch NMF on {args.dataset!r} "
          f"(loss={params.get('repr_loss', 'frobenius')}, n_components={cfg.n_components}, "
          f"backend={args.sinkhorn_backend}) ...")
    df_fit = load_with_spectra(args.dataset, args.sample, args.seed)
    h_fit, c_fit, x_fit, h_grid, c_grid = build_matrices(df_fit, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"  fit set {x_fit.shape}; fitting ({args.n_iter} iters) ...")

    engine, repr_loss = build_engine(params, cfg, h_grid, c_grid, args)

    same_set = args.apply_dataset is None
    if same_set:
        codes = engine.fit_transform(x_fit).astype(np.float64)
        df_apply, h_ap, c_ap = df_fit, h_fit, c_fit
    else:
        engine.fit(x_fit)
        print(f"\nTransforming {args.apply_dataset!r} into the code space ...")
        df_apply = load_dataset(args.apply_dataset).reset_index(drop=True)
        h_ap, c_ap, x_ap, _, _ = build_matrices(df_apply, cfg, args.h_step_ppm, args.c_step_ppm)
        codes = engine.transform(x_ap).astype(np.float64)
    print(f"  {codes.shape[0]} rows -> {codes.shape[1]} codes  "
          f"(finite={np.isfinite(codes).all()}, nonneg={(codes >= 0).all()})")

    # Reconstruction error in original intensity space (per modality + total).
    comps = engine.components_
    h_width = h_ap.shape[1]
    h_comps = comps[:, :h_width] / cfg.h_modality_weight
    c_comps = comps[:, h_width:] / cfg.c_modality_weight
    h_recon = _invert_transform(codes @ h_comps, cfg.intensity_transform)
    c_recon = _invert_transform(codes @ c_comps, cfg.intensity_transform)
    h_err = np.linalg.norm(h_ap - h_recon, axis=1)
    c_err = np.linalg.norm(c_ap - c_recon, axis=1)

    width = len(str(codes.shape[1] - 1))
    code_cols = [f"{args.prefix}{i:0{width}d}" for i in range(codes.shape[1])]
    stale = [c for c in df_apply.columns
             if c.startswith(args.prefix) or c.endswith("_reconstruction_error")]
    if stale:
        print(f"  dropping {len(stale)} stale feature columns from a previous NMF")
        df_apply = df_apply.drop(columns=stale)

    codes_df = pd.DataFrame(codes, columns=code_cols, index=df_apply.index)
    codes_df["nmr_h_reconstruction_error"] = h_err
    codes_df["nmr_c_reconstruction_error"] = c_err
    codes_df["nmr_total_reconstruction_error"] = h_err + c_err

    out_df = pd.concat([df_apply, codes_df], axis=1)
    out = args.out
    if out is None:
        target = Path(resolve_dataset(args.apply_dataset if not same_set else args.dataset)).stem
        out = REPO_ROOT / "Datasets" / f"{target}_torch_nmf_codes.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(out)
    print(f"\nWrote {len(out_df)} rows x {out_df.shape[1]} cols "
          f"({len(code_cols)} codes + 3 recon-error cols) to:\n  {out}")

    if args.save_dictionary:
        dict_out = args.dict_out
        if dict_out is None:
            stem = out.stem
            dict_stem = (stem.replace("_codes", "_dictionary")
                         if "_codes" in stem else stem + "_dictionary")
            dict_out = out.with_name(dict_stem + ".pkl")
        save_dictionary(dict_out, engine, params, cfg, h_grid, c_grid,
                        h_width, code_cols, repr_loss)


if __name__ == "__main__":
    main()
