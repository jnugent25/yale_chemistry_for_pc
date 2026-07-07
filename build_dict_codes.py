"""Build NMF dictionary codes for a dataset using the tuned representation.

Fits the NMF dictionary on a (large) FIT dataset, then transforms an APPLY
dataset into that code space and appends the codes + reconstruction-error
features to it, saving a new pickle. This is the "pretrained dictionary applied
to new data" pattern: the codes for the apply set live in the basis learned on
the fit set, so a model trained on the fit set can consume them directly.

All rows of the apply set are kept (row alignment with the input is preserved);
rows with empty spectra just get near-zero codes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from create_nmr_dictionary_features import _invert_transform, _make_grid
from tune_representation import assemble_x, build_raw_matrices
from train_gap_model import config_from_params, load_working_set, make_nmf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tuned NMF dictionary codes for a dataset.")
    p.add_argument("--fit-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"),
                   help="Dataset the NMF dictionary is fit on (large set).")
    p.add_argument("--apply-data", type=Path, default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_all_features.pkl"),
                   help="Dataset to transform into the code space and augment.")
    p.add_argument("--params-json", type=Path, required=True,
                   help="JSON of NMF hyperparameters (e.g. the best sweep trial).")
    p.add_argument("--out", type=Path, default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_tuned_nmf_features.pkl"))
    p.add_argument("--sample", type=int, default=12000, help="Cap on fit-set molecules.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--prefix", type=str, default="nmr_dict_code_", help="Code column name prefix.")
    return p.parse_args()


def build_matrices(df: pd.DataFrame, cfg, h_step: float, c_step: float):
    h_grid = _make_grid(0.0, 12.0, h_step)
    c_grid = _make_grid(0.0, 220.0, c_step)
    h, c = build_raw_matrices(df, h_grid, c_grid, cfg)
    x = assemble_x(h, c, cfg)
    return h, c, x


def main() -> None:
    args = parse_args()
    for pth in (args.fit_data, args.apply_data, args.params_json):
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")

    cfg = config_from_params(json.loads(args.params_json.read_text()))
    print(f"Config: n_components={cfg.n_components}, solver={cfg.solver}, "
          f"transform={cfg.intensity_transform}, multiplicity={cfg.h_multiplicity}")

    # Fit the dictionary on the (clean) fit set.
    print(f"\nFitting NMF dictionary on {args.fit_data.name} ...")
    df_fit, _ = load_working_set(args.fit_data, args.sample, args.seed)
    _, _, x_fit = build_matrices(df_fit, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"  fit set {x_fit.shape}; fitting (max_iter={args.max_iter}) ...")
    nmf = make_nmf(cfg, args.max_iter)
    nmf.fit(x_fit)
    comps = nmf.components_

    # Apply to the target set — keep ALL rows to preserve alignment with the file.
    print(f"\nTransforming {args.apply_data.name} into the code space ...")
    df_apply = pd.read_pickle(args.apply_data)
    df_apply = df_apply.reset_index(drop=True)
    h_ap, c_ap, x_ap = build_matrices(df_apply, cfg, args.h_step_ppm, args.c_step_ppm)
    codes = nmf.transform(x_ap).astype(np.float64)
    print(f"  {codes.shape[0]} rows -> {codes.shape[1]} codes")

    # Reconstruction error in original intensity space (per modality + total).
    h_width = h_ap.shape[1]
    h_comps = comps[:, :h_width] / cfg.h_modality_weight
    c_comps = comps[:, h_width:] / cfg.c_modality_weight
    h_recon = _invert_transform(codes @ h_comps, cfg.intensity_transform)
    c_recon = _invert_transform(codes @ c_comps, cfg.intensity_transform)
    h_err = np.linalg.norm(h_ap - h_recon, axis=1)
    c_err = np.linalg.norm(c_ap - c_recon, axis=1)

    # Assemble code columns; drop any stale dict-code columns from a prior NMF.
    width = len(str(codes.shape[1] - 1))
    code_cols = [f"{args.prefix}{i:0{width}d}" for i in range(codes.shape[1])]
    stale = [ccol for ccol in df_apply.columns
             if ccol.startswith(args.prefix) or ccol.endswith("_reconstruction_error")]
    if stale:
        print(f"  dropping {len(stale)} stale feature columns from a previous NMF")
        df_apply = df_apply.drop(columns=stale)

    codes_df = pd.DataFrame(codes, columns=code_cols, index=df_apply.index)
    codes_df["nmr_h_reconstruction_error"] = h_err
    codes_df["nmr_c_reconstruction_error"] = c_err
    codes_df["nmr_total_reconstruction_error"] = h_err + c_err

    out_df = pd.concat([df_apply, codes_df], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(args.out)
    print(f"\nWrote {len(out_df)} rows x {out_df.shape[1]} cols "
          f"({len(code_cols)} codes + 3 recon-error cols) to:\n  {args.out}")


if __name__ == "__main__":
    main()
