"""Evaluate An et al. (2014) fixed MLR logP model on IDS NMR + props datasets.

Paper: Global Model for Octanol-Water Partition Coefficients from Proton NMR
Spectra (DOI 10.1002/minf.201300172)

Methods reproduced:
- 24 x 0.5 ppm bins over 0-12 ppm from 1H peak tables (shift, nH, width proxy)
- Integration descriptor = sum(nH) per bin
- Broadness descriptors = b1-b3 molecule-level broad proton widths
- Fixed Eq. 6 coefficients (no re-fitting)

Variable mapping: paper labels x0.5, x1, ... are the *upper edges* of 0.5 ppm bins.
E.g. x1 -> bin (0.5, 1.0] ppm -> column an14_x1.
Broadness b1-b3 -> columns an14_b1, an14_b2, an14_b3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from an2014_nmr import an2014_features_from_peaks, predict_an2014_mlr


def evaluate(df: pd.DataFrame) -> dict[str, float]:
    preds = df.apply(predict_an2014_mlr, axis=1).to_numpy(dtype=float)
    y = df["log_p"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(preds)
    y, preds = y[mask], preds[mask]
    return {
        "n": int(mask.sum()),
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
        "mae": float(mean_absolute_error(y, preds)),
        "r": float(np.corrcoef(y, preds)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptors",
        default="nmr_descriptors_df_10k.pkl",
    )
    parser.add_argument(
        "--props",
        default="/Users/jacknugent/Downloads/ids_100k_props.pkl",
    )
    parser.add_argument(
        "--nmr",
        default="/Users/jacknugent/Downloads/ids_nmr_100k.pkl",
        help="Optional raw NMR pickle for larger smiles-joined evaluation",
    )
    parser.add_argument(
        "--out",
        default="an2014_mlr_transfer_results.csv",
    )
    args = parser.parse_args()

    props = pd.read_pickle(args.props)[["smiles", "log_p"]]

    desc = pd.read_pickle(args.descriptors)
    merged_desc = desc.merge(props, on="smiles", how="inner")
    m_desc = evaluate(merged_desc)
    print("=== nmr_descriptors_df_10k + props (smiles join) ===")
    print(
        f"n={m_desc['n']}  R²={m_desc['r2']:.4f}  RMSE={m_desc['rmse']:.3f}  "
        f"MAE={m_desc['mae']:.3f}  r={m_desc['r']:.4f}"
    )

    nmr_path = Path(args.nmr)
    if nmr_path.exists():
        nmr = pd.read_pickle(nmr_path)
        merged_nmr = nmr.merge(props, on="smiles", how="inner")
        feat = pd.DataFrame(
            [an2014_features_from_peaks(r.h_nmr_peaks) for r in merged_nmr.itertuples()]
        )
        merged_nmr = pd.concat(
            [merged_nmr[["smiles", "log_p"]].reset_index(drop=True), feat],
            axis=1,
        )
        m_nmr = evaluate(merged_nmr)
        print("=== ids_nmr_100k + props (smiles join, featurized) ===")
        print(
            f"n={m_nmr['n']}  R²={m_nmr['r2']:.4f}  RMSE={m_nmr['rmse']:.3f}  "
            f"MAE={m_nmr['mae']:.3f}  r={m_nmr['r']:.4f}"
        )
        out = merged_nmr[["smiles", "log_p"]].copy()
        out["an2014_pred"] = merged_nmr.apply(predict_an2014_mlr, axis=1)
        out["residual"] = out["an2014_pred"] - out["log_p"]
        out.to_csv(args.out, index=False)
        print(f"Saved {args.out}")

    print("\nNote: ids_100k_props.log_p is RDKit cLogP (exact match).")
    print("Paper trained on ECOSAR experimental logP (n=140).")


if __name__ == "__main__":
    main()
