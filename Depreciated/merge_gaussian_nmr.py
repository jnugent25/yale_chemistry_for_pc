"""Merge Gaussian orbital data with matched IDS NMR + RDKit properties.

Uses ids_nmr_matched_gaussian_1k_props.pkl (pre-matched to the Gaussian 1k set)
joined on gaussian `name` = `{molecular_formula}_{last_5_smiles_chars}`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GAUSSIAN_PATH = Path("/Users/jacknugent/Downloads/gaussian_1_1k.pkl")
PROPS_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_matched_gaussian_1k_props.pkl")
OUTPUT_PATH = Path(__file__).resolve().parent / "gaussian_nmr_matched_1k.pkl"


def build_gaussian_name(smiles: str, formula: str) -> str:
    return f"{formula}_{str(smiles)[-5:]}"


def main() -> None:
    gaussian = pd.read_pickle(GAUSSIAN_PATH)
    props = pd.read_pickle(PROPS_PATH)

    props = props.copy()
    props["gauss_name"] = props.apply(
        lambda row: build_gaussian_name(row["smiles"], row["molecular_formula"]),
        axis=1,
    )
    props = props.drop_duplicates("gauss_name", keep="first")

    merged = gaussian.merge(props, left_on="name", right_on="gauss_name", how="left")
    merged = merged.drop(columns=["gauss_name"])

    n_nmr = int(merged["h_nmr_peaks"].notna().sum())
    merged.to_pickle(OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(merged)}")
    print(f"With NMR: {n_nmr} / {len(merged)}")
    print(f"Missing NMR: {len(merged) - n_nmr}")


if __name__ == "__main__":
    main()
