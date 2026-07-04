import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from create_nmr_dictionary_features import build_soft_peak_matrix

# Paths
FEATURES_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_features.pkl")
MODEL_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_model.pkl")
ORIGINAL_DATA_PATH = Path("/Users/jacknugent/Downloads/nmr_gaussian_ids_features_1k.pkl")
OUTPUT_PLOT = Path("nmf_reconstruction_plot.png")

def plot_reconstruction(features_path: Path, model_path: Path, original_data_path: Path, sample_idx: int = 0):
    if not (features_path.exists() and model_path.exists() and original_data_path.exists()):
        print("Error: Required files not found.")
        return

    # Load data
    df_orig = pd.read_pickle(original_data_path)
    df_feats = pd.read_pickle(features_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    config = model["config"]
    h_grid = model["h_grid_ppm"]
    c_grid = model["c_grid_ppm"]
    h_components = model["h_components"]
    c_components = model["c_components"]
    code_cols = model["feature_columns"]

    # Select a sample molecule
    row_orig = df_orig.iloc[sample_idx]
    row_feats = df_feats.iloc[sample_idx]
    smiles = row_orig.get("smiles", "Unknown")
    formula = row_orig.get("molecular_formula", "Unknown")

    print(f"Plotting reconstruction for molecule {sample_idx}: {formula} (SMILES: {smiles})")

    # 1. Build original soft peak matrices
    h_orig = build_soft_peak_matrix(
        pd.Series([row_orig["h_nmr_peaks"]]),
        h_grid,
        config["h_sigma_ppm"],
        modality="h",
        width_scale=config["h_width_scale"],
        use_peak_width=config["use_peak_width"]
    )[0]

    c_orig = build_soft_peak_matrix(
        pd.Series([row_orig["c_nmr_peaks"]]),
        c_grid,
        config["c_sigma_ppm"],
        modality="c",
        width_scale=config["c_width_scale"],
        use_peak_width=config["use_peak_width"]
    )[0]

    # 2. Reconstruct from codes
    codes = row_feats[code_cols].values.astype(float)
    h_recon = codes @ h_components
    c_recon = codes @ c_components

    # Plot Comparison
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), constrained_layout=True)

    # 1H Plot
    axes[0].plot(h_grid, h_orig, label="Original Spectrum (soft)", color="black", lw=1.5)
    axes[0].plot(h_grid, h_recon, label="NMF Reconstruction", color="#1f77b4", lw=1.5, ls="--")
    axes[0].set_xlim(h_grid.max(), h_grid.min())  # reverse X-axis for chemistry convention
    axes[0].set_title(f"1H NMR Reconstruction Comparison (SMILES: {smiles})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Chemical Shift (ppm)")
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].legend()

    # 13C Plot
    axes[1].plot(c_grid, c_orig, label="Original Spectrum (soft)", color="black", lw=1.5)
    axes[1].plot(c_grid, c_recon, label="NMF Reconstruction", color="#d62728", lw=1.5, ls="--")
    axes[1].set_xlim(c_grid.max(), c_grid.min())  # reverse X-axis
    axes[1].set_title(f"13C NMR Reconstruction Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Chemical Shift (ppm)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].legend()

    plt.suptitle(f"NMF NMR Reconstruction for {formula}", fontsize=14, fontweight="bold")
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Successfully saved reconstruction plot to {OUTPUT_PLOT.resolve()}")

if __name__ == "__main__":
    plot_reconstruction(FEATURES_PATH, MODEL_PATH, ORIGINAL_DATA_PATH, sample_idx=5)
