import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem

# Import helper functions from create_nmr_dictionary_features
import sys
sys.path.append("/Users/jacknugent/Desktop/yale_chemistry_project")
from create_nmr_dictionary_features import build_soft_peak_matrix, _make_grid, NMRDictionaryConfig

def plot_aromatic_examples(data_path: Path, output_image: Path = Path("figures/aromatic_nmr_examples.png"), num_examples: int = 3):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_pickle(data_path)
    
    # Filter for aromatic molecules and sort by size (number of heavy atoms)
    print("Filtering and sorting aromatic molecules by heavy atom count...")
    aromatic_list = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None and any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            aromatic_list.append((idx, mol.GetNumHeavyAtoms()))
            
    if not aromatic_list:
        print("No aromatic molecules found in the dataset!")
        return
        
    # Sort by number of heavy atoms (ascending)
    aromatic_list.sort(key=lambda x: x[1])
    aromatic_indices = [x[0] for x in aromatic_list[:num_examples]]
    
    examples_df = df.loc[aromatic_indices].head(num_examples)
    
    # Use standard NMR dictionary configuration settings
    config = NMRDictionaryConfig()
    h_grid = _make_grid(config.h_min_ppm, config.h_max_ppm, config.h_step_ppm)
    c_grid = _make_grid(config.c_min_ppm, config.c_max_ppm, config.c_step_ppm)
    
    # Build soft peak matrices for these examples
    h_matrix = build_soft_peak_matrix(
        examples_df["h_nmr_peaks"], h_grid, config.h_sigma_ppm, modality="h",
        width_scale=config.h_width_scale, use_peak_width=config.use_peak_width
    )
    c_matrix = build_soft_peak_matrix(
        examples_df["c_nmr_peaks"], c_grid, config.c_sigma_ppm, modality="c",
        width_scale=config.c_width_scale, use_peak_width=config.use_peak_width
    )
    
    # Set up subplots
    fig, axes = plt.subplots(
        nrows=num_examples,
        ncols=2,
        figsize=(15, 4 * num_examples),
        constrained_layout=True
    )
    
    for i, (_, row) in enumerate(examples_df.iterrows()):
        smiles = row["smiles"]
        formula = row["molecular_formula"]
        
        # 1H NMR Plot
        ax_h = axes[i, 0]
        # Reconstructed smooth spectrum
        ax_h.plot(h_grid, h_matrix[i], color="#1f77b4", lw=1.5, label="Smooth envelope")
        
        # Original peaks as vertical lines (sticks)
        h_peaks = row["h_nmr_peaks"]
        has_sticks_h = False
        for peak in h_peaks:
            center = peak.get("delta", peak.get("centroid"))
            weight = peak.get("nH", 1.0)
            if center is not None:
                # scale height to match the smooth curve roughly
                ax_h.axvline(x=center, ymin=0, ymax=0.8, color="#1f77b4", alpha=0.5, linestyle="--", lw=1)
                has_sticks_h = True
        
        ax_h.set_xlim(h_grid.max(), h_grid.min()) # Reverse x-axis
        ax_h.set_ylabel("Intensity")
        ax_h.set_title(f"Example {i+1}: {formula} - 1H NMR\nSMILES: {smiles}", fontsize=10)
        if i == num_examples - 1:
            ax_h.set_xlabel("Chemical Shift (ppm)")
            
        # 13C NMR Plot
        ax_c = axes[i, 1]
        # Reconstructed smooth spectrum
        ax_c.plot(c_grid, c_matrix[i], color="#d62728", lw=1.5, label="Smooth envelope")
        
        # Original peaks as vertical lines (sticks)
        c_peaks = row["c_nmr_peaks"]
        for peak in c_peaks:
            center = peak.get("delta (ppm)")
            weight = peak.get("integral", peak.get("intensity", 1.0))
            if center is not None:
                ax_c.axvline(x=center, ymin=0, ymax=0.8, color="#d62728", alpha=0.5, linestyle="--", lw=1)
                
        ax_c.set_xlim(c_grid.max(), c_grid.min()) # Reverse x-axis
        ax_c.set_ylabel("Intensity")
        ax_c.set_title(f"Example {i+1}: {formula} - 13C NMR", fontsize=10)
        if i == num_examples - 1:
            ax_c.set_xlabel("Chemical Shift (ppm)")
            
    plt.suptitle("NMR Spectra of Aromatic-like Molecules (1H and 13C)", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"Successfully saved plot to {output_image.resolve()}")

if __name__ == "__main__":
    DATA_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_10k_filtered.pkl")
    plot_aromatic_examples(DATA_PATH)
