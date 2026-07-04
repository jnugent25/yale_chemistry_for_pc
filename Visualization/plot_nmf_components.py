import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths to the model and output plot
MODEL_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_model.pkl")
OUTPUT_PLOT = Path("nmf_components_plot_kld_nnsvda_10k_ids_nmr.png")

def plot_nmf_components(model_path: Path, comp_indices: list[int] | None = None, output_plot: Path = OUTPUT_PLOT):
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the model artifact
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    h_components = model["h_components"]
    c_components = model["c_components"]
    h_grid = model["h_grid_ppm"]
    c_grid = model["c_grid_ppm"]
    n_components = h_components.shape[0]

    if comp_indices is None:
        comp_indices = list(range(min(4, n_components)))

    n_to_plot = len(comp_indices)
    print(f"Loaded {n_components} components. Plotting {n_to_plot} specified components: {comp_indices}")
    print(f"H components shape: {h_components.shape}, grid range: {h_grid.min():.1f} to {h_grid.max():.1f} ppm")
    print(f"C components shape: {c_components.shape}, grid range: {c_grid.min():.1f} to {c_grid.max():.1f} ppm")

    # Plot the specified components
    fig, axes = plt.subplots(
        nrows=n_to_plot,
        ncols=2,
        figsize=(14, 2.5 * n_to_plot),
        sharex="col",
        constrained_layout=True
    )

    # In case we plot only 1 component, make sure axes is 2D
    if n_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, comp_idx in enumerate(comp_indices):
        if comp_idx >= n_components:
            print(f"Warning: Component index {comp_idx} is out of bounds (max {n_components - 1})")
            continue
            
        # Plot 1H Component
        ax_h = axes[i, 0]
        ax_h.plot(h_grid, h_components[comp_idx], color="#1f77b4", lw=1.5)
        ax_h.set_ylabel(f"Comp {comp_idx}\nIntensity")
        # Reverse X-axis for chemical shifts (chemistry convention: high to low ppm)
        ax_h.set_xlim(h_grid.max(), h_grid.min())
        if i == 0:
            ax_h.set_title("learned 1H NMR Motifs", fontsize=12, fontweight="bold")
        if i == n_to_plot - 1:
            ax_h.set_xlabel("Chemical Shift (ppm)")

        # Plot 13C Component
        ax_c = axes[i, 1]
        ax_c.plot(c_grid, c_components[comp_idx], color="#d62728", lw=1.5)
        # Reverse X-axis for chemical shifts
        ax_c.set_xlim(c_grid.max(), c_grid.min())
        if i == 0:
            ax_c.set_title("learned 13C NMR Motifs", fontsize=12, fontweight="bold")
        if i == n_to_plot - 1:
            ax_c.set_xlabel("Chemical Shift (ppm)")

    plt.suptitle("NMF NMR Dictionary Components (Motifs)", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Successfully saved components plot to {output_plot.resolve()}")

if __name__ == "__main__":
    # Top SHAP features from the plot: e.g., 24, 05, 10, 04, 06, 17, 00, 12, 18
    top_shap_components = [24, 5, 10, 4, 6, 17, 0, 12, 18]
    
    # We can plot the top SHAP components:
    plot_nmf_components(
        MODEL_PATH, 
        comp_indices=top_shap_components, 
        output_plot=Path("top_shap_nmf_components.png")
    )

