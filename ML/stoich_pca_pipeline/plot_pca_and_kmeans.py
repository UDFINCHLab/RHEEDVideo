"""
Plot PCA and K-Means results from the combined H5 file.

Run this after combine_and_pca.py to visualize:
  - PCA eigenvalue time series + eigenvector spatial modes
  - K-Means cluster trajectory plots + centroid images

Edit DATASET_NAME below to switch datasets.
K value is auto-read from the H5 file — no manual setting needed.

Run from project root:
    python ML\\stoich_pca_pipeline\\plot_pca_and_kmeans.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR       = PROJECT_ROOT / "ML"
sys.path.insert(0, str(ML_DIR))

from plot_pca    import plot_pca       # noqa: E402
from plot_kmeans import plot_k_means   # noqa: E402


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS — derived automatically from DATASET_NAME
# -----------------------------
COMBINED_H5     = PROJECT_ROOT / "results" / DATASET_NAME / f"combined_{DATASET_NAME}.h5"
PCA_PLOT_DIR    = PROJECT_ROOT / "results" / DATASET_NAME / "plots_pca"
KMEANS_PLOT_DIR = PROJECT_ROOT / "results" / DATASET_NAME / "plots_kmeans"


# -----------------------------
# PLOT SETTINGS
# -----------------------------
PCA_FIG_SIZE    = (3.375, 6.75)    # single journal column, tall for 6 components
KMEANS_FIG_SIZE = (3.375, 3.375)   # single journal column, square


def main():
    """
    Generate PCA and K-Means plots from the combined H5 file.
    K value is auto-read from whatever K-Means run was saved in the H5.
    No retraining — purely visualization.
    """
    if not COMBINED_H5.exists():
        raise FileNotFoundError(
            f"Not found: {COMBINED_H5}\n"
            f"Run combine_and_pca.py first.")

    PCA_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    KMEANS_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # PCA plots: eigenvalue time series + eigenvector spatial modes
    print(f"\n--- Generating PCA plots ---")
    plot_pca(
        Input_File=str(COMBINED_H5),
        Out_Path=str(PCA_PLOT_DIR),
        Title=f"{DATASET_NAME} Combined RHEED Videos",
        Num_Vectors=-1,
        Fig_Size=PCA_FIG_SIZE,
        show=False,
    )
    print(f"PCA plots saved to: {PCA_PLOT_DIR}\n")

    # K-Means plots: cluster trajectory + centroid images
    print(f"--- Generating K-Means plots ---")
    plot_k_means(
        Input_File=str(COMBINED_H5),
        Out_Path=str(KMEANS_PLOT_DIR),
        Fig_size=KMEANS_FIG_SIZE,
        show=False,
    )
    print(f"K-Means plots saved to: {KMEANS_PLOT_DIR}\n")

    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"PCA plots:     {PCA_PLOT_DIR}")
    print(f"K-Means plots: {KMEANS_PLOT_DIR}")


if __name__ == "__main__":
    main()