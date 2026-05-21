"""
Compute per-video and overall metrics for any subset of clusters, using the
per-frame predictions CSV from train_gated_cnn_final.py.

No retraining. Just filters the per-frame predictions, recomputes per-video
means, and reports MAE/RMSE/R^2.

Edit DATASET_NAME and CLUSTER_SUBSETS below, then run:
    python ML\\stoich_pca_pipeline\\evaluate_cluster_subsets.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS — derived automatically from DATASET_NAME
# -----------------------------
# PER_FRAME_CSV: per-frame predictions from train_gated_cnn_final.py
# OUTPUT_CSV:    comparison table saved after evaluating all subsets
PER_FRAME_CSV = (PROJECT_ROOT / "results" / DATASET_NAME
                 / "gated_cnn_cluster_filtered"
                 / "gated_cnn_cluster_filtered_per_frame.csv")
OUTPUT_CSV    = (PROJECT_ROOT / "results" / DATASET_NAME
                 / "gated_cnn_cluster_filtered"
                 / "cluster_subset_comparison.csv")


# -----------------------------
# CLUSTER SUBSETS TO EVALUATE
# -----------------------------
# Each entry is evaluated independently using the already-saved predictions.
# No retraining happens — this just filters per-frame predictions differently.
#
# How to use:
#   1. Run train_gated_cnn_final.py first to generate the per-frame CSV.
#   2. Look at plot_kmeans.py centroid images to understand what each cluster looks like.
#   3. Add entries here with meaningful names and cluster lists.
#   4. Run this script to compare metrics across different cluster combinations.
#
# The script will also always include "all_frames" automatically.
# Add your own subsets below:

CLUSTER_SUBSETS = [
    # Edit or add entries here — name can be anything descriptive
    # {"name": "my_selection",   "clusters": [1, 3, 5]},
    # {"name": "just_cluster_1", "clusters": [1]},
    # {"name": "just_cluster_2", "clusters": [2]},
]

# If CLUSTER_SUBSETS is empty, the script will auto-generate one entry
# per cluster (just_cluster_1, just_cluster_2, ...) so you can see each
# cluster's individual contribution without any manual setup.


def compute_metrics(pf_df, cluster_set, name):
    """
    Filter per-frame predictions to given clusters, compute per-video mean,
    and return metrics dict. Videos with no frames in the cluster set are
    excluded — n_videos_covered shows how many remain.
    """
    subset = pf_df[pf_df["cluster"].isin(cluster_set)]

    grouped = subset.groupby("sample_id").agg(
        actual_xps=("actual_xps", "first"),
        predicted_xps=("predicted_xps", "mean"),
        n_frames=("predicted_xps", "size"),
        pred_std=("predicted_xps", "std"),
    ).reset_index()

    if len(grouped) == 0:
        return None, None

    actuals = grouped["actual_xps"].values
    preds   = grouped["predicted_xps"].values

    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)

    return grouped, {
        "subset_name":          name,
        "clusters":             str(sorted(cluster_set)),
        "n_videos_covered":     len(grouped),
        "mean_frames_per_video": float(grouped["n_frames"].mean()),
        "mae":                  round(mae,  4),
        "rmse":                 round(rmse, 4),
        "r2":                   round(r2,   4),
    }


def main():
    """
    Load per-frame predictions CSV, evaluate all cluster subsets,
    print a summary table, and save results to OUTPUT_CSV.

    If CLUSTER_SUBSETS is empty, auto-generates one entry per cluster
    so you can see which individual clusters contribute most to accuracy.
    Always includes an 'all_frames' baseline entry.
    """
    if not PER_FRAME_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {PER_FRAME_CSV}\n"
            f"Run train_gated_cnn_final.py first.")

    pf_df = pd.read_csv(str(PER_FRAME_CSV))
    n_videos = pf_df["sample_id"].nunique()

    print(f"Dataset:          {DATASET_NAME}")
    print(f"Loaded {len(pf_df)} per-frame predictions")
    print(f"Unique samples:   {n_videos}")

    all_clusters = sorted(pf_df["cluster"].unique())
    print(f"Clusters present: {all_clusters}\n")

    # Build the final list of subsets to evaluate
    subsets_to_run = []

    # Always add all_frames baseline first
    subsets_to_run.append({
        "name":     "all_frames",
        "clusters": all_clusters,
    })

    if CLUSTER_SUBSETS:
        # Use manually defined subsets
        subsets_to_run.extend(CLUSTER_SUBSETS)
    else:
        # Auto-generate one entry per cluster
        print("CLUSTER_SUBSETS is empty — auto-generating one entry per cluster.\n")
        for c in all_clusters:
            subsets_to_run.append({
                "name":     f"just_cluster_{c}",
                "clusters": [c],
            })

    # Evaluate each subset
    results = []
    for subset in subsets_to_run:
        name     = subset["name"]
        clusters = subset["clusters"]
        print(f"\n=== {name}: clusters {clusters} ===")
        grouped, metrics = compute_metrics(pf_df, set(clusters), name)
        if metrics is None:
            print("  No data for this cluster set.")
            continue
        print(f"  Videos covered:    {metrics['n_videos_covered']}/{n_videos}")
        print(f"  Mean frames/video: {metrics['mean_frames_per_video']:.1f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R^2:  {metrics['r2']:.4f}")
        results.append(metrics)

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved comparison: {OUTPUT_CSV}")

    # Print summary table
    print("\n" + "=" * 85)
    print(f"[{DATASET_NAME}] CLUSTER SUBSET COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'subset':<30}{'clusters':<20}{'n_videos':>10}"
          f"{'mean_fr':>10}{'MAE':>10}{'RMSE':>10}{'R^2':>10}")
    print("-" * 85)
    for r in results:
        print(f"{r['subset_name']:<30}{r['clusters']:<20}"
              f"{r['n_videos_covered']:>10}"
              f"{r['mean_frames_per_video']:>10.1f}"
              f"{r['mae']:>10.4f}{r['rmse']:>10.4f}{r['r2']:>10.4f}")
    print("=" * 85)
    print("\nTip: the subset with the highest R^2 and lowest MAE is the best")
    print("     cluster filter to use in TRAINING_CLUSTERS in train_gated_cnn_final.py")


if __name__ == "__main__":
    main()