"""
Feature importance analysis for Ridge regression.

Tests different combinations of features to show which ones
contribute most to prediction accuracy.

Edit DATASET_NAME below to switch datasets.

Run from project root:
    python ML\\stoich_pca_pipeline\\analyze_feature_importance.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS
# -----------------------------
FEATURES_CSV = PROJECT_ROOT / "results" / DATASET_NAME / "per_video_features.csv"
OUTPUT_CSV   = PROJECT_ROOT / "results" / DATASET_NAME / "feature_importance_analysis.csv"
OUTPUT_JSON  = PROJECT_ROOT / "results" / DATASET_NAME / "feature_importance_analysis.json"


def run_loocv(df, feature_cols, y, label):
    """
    Run Ridge LOOCV with given feature columns.
    Returns a dict with label, features used, and metrics.
    """
    X   = df[feature_cols].values.astype(np.float64)
    loo = LeaveOneOut()
    preds = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        scaler   = StandardScaler()
        X_tr     = scaler.fit_transform(X[train_idx])
        X_te     = scaler.transform(X[test_idx])
        model    = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_tr, y[train_idx])
        preds[test_idx[0]] = model.predict(X_te)[0]

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)

    return {
        "feature_set":    label,
        "n_features":     len(feature_cols),
        "features_used":  ", ".join(feature_cols),
        "mae":            round(mae,  4),
        "rmse":           round(rmse, 4),
        "r2":             round(r2,   4),
    }


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {FEATURES_CSV}\n"
            f"Run extract_features.py first.")

    df = pd.read_csv(FEATURES_CSV)
    print(f"Dataset:  {DATASET_NAME}")
    print(f"Samples:  {len(df)}")

    exclude = {"sample_id", "sample_number", "dominant_cluster",
               "xps_value", "target_value"}
    all_feature_cols = [c for c in df.columns if c not in exclude]

    # Use target_value if present, fall back to xps_value
    y = (df["target_value"] if "target_value" in df.columns
         else df["xps_value"]).values.astype(np.float64)

    # Detect available feature groups
    pca_cols     = [c for c in all_feature_cols if c.startswith("pc")]
    cluster_cols = [c for c in all_feature_cols if c.startswith("cluster_frac")]
    extra_cols   = [c for c in all_feature_cols
                    if not c.startswith("pc") and not c.startswith("cluster_frac")]

    print(f"\nPCA features:     {pca_cols}")
    print(f"Cluster features: {cluster_cols}")
    print(f"Extra features:   {extra_cols}")
    print()

    results = []

    # 1. Baseline — predict mean
    mean_pred = np.full_like(y, y.mean())
    results.append({
        "feature_set":   "baseline_mean",
        "n_features":    0,
        "features_used": "none",
        "mae":           round(mean_absolute_error(y, mean_pred), 4),
        "rmse":          round(np.sqrt(mean_squared_error(y, mean_pred)), 4),
        "r2":            0.0,
    })

    # 2. All features
    results.append(run_loocv(df, all_feature_cols,
                             y, "ALL features"))

    # 3. Without extra features (PCA + clusters only)
    no_extra = pca_cols + cluster_cols
    if no_extra:
        results.append(run_loocv(df, no_extra,
                                 y, "PCA + clusters (no extra)"))

    # 4. Extra features only (e.g. lattice parameter)
    if extra_cols:
        results.append(run_loocv(df, extra_cols,
                                 y, f"Extra only ({', '.join(extra_cols)})"))

    # 5. PCA only
    if pca_cols:
        results.append(run_loocv(df, pca_cols, y, "PCA features only"))

    # 6. Cluster fractions only
    if cluster_cols:
        results.append(run_loocv(df, cluster_cols,
                                 y, "Cluster fractions only"))

    # 7. PCA mean only (no std/final)
    pca_mean_cols = [c for c in pca_cols if c.endswith("_mean")]
    if pca_mean_cols:
        results.append(run_loocv(df, pca_mean_cols,
                                 y, "PCA mean only"))

    # 8. PCA final only (last frame)
    pca_final_cols = [c for c in pca_cols if c.endswith("_final")]
    if pca_final_cols:
        results.append(run_loocv(df, pca_final_cols,
                                 y, "PCA final frame only"))

    # 9. Each extra feature individually
    for col in extra_cols:
        results.append(run_loocv(df, [col], y, f"Only: {col}"))

    # Save CSV
    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("r2", ascending=False).reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"dataset": DATASET_NAME,
                   "results": results}, f, indent=2)
    print(f"Saved: {OUTPUT_JSON}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"[{DATASET_NAME}] FEATURE IMPORTANCE ANALYSIS (sorted by R²)")
    print("=" * 75)
    print(f"{'Feature Set':<40}{'N':>4}{'MAE':>10}{'RMSE':>10}{'R²':>10}")
    print("-" * 75)
    for _, row in out_df.iterrows():
        print(f"{row['feature_set']:<40}{row['n_features']:>4}"
              f"{row['mae']:>10.4f}{row['rmse']:>10.4f}{row['r2']:>10.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()