"""
Script 5: train a regressor on per-video PCA+KMeans features using
leave-one-out cross-validation. Compares Ridge and Random Forest.

Edit DATASET_NAME below to switch datasets.
Results saved to results/{DATASET_NAME}/ so datasets never overwrite each other.

Run from project root:
    python ML\\stoich_pca_pipeline\\train_regressor.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS — derived automatically from DATASET_NAME
# -----------------------------
# FEATURES_CSV:  feature table produced by extract_features.py
# PRED_CSV:      per-video predictions from all models saved here
# METRICS_JSON:  MAE / RMSE / R² for all models saved here
# RANDOM_SEED:   fixed for reproducible Random Forest results
FEATURES_CSV = PROJECT_ROOT / "results" / DATASET_NAME / "per_video_features.csv"
OUTPUT_DIR   = PROJECT_ROOT / "results" / DATASET_NAME
PRED_CSV     = OUTPUT_DIR / "regression_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "regression_metrics.json"

RANDOM_SEED  = 42


def evaluate_loocv(model_factory, X, y, sample_ids, model_name):
    """
    Run LOOCV with a given model factory. Standardizes features per fold
    to prevent data leakage from the test sample into the scaler fit.
    Returns (predictions array, metrics dict).
    """
    loo    = LeaveOneOut()
    preds  = np.zeros(len(y))
    actuals = y.copy()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        scaler      = StandardScaler()
        X_train_s   = scaler.fit_transform(X_train)
        X_test_s    = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_s, y_train)
        preds[test_idx[0]] = model.predict(X_test_s)[0]

    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)

    print(f"\n=== {model_name} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    return preds, {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    """
    Load per-video PCA + K-Means features, run leave-one-out cross-validation
    with Ridge regression and Random Forest, compare against a mean baseline,
    and save predictions and metrics to results/{DATASET_NAME}/.

    Feature standardization is applied inside each LOOCV fold to prevent
    data leakage. Lattice parameters (CSO) are included automatically if
    present in the features CSV produced by extract_features.py.
    """
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {FEATURES_CSV}\n"
            f"Run extract_features.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    print(f"Dataset:  {DATASET_NAME}")
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")

    # Exclude non-feature columns — target_value is the label
    exclude_cols = {"sample_id", "sample_number", "dominant_cluster",
                    "xps_value", "target_value"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Use target_value if present, fall back to xps_value for STO compatibility
    if "target_value" in df.columns:
        y = df["target_value"].values.astype(np.float64)
    else:
        y = df["xps_value"].values.astype(np.float64)

    # Drop any feature columns that have NaN (e.g. missing lattice params)
    feature_df   = df[feature_cols].copy()
    nan_cols     = feature_df.columns[feature_df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: dropping columns with NaN values: {nan_cols}")
        feature_df = feature_df.drop(columns=nan_cols)
        feature_cols = list(feature_df.columns)

    X          = feature_df.values.astype(np.float64)
    sample_ids = df["sample_id"].values

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target range: {y.min():.4f} - {y.max():.4f}")
    print(f"Target mean:  {y.mean():.4f},  std: {y.std():.4f}")

    results   = {}
    all_preds = {"sample_id": sample_ids, "actual": y}

    # ── Run LOOCV for each model ───────────────────────────────────────
    # Ridge uses inner cross-validation to select the regularization alpha.
    # Random Forest uses fixed hyperparameters.
    # Both are compared against a trivial baseline of predicting the mean.

    # Ridge
    def ridge_factory():
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    preds_ridge, m_ridge = evaluate_loocv(
        ridge_factory, X, y, sample_ids, "Ridge (CV alpha)")
    all_preds["ridge_pred"]    = preds_ridge
    all_preds["ridge_abs_err"] = np.abs(preds_ridge - y)
    results["ridge"]           = m_ridge

    # Random Forest
    def rf_factory():
        return RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)

    preds_rf, m_rf = evaluate_loocv(
        rf_factory, X, y, sample_ids, "Random Forest")
    all_preds["rf_pred"]    = preds_rf
    all_preds["rf_abs_err"] = np.abs(preds_rf - y)
    results["random_forest"] = m_rf

    # Baseline: predict mean
    mean_pred = np.full_like(y, y.mean())
    base_mae  = mean_absolute_error(y, mean_pred)
    base_rmse = np.sqrt(mean_squared_error(y, mean_pred))
    print(f"\n=== Baseline (predict mean) ===")
    print(f"MAE  : {base_mae:.4f}")
    print(f"RMSE : {base_rmse:.4f}")
    results["baseline_mean"] = {
        "model": "baseline_mean",
        "mae":   base_mae,
        "rmse":  base_rmse,
        "r2":    0.0,
    }

    # Save predictions CSV
    preds_df = (pd.DataFrame(all_preds)
                .sort_values("actual")
                .reset_index(drop=True))
    preds_df.to_csv(PRED_CSV, index=False)
    print(f"\nSaved predictions: {PRED_CSV}")

    # Save metrics JSON
    with open(METRICS_JSON, "w") as f:
        json.dump({"dataset": DATASET_NAME, **results}, f, indent=2)
    print(f"Saved metrics:     {METRICS_JSON}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"[{DATASET_NAME}] SUMMARY (Leave-One-Out CV, {len(df)} samples)")
    print("=" * 55)
    print(f"{'Model':<25}{'MAE':>10}{'RMSE':>10}{'R^2':>10}")
    print("-" * 55)
    for key in ("baseline_mean", "ridge", "random_forest"):
        r = results[key]
        print(f"{r['model']:<25}{r['mae']:>10.4f}"
              f"{r['rmse']:>10.4f}{r['r2']:>10.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()