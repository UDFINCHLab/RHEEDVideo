# Stoichiometry Prediction Pipeline — Setup & Usage Guide

This pipeline predicts thin-film stoichiometry from RHEED videos using PCA, K-Means clustering, and a gated CNN regressor.  
It was developed and validated on **CSO (CaSnO₃)** data. This guide shows you how to set it up for CSO **and** how to adapt it for a new material.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Prerequisites](#2-prerequisites)
3. [Dataset Folder Setup (Manual Step)](#3-dataset-folder-setup-manual-step)
4. [Path Configuration in Scripts](#4-path-configuration-in-scripts)
5. [Running the Pipeline — Step by Step](#5-running-the-pipeline--step-by-step)
6. [Adapting for a New Material](#6-adapting-for-a-new-material)
7. [Output Files Reference](#7-output-files-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Repository Structure

After cloning, the relevant folders are:

```
rheed-ml-pipeline-final/
│
├── ML/
│   ├── pca.py                          # PCA computation (called by combine_and_pca.py)
│   ├── k_means.py                      # K-Means clustering (called by combine_and_pca.py)
│   ├── pre_processing.py               # Frame-level preprocessing utilities
│   ├── run_full.py                     # Optional: run full pipeline end-to-end
│   │
│   └── stoich_pca_pipeline/            # ← Main pipeline scripts
│       ├── global_crop.py              # Step 1: Compute global crop from all videos
│       ├── batch_preprocess.py         # Step 2: Preprocess all videos → HDF5
│       ├── combine_and_pca.py          # Step 3: PCA + K-Means on combined data
│       ├── plot_pca_and_kmeans.py      # Step 3b: Visualize PCA/K-Means results
│       ├── extract_features.py         # Step 4: Extract per-frame cluster features
│       ├── train_gated_cnn_final.py    # Step 5: Train CNN with LOOCV
│       ├── evaluate_cluster_subsets.py # Step 6: Evaluate cluster subsets
│       ├── plot_per_frame_predictions.py  # Step 6b: Plot per-frame prediction curves
│       ├── feature_importance_analysis.py # Optional: feature importance
│       └── train_regressor.py          # Optional: train linear/tree regressor
│
├── stoich_dataset/                     # ← YOU CREATE THIS (see Section 3)
│   └── CSO/
│       ├── labels/
│       ├── preprocessed_h5/
│       └── videos_raw/
│
└── results/                            # Auto-created by scripts
    └── CSO/
        └── ...
```

---

## 2. Prerequisites

**Python environment** — activate the project virtual environment:

```powershell
# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

**Required packages** (already in `requirements.txt`):

```
numpy, pandas, h5py, opencv-python, scikit-learn, torch, torchvision, matplotlib, scipy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Folder Setup (Manual Step)

> ⚠️ **This step is manual. The scripts cannot run without these folders and files.**

You need to create a dataset folder for your material. The folder name must match the `DATASET_NAME` variable in each script (see Section 4).

### For CSO (existing dataset):

Create this exact folder structure under the project root:

```
stoich_dataset/
└── CSO/
    ├── videos_raw/          ← Put all .avi RHEED videos here
    │   ├── CSO_001.avi
    │   ├── CSO_002.avi
    │   └── ...
    │
    ├── labels/              ← Put your XPS/stoichiometry labels CSV here
    │   └── labels_CSO.csv
    │
    └── preprocessed_h5/     ← Leave empty; Step 2 fills this automatically
```

### Labels CSV format:

The labels CSV must have exactly these two columns (column names are case-sensitive):

```
sample_id,xps_value
CSO_001,0.512
CSO_002,0.489
...
```

- `sample_id` must match the video filename **without the extension** (e.g., `CSO_001` for `CSO_001.avi`)
- `xps_value` is the atomic fraction from XPS measurement (float between 0 and 1)

### Windows commands to create folders:

```powershell
mkdir stoich_dataset\CSO\videos_raw
mkdir stoich_dataset\CSO\labels
mkdir stoich_dataset\CSO\preprocessed_h5
```

---

## 4. Path Configuration in Scripts

Every script in `stoich_pca_pipeline/` has a **configuration block at the top**. You only need to change `DATASET_NAME` for most scripts. Some scripts have additional settings.

### The one variable you always change:

```python
# ─── USER CONFIGURATION ────────────────────────────────
DATASET_NAME = "CSO"      # ← Change this to your material folder name
# ───────────────────────────────────────────────────────
```

All paths (input videos, labels, output results) are built automatically from this name.

### Scripts with additional settings to review:

| Script | Extra settings to check |
|--------|------------------------|
| `global_crop.py` | `PERCENTILE` (default 97), `PADDING` (default 15) |
| `batch_preprocess.py` | `Frame_Period`, `RSS`, `BKGRND_CROP` — preprocessing flags |
| `combine_and_pca.py` | `N_COMPONENTS` (PCA components, default 6), `K` (K-Means clusters, default 5) |
| `train_gated_cnn_final.py` | `SELECTED_CLUSTERS` — which clusters to use for training |

---

## 5. Running the Pipeline — Step by Step

Run all scripts from the **project root**, not from inside the ML folder.

---

### Step 1 — Compute global crop

```bash
python ML/stoich_pca_pipeline/global_crop.py
```

**What it does:** Samples frames from all videos, finds the bright diffraction region, computes one bounding box that covers all videos.

**Output:**
```
ML/stoich_pca_pipeline/global_crop_CSO.txt          ← crop coordinates
ML/stoich_pca_pipeline/debug_global_crop_CSO/       ← debug images (one per video)
```

**Before proceeding:** Open a few images in `debug_global_crop_CSO/`. You should see a green rectangle around the diffraction streaks. If it looks too loose or too tight, adjust `PADDING` and `PERCENTILE` in `global_crop.py` and rerun.

---

### Step 2 — Batch preprocess all videos

```bash
python ML/stoich_pca_pipeline/batch_preprocess.py
```

**What it does:** Applies alignment, crop, and background subtraction to every video. Saves each as an individual `.h5` file.

**Output:**
```
stoich_dataset/CSO/preprocessed_h5/
    CSO_001.h5
    CSO_002.h5
    ...
```

**Runtime:** ~5–15 minutes depending on number of videos and frame count.

---

### Step 3 — Combine, PCA, and K-Means

```bash
python ML/stoich_pca_pipeline/combine_and_pca.py
```

**What it does:** Merges all per-video H5 files into one combined file, runs PCA, runs K-Means clustering, and saves results.

**Output:**
```
results/CSO/
    combined_CSO.h5          ← all frames + PCA + K-Means labels
```

---

### Step 3b — Plot PCA and K-Means results

```bash
python ML/stoich_pca_pipeline/plot_pca_and_kmeans.py
```

**Output:**
```
results/CSO/pca_plots/        ← eigenvalue plots, eigenvector images
results/CSO/kmeans_plots/     ← cluster trajectory plots, centroid images
```

Review these before moving to Step 4. The cluster trajectories should show meaningful groupings.

---

### Step 4 — Extract per-frame cluster features

```bash
python ML/stoich_pca_pipeline/extract_features.py
```

**Output:**
```
results/CSO/features/
    per_frame_features_CSO.csv
```

---

### Step 5 — Train gated CNN (LOOCV)

```bash
python ML/stoich_pca_pipeline/train_gated_cnn_final.py
```

**What it does:** Trains a CNN using Leave-One-Out Cross Validation. For each fold, one video is held out as the test sample. Only frames from `SELECTED_CLUSTERS` are used.

**⚠️ Set `SELECTED_CLUSTERS` first:**  
After reviewing the K-Means plots from Step 3b, open `train_gated_cnn_final.py` and set:

```python
SELECTED_CLUSTERS = [1, 2, 4]   # ← change to clusters with clean diffraction patterns
```

**Output:**
```
results/CSO/gated_cnn/
    per_frame_predictions_CSO.csv
    loocv_summary_CSO.csv
    model_fold_*.pt               ← saved model weights per fold
```

**Runtime:** ~6–12 hours. Run overnight.

---

### Step 6 — Evaluate cluster subsets

```bash
python ML/stoich_pca_pipeline/evaluate_cluster_subsets.py
```

Prints MAE, RMSE, R² for each cluster subset combination.

---

### Step 6b — Plot per-frame predictions

```bash
python ML/stoich_pca_pipeline/plot_per_frame_predictions.py
```

**Output:**
```
results/CSO/plots/
    per_frame_predictions_*.png
```

---

## 6. Adapting for a New Material

Say you have a new material called **STO**. Here is exactly what to do.

### A. Create the data folder

```powershell
mkdir stoich_dataset\STO\videos_raw
mkdir stoich_dataset\STO\labels
mkdir stoich_dataset\STO\preprocessed_h5
```

Copy your `.avi` RHEED videos into `videos_raw/` and your XPS labels CSV into `labels/`.  
Name the CSV `labels_STO.csv` and make sure `sample_id` values match video filenames.

### B. Change `DATASET_NAME` in every script

Open each script listed below and change the top line:

```python
DATASET_NAME = "STO"    # was "CSO"
```

Scripts to update:

- `global_crop.py`
- `batch_preprocess.py`
- `combine_and_pca.py`
- `plot_pca_and_kmeans.py`
- `extract_features.py`
- `train_gated_cnn_final.py`
- `evaluate_cluster_subsets.py`
- `plot_per_frame_predictions.py`

### C. Re-run global crop

The crop coordinates are material-specific (the diffraction region is different for each material). Always re-run `global_crop.py` first and visually verify the debug images before batch preprocessing.

### D. Re-tune K-Means K value

The optimal number of clusters may differ between materials. After running `combine_and_pca.py`, review the K-Means plots and adjust `K` in `combine_and_pca.py` if needed.

### E. Re-select `SELECTED_CLUSTERS`

After K-Means, look at the centroid images in `results/STO/kmeans_plots/`. Choose clusters that show clean, sharp diffraction patterns for `SELECTED_CLUSTERS` in `train_gated_cnn_final.py`.

### Summary of what changes per material

| What changes | Where to change it |
|---|---|
| `DATASET_NAME` | Top of every script |
| Global crop coordinates | Auto-computed by `global_crop.py` — just rerun |
| Number of K-Means clusters (`K`) | `combine_and_pca.py` → `K = ...` |
| Which clusters to train on | `train_gated_cnn_final.py` → `SELECTED_CLUSTERS` |
| XPS labels CSV | New file in `stoich_dataset/STO/labels/labels_STO.csv` |

Everything else (folder structure, script logic, output paths) is the same.

---

## 7. Output Files Reference

| File | Location | Description |
|---|---|---|
| `global_crop_CSO.txt` | `ML/stoich_pca_pipeline/` | Crop bounding box for preprocessing |
| `combined_CSO.h5` | `results/CSO/` | All frames + PCA eigenvectors + K-Means labels |
| `per_frame_features_CSO.csv` | `results/CSO/features/` | Per-frame cluster membership features |
| `per_frame_predictions_CSO.csv` | `results/CSO/gated_cnn/` | CNN predictions per frame per fold |
| `loocv_summary_CSO.csv` | `results/CSO/gated_cnn/` | MAE, RMSE, R² per LOOCV fold |

---

## 8. Troubleshooting

**`FileNotFoundError` on a video or labels file**  
→ Check that `DATASET_NAME` matches your folder name exactly (case-sensitive on Linux/Mac).

**`KeyError: 'xps_value'` when loading labels**  
→ Check your CSV column names. They must be exactly `sample_id` and `xps_value`.

**Global crop green box misses the diffraction region**  
→ Lower `PERCENTILE` (try 95) or increase `PADDING` (try 30) in `global_crop.py`.

**K-Means clusters don't separate by stoichiometry**  
→ Try different `K` values in `combine_and_pca.py`. Use `plot_pca_and_kmeans.py` to visually inspect which K gives meaningful clusters.

**CNN training gives poor R²**  
→ Review `SELECTED_CLUSTERS`. Remove clusters where centroid images show blurry or noisy patterns. Only keep clusters with sharp diffraction streaks.

**Out of memory during CNN training**  
→ Reduce `BATCH_SIZE` in `train_gated_cnn_final.py`.

---

*Pipeline developed in the [Your Lab Name] lab. For questions, contact [your email].*
