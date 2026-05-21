"""
predict_new_video.py
────────────────────
Predict stoichiometry of a NEW video using the best saved model from
a previously trained material pipeline. No retraining needed.

Usage:
    1. Set DATASET_NAME to the material the model was trained on (e.g. "CSO")
    2. Set NEW_VIDEO_PATH to the full path of your new .avi video
    3. Run:
           python ML/stoich_pca_pipeline/predict_new_video.py

Works for any material as long as:
    - That material's pipeline has already been run (global_crop, batch_preprocess,
      combine_and_pca, train_gated_cnn_final)
    - Saved models exist in results/<DATASET_NAME>/gated_cnn/models/
    - The new video is the same type/material as the training data
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py as h5
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
DATASET_NAME   = "CSO"           # Material the model was trained on
NEW_VIDEO_PATH = r"C:\path\to\your\new_video.avi"   # Full path to new video
# ───────────────────────────────────────────────────────────────────────────────

# ─── Paths (auto-built — do not edit) ──────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR   = PROJECT_ROOT / "results" / DATASET_NAME
MODELS_DIR    = RESULTS_DIR / "gated_cnn" / "models"
LOOCV_CSV     = RESULTS_DIR / "gated_cnn" / f"loocv_summary_{DATASET_NAME}.csv"
COMBINED_H5   = RESULTS_DIR / f"combined_{DATASET_NAME}.h5"
CROP_TXT      = PROJECT_ROOT / "ML" / "stoich_pca_pipeline" / f"global_crop_{DATASET_NAME}.txt"
OUTPUT_DIR    = RESULTS_DIR / "gated_cnn" / "new_video_predictions"
# ───────────────────────────────────────────────────────────────────────────────

# ─── Preprocessing settings — must match what was used in batch_preprocess.py ──
DOWNSAMPLE     = 2       # Spatial downsampling factor
FRAME_PERIOD   = 1       # Use every Nth frame (1 = all frames)
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# CNN Architecture — must exactly match train_gated_cnn_final.py
# ══════════════════════════════════════════════════════════════════════════════

class SpatialAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class GatedCNN2(nn.Module):
    def __init__(self, start_channels=8, kernel_size=5, dropout_pct=0.3):
        super().__init__()
        c = start_channels
        k = kernel_size
        p = k // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c,    k, padding=p), nn.BatchNorm2d(c),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, c*2,  k, padding=p), nn.BatchNorm2d(c*2),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c*2, c*4,k, padding=p), nn.BatchNorm2d(c*4),  nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gate       = SpatialAttentionGate(c * 4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout_pct),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.gate(x)
        x = self.global_pool(x)
        return self.regressor(x)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames  # shape (N, 1, H, W) float32 tensor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def load_crop(crop_txt_path):
    """Read global_crop_<MATERIAL>.txt and return [top, bottom, left, right]."""
    if not crop_txt_path.exists():
        raise FileNotFoundError(
            f"Crop file not found: {crop_txt_path}\n"
            f"Make sure global_crop.py has been run for {DATASET_NAME}."
        )
    with open(crop_txt_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GLOBAL_CROP"):
                # parse: GLOBAL_CROP = [85, 480, 120, 600]
                vals = line.split("=")[1].strip().strip("[]").split(",")
                return [int(v.strip()) for v in vals]
    raise ValueError(f"Could not parse GLOBAL_CROP from {crop_txt_path}")


def preprocess_video(video_path, crop):
    """
    Read .avi video, apply crop and downsampling.
    Returns tensor of shape (N_frames, 1, H, W) float32.
    """
    top, bottom, left, right = crop
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_PERIOD == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cropped = gray[top:bottom, left:right]
            if DOWNSAMPLE > 1:
                h, w = cropped.shape
                cropped = cv2.resize(cropped,
                                     (w // DOWNSAMPLE, h // DOWNSAMPLE),
                                     interpolation=cv2.INTER_AREA)
            # Normalize per frame
            mean, std = cropped.mean(), cropped.std()
            if std > 1e-6:
                cropped = (cropped - mean) / std
            frames.append(cropped)
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    arr = np.stack(frames, axis=0)[:, np.newaxis, :, :]  # (N, 1, H, W)
    return torch.tensor(arr, dtype=torch.float32)


def find_best_model(models_dir, loocv_csv):
    """
    Find the fold model with the lowest MAE from the LOOCV summary CSV.
    Returns path to best model .pt file and the fold number.
    """
    if not loocv_csv.exists():
        raise FileNotFoundError(
            f"LOOCV summary not found: {loocv_csv}\n"
            f"Make sure train_gated_cnn_final.py has been run for {DATASET_NAME}."
        )

    df = pd.read_csv(loocv_csv)

    # Column names may vary slightly — find MAE column
    mae_col = None
    for col in df.columns:
        if "mae" in col.lower():
            mae_col = col
            break
    if mae_col is None:
        raise ValueError(f"No MAE column found in {loocv_csv}. Columns: {list(df.columns)}")

    best_row = df.loc[df[mae_col].idxmin()]

    # Find fold number
    fold_col = None
    for col in df.columns:
        if "fold" in col.lower():
            fold_col = col
            break

    if fold_col:
        best_fold = int(best_row[fold_col])
    else:
        best_fold = int(df[mae_col].idxmin()) + 1  # fallback: row index + 1

    best_mae = float(best_row[mae_col])

    # Find the model file
    model_path = models_dir / f"model_fold_{best_fold}.pt"
    if not model_path.exists():
        # Try to find any model file matching the fold
        candidates = list(models_dir.glob(f"*fold*{best_fold}*.pt"))
        if candidates:
            model_path = candidates[0]
        else:
            # Fall back to final model if available
            final = models_dir / "model_final.pt"
            if final.exists():
                print(f"  WARNING: fold {best_fold} model not found, using final model instead.")
                return final, "final"
            raise FileNotFoundError(
                f"No model file found for fold {best_fold} in {models_dir}"
            )

    return model_path, best_fold, best_mae


def load_model(model_path, device):
    """Load saved GatedCNN2 weights from .pt file."""
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both raw state_dict and wrapped checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict    = checkpoint["model_state_dict"]
        start_channels = checkpoint.get("start_channels", 8)
        kernel_size    = checkpoint.get("kernel_size", 5)
        dropout_pct    = checkpoint.get("dropout_pct", 0.3)
        target_scale   = checkpoint.get("target_scale", 1.0)
    else:
        state_dict    = checkpoint
        start_channels = 8
        kernel_size    = 5
        dropout_pct    = 0.3
        target_scale   = 1.0

    model = GatedCNN2(start_channels=start_channels,
                      kernel_size=kernel_size,
                      dropout_pct=dropout_pct).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, target_scale


def run_inference(model, frames_tensor, target_scale, device, batch_size=64):
    """
    Run forward pass on all frames.
    Returns numpy array of per-frame predictions in original units.
    """
    dataset = FrameDataset(frames_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds   = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(device)).cpu().numpy().flatten()
            preds.append(out)
    return np.concatenate(preds) * target_scale


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  Stoichiometry Prediction — New Video Inference")
    print(f"  Material model : {DATASET_NAME}")
    print(f"  New video      : {NEW_VIDEO_PATH}")
    print("=" * 60)

    # ── Validate inputs ──────────────────────────────────────────────
    video_path = Path(NEW_VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Models folder not found: {MODELS_DIR}\n"
            f"Run train_gated_cnn_final.py for {DATASET_NAME} first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Step 1: Load crop ────────────────────────────────────────────
    print(f"\n[1/4] Loading crop coordinates for {DATASET_NAME}...")
    crop = load_crop(CROP_TXT)
    print(f"      Crop: top={crop[0]}, bottom={crop[1]}, left={crop[2]}, right={crop[3]}")

    # ── Step 2: Preprocess new video ─────────────────────────────────
    print(f"\n[2/4] Preprocessing video...")
    frames = preprocess_video(video_path, crop)
    print(f"      Frames loaded: {len(frames)}  |  Frame shape: {frames.shape[2]}×{frames.shape[3]}")

    # ── Step 3: Find and load best model ─────────────────────────────
    print(f"\n[3/4] Finding best model from {DATASET_NAME} training...")
    model_path, best_fold, best_mae = find_best_model(MODELS_DIR, LOOCV_CSV)
    print(f"      Best fold : {best_fold}  (MAE = {best_mae:.4f})")
    print(f"      Model file: {model_path.name}")

    model, target_scale = load_model(model_path, device)
    print(f"      Model loaded successfully.")

    # ── Step 4: Run inference ────────────────────────────────────────
    print(f"\n[4/4] Running inference on {len(frames)} frames...")
    per_frame_preds = run_inference(model, frames, target_scale, device)

    mean_pred = float(np.mean(per_frame_preds))
    std_pred  = float(np.std(per_frame_preds))
    median_pred = float(np.median(per_frame_preds))

    # ── Save results ─────────────────────────────────────────────────
    video_name = video_path.stem
    out_csv = OUTPUT_DIR / f"prediction_{video_name}.csv"

    df_out = pd.DataFrame({
        "frame_index": np.arange(len(per_frame_preds)),
        "predicted_stoich": per_frame_preds
    })
    df_out.to_csv(out_csv, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Video          : {video_path.name}")
    print(f"  Material model : {DATASET_NAME}  (fold {best_fold}, MAE={best_mae:.4f})")
    print(f"  Frames used    : {len(per_frame_preds)}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Mean prediction  : {mean_pred:.4f}")
    print(f"  Median prediction: {median_pred:.4f}")
    print(f"  Std dev          : {std_pred:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Per-frame CSV saved to:")
    print(f"  {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
