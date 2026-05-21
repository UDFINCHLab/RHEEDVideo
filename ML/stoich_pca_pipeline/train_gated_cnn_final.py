"""
Gated CNN cluster-filtered training.

Edit DATASET_NAME and TRAINING_CLUSTERS below to switch datasets
and control which clusters are used for training.

Set TRAINING_CLUSTERS = [] to train on ALL frames (no cluster filtering).
Set TRAINING_CLUSTERS = [1, 3, 5] to train only on those cluster labels.

Look at the centroid images from plot_kmeans.py to decide which clusters
represent clean diffraction patterns before setting TRAINING_CLUSTERS.

Results saved to results/{DATASET_NAME}/gated_cnn_cluster_filtered/
so different datasets never overwrite each other.

Run from project root:
    python ML\\stoich_pca_pipeline\\train_gated_cnn_final.py
"""

from pathlib import Path
import json
import re
import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# CLUSTER SELECTION
# -----------------------------
# Set TRAINING_CLUSTERS to a list of cluster numbers (1-indexed) to train
# only on frames from those clusters. Look at the centroid images produced
# by plot_kmeans.py to decide which clusters represent clean diffraction patterns.
#
# Set TRAINING_CLUSTERS = [] to use ALL frames from every video (no filtering).
#
# Examples:
#   TRAINING_CLUSTERS = []         -> train on all frames
#   TRAINING_CLUSTERS = [1, 2, 4]  -> train only on clusters 1, 2 and 4
TRAINING_CLUSTERS = []   # start with all clusters, update after inspecting centroids


# -----------------------------
# PATHS — derived automatically from DATASET_NAME
# -----------------------------
COMBINED_H5 = PROJECT_ROOT / "results" / DATASET_NAME / f"combined_{DATASET_NAME}.h5"
OUTPUT_DIR  = PROJECT_ROOT / "results" / DATASET_NAME / "gated_cnn_cluster_filtered"
MODELS_DIR  = OUTPUT_DIR / "models"


# ── Training settings ─────────────────────────────────────────────────
# K_USED:             set to -1 to auto-read from H5, or set manually
# DOWNSAMPLE:         spatial downscale factor (4 = 75% smaller)
# MIN_CLUSTER_FRAMES: if filtered frames < this, fall back to last 20% of video
K_USED             = -1
DOWNSAMPLE         = 4
EPOCHS             = 50
BATCH_SIZE         = 16
LR                 = 1e-3
START_CHANNELS     = 8
KERNEL_SIZE        = 7
DROPOUT            = 0.15
RANDOM_SEED        = 42
MIN_CLUSTER_FRAMES = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
class Conv2dGLU2(nn.Module):
    """
    Gated convolutional layer using a Gated Linear Unit (GLU) mechanism.
    Splits the Conv2d output in half along the channel dimension —
    one half is activated with LeakyReLU, the other with Sigmoid (the gate).
    The element-wise product acts as a learned spatial attention mask,
    suppressing uninformative pixel regions before the next layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv      = nn.Conv2d(in_channels, out_channels * 2, kernel_size,
                                   stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels * 2)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        conv_out  = self.conv(x)
        conv_out  = self.batchnorm(conv_out)
        out, gate = torch.chunk(conv_out, 2, dim=1)
        return F.leaky_relu(out) * self.sigmoid(gate)


class GatedCNN2(nn.Module):
    """
    Gated CNN for regression of a target value from RHEED frame images.
    Architecture: 3 x (Conv2dGLU2 -> MaxPool) -> AdaptiveMaxPool -> Flatten
                  -> GLU -> Dropout -> GLU -> Linear(1)
    Output is a single scalar in [0, 1] (target / target_scale).
    """
    def __init__(self, start_channels=8, kernel_size=7, dropout_pct=0.15):
        super().__init__()
        self.conv1      = Conv2dGLU2(1, start_channels, kernel_size, 1)
        self.conv2      = Conv2dGLU2(start_channels, start_channels * 4, kernel_size, 1)
        self.conv3      = Conv2dGLU2(start_channels * 4, start_channels * 8, kernel_size, 1)
        self.pool       = nn.MaxPool2d(2)
        self.final_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten    = nn.Flatten(start_dim=1)
        self.fc         = nn.Sequential(
            nn.GLU(dim=-1),
            nn.Dropout(dropout_pct),
            nn.GLU(dim=-1),
            nn.Linear(start_channels * 2, 1),
        )

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(self.final_pool(x))
        return self.fc(x)


AUG_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),
])


class FrameDataset(Dataset):
    """
    PyTorch Dataset wrapping RHEED frame tensors and targets.
    Applies log normalization and min-max scaling per frame at load time.
    When augment=True, applies random flips, rotations, and small target
    jitter to reduce overfitting on limited datasets.
    """
    def __init__(self, frames, targets, augment=True):
        self.frames  = frames
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.frames[idx]
        x = torch.log(x + 1.0)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y = self.targets[idx]
        if self.augment:
            x = AUG_TRANSFORMS(x)
            y = y + torch.randn_like(y) * 0.015
        return x, y


# -----------------------------
# DATA LOADING
# -----------------------------
def load_all_data():
    """
    Load all videos' frames plus cluster labels and training filter indices.

    If TRAINING_CLUSTERS is empty, all frames are used for training.
    If TRAINING_CLUSTERS is set, only frames from those clusters are used,
    with a fallback to the last 20% of a video if too few filtered frames exist.
    Auto-detects K from the H5 file if K_USED = -1.
    """
    with h5.File(COMBINED_H5, "r") as f:
        image_data = f["data/image_data"][:]

        if K_USED == -1:
            kmeans_keys = list(f["kmeans"].keys())
            k_key       = kmeans_keys[-1]
            k_actual    = int(k_key.split("=")[1])
            print(f"Auto-detected k = {k_actual} from H5 file")
        else:
            k_actual = K_USED
            k_key    = f"k={k_actual}"

        labels        = f[f"kmeans/{k_key}/labels"][:]
        fmap          = f["data/frame_map"]
        sample_ids    = [s.decode() if isinstance(s, bytes) else s
                         for s in fmap["sample_ids"][:]]
        start_frames  = fmap["start_frames"][:]
        end_frames    = fmap["end_frames"][:]
        target_values = fmap["target_values"][:]

    target_scale     = float(np.max(np.abs(target_values))) * 1.1
    if target_scale  < 1e-6:
        target_scale = 1.0

    use_all_clusters = (len(TRAINING_CLUSTERS) == 0)
    filter_label     = "ALL frames" if use_all_clusters else f"clusters {TRAINING_CLUSTERS}"

    H, W, total = image_data.shape
    print(f"Dataset:      {DATASET_NAME}")
    print(f"Frames:       ({H}, {W}, {total})")
    print(f"Training on:  {filter_label}  (k={k_actual})")
    print(f"Target range: {target_values.min():.4f} - {target_values.max():.4f}")
    print(f"Target scale: {target_scale:.4f}\n")

    all_frames_by_video         = []
    all_clusters_by_video       = []
    train_local_idx_by_video    = []
    targets_by_video            = []
    total_video_frames_by_video = []
    fallback_count              = 0

    for i, sid in enumerate(sample_ids):
        start   = int(start_frames[i])
        end     = int(end_frames[i])
        n_video = end - start + 1

        video_frames_full = image_data[:, :, start:end + 1]
        video_frames_full = np.transpose(video_frames_full, (2, 0, 1)).astype(np.float32)
        video_labels      = labels[start:end + 1] + 1  # 1-indexed

        if use_all_clusters:
            target_local_idx = np.arange(n_video)
            note = f"all {n_video} frames"
        else:
            in_filter        = np.array([lab in set(TRAINING_CLUSTERS)
                                         for lab in video_labels])
            target_local_idx = np.where(in_filter)[0]

            if len(target_local_idx) < MIN_CLUSTER_FRAMES:
                fallback_start   = int(n_video * 0.80)
                target_local_idx = np.arange(fallback_start, n_video)
                fallback_count  += 1
                note = f"FALLBACK last 20%: {len(target_local_idx)} frames"
            else:
                note = (f"{len(target_local_idx)}/{n_video} frames "
                        f"in selected clusters")

        all_frames_by_video.append(torch.from_numpy(video_frames_full))
        all_clusters_by_video.append(video_labels)
        train_local_idx_by_video.append(target_local_idx)
        targets_by_video.append(float(target_values[i]))
        total_video_frames_by_video.append(n_video)

        print(f"[{i+1:2d}] {sid}: {note},  target={target_values[i]:.4f}")

    if not use_all_clusters:
        print(f"\nFallback used for {fallback_count}/{len(sample_ids)} videos")

    return (all_frames_by_video, targets_by_video, sample_ids,
            all_clusters_by_video, train_local_idx_by_video,
            total_video_frames_by_video, target_scale, k_actual,
            use_all_clusters)


# -----------------------------
# HELPERS
# -----------------------------
def resize_frames(frames_tensor):
    """
    Spatially downsample all frames by DOWNSAMPLE factor.
    Returns tensor of shape (n_frames, 1, H//DOWNSAMPLE, W//DOWNSAMPLE).
    """
    H      = frames_tensor.shape[1]
    W      = frames_tensor.shape[2]
    resize = transforms.Resize((H // DOWNSAMPLE, W // DOWNSAMPLE), antialias=True)
    return torch.stack([resize(f) for f in frames_tensor.unsqueeze(1)])


def parse_sample_number(sample_id):
    """
    Extract a numeric identifier from the sample ID.
    CSO_007 -> 7,  STO N=-5_0 -> -5.  Returns None if not found.
    """
    m = re.search(r"CSO_(\d+)", sample_id)
    if m:
        return int(m.group(1))
    m = re.search(r"N=(-?\d+)_", sample_id)
    if m:
        return int(m.group(1))
    return None


def train_model(train_frames, train_targets, training_log, fold_idx):
    """
    Train a GatedCNN2 model for EPOCHS epochs with Adam + L1 loss.
    Appends per-epoch loss to training_log. Returns the trained model.
    """
    model     = GatedCNN2(start_channels=START_CHANNELS, kernel_size=KERNEL_SIZE,
                          dropout_pct=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    loader    = DataLoader(FrameDataset(train_frames, train_targets, augment=True),
                           batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        training_log.append({"fold": fold_idx, "epoch": epoch + 1,
                              "train_loss": avg_loss})
        if (epoch + 1) % 10 == 0:
            print(f"    fold {fold_idx}  epoch {epoch+1:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.6f}")
    return model


def predict_per_frame(model, video_frames, target_scale):
    """
    Run inference on all frames without augmentation.
    Rescales predictions back to original target units.
    Returns 1D numpy array of predicted values.
    """
    model.eval()
    loader = DataLoader(
        FrameDataset(video_frames, torch.zeros(len(video_frames), 1), augment=False),
        batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(preds) * target_scale


# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Run the full LOOCV training pipeline:
        1. Load all videos, determine training frame filter.
        2. Downsample frames, build per-video training subsets.
        3. For each fold: train on N-1 videos, predict all frames of held-out
           video, log per-frame predictions with cluster membership flags.
        4. Train a final model on all videos.
        5. Save all outputs to results/{DATASET_NAME}/gated_cnn_cluster_filtered/
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    filter_label = ("ALL clusters" if not TRAINING_CLUSTERS
                    else f"clusters {TRAINING_CLUSTERS}")
    print(f"Dataset:  {DATASET_NAME}")
    print(f"Device:   {DEVICE}")
    print(f"Training: {filter_label}")
    print(f"Epochs:   {EPOCHS},  batch: {BATCH_SIZE},  channels: {START_CHANNELS}\n")

    (all_frames_by_video, targets_by_video, sample_ids,
     all_clusters_by_video, train_local_idx_by_video,
     total_video_frames_by_video, target_scale,
     k_actual, use_all_clusters) = load_all_data()

    n_videos = len(all_frames_by_video)

    print("\nResizing frames...")
    all_frames_by_video = [resize_frames(f) for f in all_frames_by_video]
    print(f"Resized to: {list(all_frames_by_video[0].shape[2:4])}")

    train_frames_by_video  = []
    train_targets_by_video = []
    for i in range(n_videos):
        tr_idx    = train_local_idx_by_video[i]
        tr_frames = all_frames_by_video[i][tr_idx]
        tr_tgts   = torch.full((len(tr_idx), 1),
                               targets_by_video[i] / target_scale,
                               dtype=torch.float32)
        train_frames_by_video.append(tr_frames)
        train_targets_by_video.append(tr_tgts)

    print("\n" + "=" * 60)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print("=" * 60)

    predictions     = np.zeros(n_videos)
    prediction_stds = np.zeros(n_videos)
    actuals         = np.array(targets_by_video)
    training_log    = []
    per_frame_rows  = []

    for test_idx in range(n_videos):
        test_sid = sample_ids[test_idx]
        print(f"\n[fold {test_idx+1}/{n_videos}] test: {test_sid} "
              f"(target={actuals[test_idx]:.4f})")

        train_frames  = torch.cat(
            [train_frames_by_video[j]  for j in range(n_videos) if j != test_idx], dim=0)
        train_targets = torch.cat(
            [train_targets_by_video[j] for j in range(n_videos) if j != test_idx], dim=0)

        print(f"  train: {train_frames.shape[0]} frames  "
              f"test-all: {all_frames_by_video[test_idx].shape[0]}  "
              f"test-filter: {train_local_idx_by_video[test_idx].size}")

        model = train_model(train_frames, train_targets, training_log, test_idx + 1)
        torch.save(model.state_dict(), MODELS_DIR / f"fold_{test_idx+1:02d}.pth")

        all_preds    = predict_per_frame(model, all_frames_by_video[test_idx],
                                         target_scale)
        filter_idx   = train_local_idx_by_video[test_idx]
        filter_preds = all_preds[filter_idx]

        predictions[test_idx]     = float(np.mean(filter_preds))
        prediction_stds[test_idx] = float(np.std(filter_preds))

        all_clusters = all_clusters_by_video[test_idx]
        filter_set   = set(filter_idx.tolist())
        total_v      = total_video_frames_by_video[test_idx]

        for j, pred in enumerate(all_preds):
            per_frame_rows.append({
                "sample_id":          test_sid,
                "sample_number":      parse_sample_number(test_sid),
                "frame_idx_in_video": j,
                "frame_pct_in_video": round(100.0 * j / total_v, 2),
                "cluster":            int(all_clusters[j]),
                "in_training_filter": j in filter_set,
                "actual_xps":         float(actuals[test_idx]),
                "predicted_xps":      float(pred),
            })

        abs_err = abs(predictions[test_idx] - actuals[test_idx])
        print(f"  >> actual: {actuals[test_idx]:.4f}  "
              f"predicted: {predictions[test_idx]:.4f}  "
              f"abs_err: {abs_err:.4f}")

    # Final model
    print("\n" + "=" * 60)
    print(f"TRAINING FINAL MODEL ON ALL {n_videos} VIDEOS")
    print("=" * 60)
    all_train_frames  = torch.cat(train_frames_by_video, dim=0)
    all_train_targets = torch.cat(train_targets_by_video, dim=0)
    print(f"Total training frames: {all_train_frames.shape[0]}")
    final_model = train_model(all_train_frames, all_train_targets, [], fold_idx=0)
    final_path  = MODELS_DIR / "final_model.pth"
    torch.save(final_model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    # Metrics
    mae       = mean_absolute_error(actuals, predictions)
    rmse      = np.sqrt(mean_squared_error(actuals, predictions))
    r2        = r2_score(actuals, predictions)
    base_pred = np.full_like(actuals, actuals.mean())
    base_mae  = mean_absolute_error(actuals, base_pred)
    base_rmse = np.sqrt(mean_squared_error(actuals, base_pred))

    # Per-video CSV
    df = pd.DataFrame({
        "sample_id":                 sample_ids,
        "sample_number":             [parse_sample_number(s) for s in sample_ids],
        "n_train_frames_in_filter":  [len(tr) for tr in train_local_idx_by_video],
        "actual_target":             actuals,
        "predicted_target":          predictions,
        "prediction_std":            prediction_stds,
        "abs_error":                 np.abs(predictions - actuals),
        "squared_error":             (predictions - actuals) ** 2,
        # keep these column names for compatibility with plot_per_frame_predictions.py
        "actual_xps":                actuals,
        "predicted_xps":             predictions,
    })
    df = df.sort_values("actual_target").reset_index(drop=True)
    pred_csv = OUTPUT_DIR / "gated_cnn_cluster_filtered_predictions.csv"
    df.to_csv(pred_csv, index=False)
    print(f"\nSaved per-video: {pred_csv}")

    pf_df  = pd.DataFrame(per_frame_rows)
    pf_csv = OUTPUT_DIR / "gated_cnn_cluster_filtered_per_frame.csv"
    pf_df.to_csv(pf_csv, index=False)
    print(f"Saved per-frame: {pf_csv}  ({len(pf_df)} rows)")

    pd.DataFrame(training_log).to_csv(
        OUTPUT_DIR / "gated_cnn_cluster_filtered_training_log.csv", index=False)

    metrics = {
        "dataset":           DATASET_NAME,
        "model":             "gated_cnn_loocv",
        "training_clusters": TRAINING_CLUSTERS if TRAINING_CLUSTERS else "all",
        "k_used":            k_actual,
        "target_scale":      target_scale,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "start_channels":    START_CHANNELS,
        "device":            DEVICE,
        "mae":               mae,
        "rmse":              rmse,
        "r2":                r2,
        "baseline_mae":      base_mae,
        "baseline_rmse":     base_rmse,
    }
    with open(OUTPUT_DIR / "gated_cnn_cluster_filtered_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[{DATASET_NAME}] RESULTS — {filter_label}, k={k_actual}")
    print("=" * 70)
    print(f"{'sample_id':<18}{'#':>6}{'frames':>8}{'actual':>12}"
          f"{'predicted':>12}{'err':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['sample_id']:<18}{str(row['sample_number']):>6}"
              f"{row['n_train_frames_in_filter']:>8}"
              f"{row['actual_target']:>12.4f}"
              f"{row['predicted_target']:>12.4f}"
              f"{row['abs_error']:>10.4f}")

    print("\n" + "=" * 70)
    print(f"{'Model':<35}{'MAE':>10}{'RMSE':>10}{'R^2':>10}")
    print("-" * 70)
    print(f"{'Baseline (predict mean)':<35}"
          f"{base_mae:>10.4f}{base_rmse:>10.4f}{0.0:>10.3f}")
    print(f"{'Gated CNN':<35}{mae:>10.4f}{rmse:>10.4f}{r2:>10.3f}")
    print("=" * 70)
    print(f"\n{n_videos} fold models: {MODELS_DIR}")
    print(f"Final model:     {final_path}")


if __name__ == "__main__":
    main()