"""
Cluster-vs-video plot with REAL TIME (seconds) on the x-axis.

Shows BOTH the video labels (top x-axis) and continuous time numbers
(bottom x-axis) so you can see the actual elapsed seconds across the
combined timeline.

Edit DATASET_NAME below to switch datasets.

Run from project root:
    python ML\\stoich_pca_pipeline\\plot_clusters_by_video.py
"""

from pathlib import Path
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS — derived automatically from DATASET_NAME
# -----------------------------
COMBINED_H5 = PROJECT_ROOT / "results" / DATASET_NAME / f"combined_{DATASET_NAME}.h5"
OUTPUT_DIR  = PROJECT_ROOT / "results" / DATASET_NAME / "cluster_by_video_plots"


# ── Settings ──────────────────────────────────────────────────────────
# K_TO_PLOT: set to -1 to auto-read the k that was saved in the H5 file,
#            or set to a specific integer to force a particular k.
K_TO_PLOT = -1


def load_everything(k):
    """
    Load cluster labels, timestamps, frame mapping, and target values
    from combined_{DATASET_NAME}.h5 for a given k value.

    Returns: labels, times, sample_ids, start_frames, end_frames, target_values
    """
    with h5.File(COMBINED_H5, "r") as f:
        # Auto-detect k if not specified
        if k == -1:
            kmeans_keys = list(f["kmeans"].keys())
            k_key = kmeans_keys[-1]
            k = int(k_key.split("=")[1])
            print(f"Auto-detected k = {k} from H5 file")
        else:
            k_key = f"k={k}"

        labels       = f[f"kmeans/{k_key}/labels"][:]
        times        = f["data/times"][:]
        fmap         = f["data/frame_map"]
        sample_ids   = [s.decode() if isinstance(s, bytes) else s
                        for s in fmap["sample_ids"][:]]
        start_frames = fmap["start_frames"][:]
        end_frames   = fmap["end_frames"][:]
        target_values = fmap["target_values"][:]

    return labels, times, sample_ids, start_frames, end_frames, target_values, k


def build_continuous_time(times, start_frames, end_frames):
    """Offset each video's times so they concatenate into one timeline."""
    cont   = np.copy(times).astype(np.float64)
    offset = 0.0
    for i in range(len(start_frames)):
        s   = int(start_frames[i])
        e   = int(end_frames[i])
        seg = cont[s:e + 1]
        if len(seg) == 0:
            continue
        seg_zeroed    = seg - seg[0]
        cont[s:e + 1] = seg_zeroed + offset
        step          = seg_zeroed[1] - seg_zeroed[0] if len(seg_zeroed) > 1 else 1.0
        offset        = cont[e] + step
    return cont


def plot_clusters_vs_time(time_axis, labels, sample_ids, start_frames, end_frames,
                          target_values, k, title, save_path):
    """
    Plot with TWO x-axes:
      - bottom: continuous seconds (numbers)
      - top:    video labels (sample_id + target value)
    """
    fig, ax = plt.subplots(figsize=(22, 7))

    cmap = plt.get_cmap("tab20", k)
    ax.scatter(time_axis, labels + 1, c=labels, s=3, cmap=cmap, edgecolors="none")

    for i in range(1, len(sample_ids)):
        boundary_time = time_axis[int(start_frames[i])]
        ax.axvline(boundary_time, color="red", linewidth=0.7, alpha=0.6)

    total_time = time_axis[-1]
    ax.set_xlim(time_axis[0], total_time)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    ax.set_xlabel(f"Time (seconds, continuous across videos)  --  total = {total_time:.0f}s",
                  fontweight="bold", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    label_times   = []
    label_strings = []
    for i in range(len(sample_ids)):
        s     = int(start_frames[i])
        e     = int(end_frames[i])
        mid_t = (time_axis[s] + time_axis[e]) / 2.0
        label_times.append(mid_t)
        label_strings.append(f"{sample_ids[i]}\n{target_values[i]:.4f}")

    ax_top.set_xticks(label_times)
    ax_top.set_xticklabels(label_strings, rotation=90, fontsize=7)
    ax_top.tick_params(axis="x", which="major", pad=2)

    ax.set_yticks(range(1, k + 1))
    ax.set_ylabel("Cluster", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=70)
    ax.set_ylim(0.3, k + 0.7)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """
    Generate three cluster visualization plots for K_TO_PLOT:
        1. Cluster vs. continuous time — videos in original load order
        2. Cluster vs. continuous time — videos sorted by target value
        3. Cluster composition heatmap — fraction of frames per cluster per video

    All plots are saved to results/{DATASET_NAME}/cluster_by_video_plots/
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Reading: {COMBINED_H5}\n")

    labels, times, sample_ids, start_frames, end_frames, target_values, k = \
        load_everything(K_TO_PLOT)

    print(f"Frames: {len(labels)},  Videos: {len(sample_ids)},  k = {k}")

    cont_time = build_continuous_time(times, start_frames, end_frames)
    print(f"Continuous timeline: 0 to {cont_time[-1]:.1f} seconds")

    # Plot 1: original load order
    plot_clusters_vs_time(
        cont_time, labels, sample_ids, start_frames, end_frames, target_values,
        k,
        f"[{DATASET_NAME}] K-Means clusters over time, k={k}  (original load order)",
        OUTPUT_DIR / "clusters_by_video_original_time.png",
    )

    # Plot 2: sorted by target value
    sort_idx         = np.argsort(target_values)
    new_labels       = []
    new_times        = []
    new_sample_ids   = []
    new_start_frames = []
    new_end_frames   = []
    new_targets      = []
    cursor_idx       = 0
    cursor_time      = 0.0

    for si in sort_idx:
        s          = int(start_frames[si])
        e          = int(end_frames[si])
        seg_labels = labels[s:e + 1]
        seg_time   = cont_time[s:e + 1]
        if len(seg_time) == 0:
            continue
        seg_time_z   = seg_time - seg_time[0]
        seg_time_off = seg_time_z + cursor_time

        new_labels.append(seg_labels)
        new_times.append(seg_time_off)
        new_sample_ids.append(sample_ids[si])
        new_start_frames.append(cursor_idx)
        new_end_frames.append(cursor_idx + len(seg_labels) - 1)
        new_targets.append(float(target_values[si]))

        cursor_idx  += len(seg_labels)
        cursor_time  = seg_time_off[-1] + 1.0

    new_labels       = np.concatenate(new_labels)
    new_times        = np.concatenate(new_times)
    new_start_frames = np.array(new_start_frames)
    new_end_frames   = np.array(new_end_frames)
    new_targets      = np.array(new_targets)

    plot_clusters_vs_time(
        new_times, new_labels, new_sample_ids, new_start_frames, new_end_frames,
        new_targets, k,
        f"[{DATASET_NAME}] K-Means clusters over time, k={k}  (sorted by target value)",
        OUTPUT_DIR / "clusters_by_video_sorted_by_target_time.png",
    )

    # Plot 3: heatmap
    n_videos = len(sample_ids)
    comp     = np.zeros((n_videos, k))
    for i in range(n_videos):
        s   = int(start_frames[i])
        e   = int(end_frames[i])
        seg = labels[s:e + 1]
        for c in range(k):
            comp[i, c] = np.mean(seg == c)

    comp_sorted = comp[sort_idx]
    sorted_ids  = [sample_ids[i] for i in sort_idx]
    sorted_tgts = target_values[sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n_videos // 2)))
    im = ax.imshow(comp_sorted, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{c+1}" for c in range(k)])
    ax.set_yticks(range(n_videos))
    ax.set_yticklabels([f"{sid}  ({tgt:.4f})"
                        for sid, tgt in zip(sorted_ids, sorted_tgts)], fontsize=8)
    ax.set_xlabel("Cluster", fontweight="bold")
    ax.set_ylabel(f"Video (sorted by target value)", fontweight="bold")
    ax.set_title(f"[{DATASET_NAME}] Cluster composition heatmap, k={k}", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Fraction of frames")
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "cluster_composition_heatmap.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()