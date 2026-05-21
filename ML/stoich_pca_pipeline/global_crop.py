"""
Script 1 of the stoich PCA pipeline (FIXED v2).

Loops over all RHEED videos for a given dataset, samples frames, detects
the bright diffraction region using a percentile-based threshold, filters
out tiny noise specks, and rejects any bbox covering more than half the frame.

Edit DATASET_NAME below to switch datasets.
Saves the crop to global_crop_{DATASET_NAME}.txt so each dataset has its own
crop file and results never overwrite each other.
"""

import random
from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # "STO" or "CSO"


# -----------------------------
# PATHS — all derived from DATASET_NAME automatically
# -----------------------------
VIDEOS_DIR = PROJECT_ROOT / "stoich_dataset" / DATASET_NAME / "videos_raw"
OUTPUT_DIR = Path(__file__).resolve().parent
DEBUG_DIR  = OUTPUT_DIR / f"debug_global_crop_{DATASET_NAME}"


# ── Detection settings ────────────────────────────────────────────────
# FRAMES_PER_VIDEO: frames sampled per video for bbox detection
#                   (skips first 10% to avoid blank startup frames)
# PERCENTILE:       brightness threshold — top 5% of pixels = diffraction
# MIN_CONTOUR_AREA: minimum px² to count as a real diffraction feature
# MAX_BBOX_FRAC:    reject detections covering more than 50% of the frame
# PADDING:          pixels added around the union bbox for safety margin
FRAMES_PER_VIDEO = 8
PERCENTILE       = 95
MIN_CONTOUR_AREA = 200
MAX_BBOX_FRAC    = 0.50
PADDING          = 30
RANDOM_SEED      = 42


def detect_bright_region(frame, frame_height, frame_width):
    """
    Detect the diffraction region in a single frame using a percentile threshold.
    Applies Gaussian blur to merge nearby bright pixels, finds contours,
    filters out hot pixels by minimum area, and returns the union bounding box.
    Returns None if no valid region is found or the bbox covers too much of the frame.

    Returns: (x, y, w, h) bounding box or None
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    threshold = np.percentile(gray, PERCENTILE)
    if threshold < 20:
        return None

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not valid:
        return None

    all_points = np.vstack(valid)
    x, y, w, h = cv2.boundingRect(all_points)

    if (w * h) / (frame_height * frame_width) > MAX_BBOX_FRAC:
        return None

    return (x, y, w, h)


def sample_video_bboxes(video_path):
    """
    Sample FRAMES_PER_VIDEO frames from a video (skipping the first 10%)
    and run detect_bright_region on each to collect bounding boxes.
    Also captures the first valid frame as a reference image for debug output.

    Returns: (bboxes, ref_frame, video_width, video_height)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_samples    = min(FRAMES_PER_VIDEO, total_frames)
    skip_start   = int(total_frames * 0.10)
    valid_range  = list(range(skip_start, total_frames))
    frame_indices = sorted(random.sample(valid_range, n_samples))

    bboxes    = []
    ref_frame = None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = detect_bright_region(frame, height, width)
        if bbox is not None:
            bboxes.append(bbox)
            if ref_frame is None:
                ref_frame = frame.copy()

    cap.release()
    return bboxes, ref_frame, width, height


def compute_global_crop(all_bboxes, video_height, video_width):
    """
    Compute the union bounding box across all collected bboxes from all videos,
    then expand by PADDING pixels on each side and clamp to frame boundaries.
    This single crop region is applied uniformly to all videos in this dataset.

    Returns: (top, bottom, left, right)
    """
    x_min = min(b[0] for b in all_bboxes)
    y_min = min(b[1] for b in all_bboxes)
    x_max = max(b[0] + b[2] for b in all_bboxes)
    y_max = max(b[1] + b[3] for b in all_bboxes)

    top    = max(0, y_min - PADDING)
    bottom = min(video_height, y_max + PADDING)
    left   = max(0, x_min - PADDING)
    right  = min(video_width, x_max + PADDING)
    return top, bottom, left, right


def draw_debug_image(ref_frame, crop, sample_id, save_path):
    """
    Draw the global crop rectangle on a reference frame and save as PNG.
    Used to visually verify the detected crop contains the diffraction streaks
    for every video before running batch_preprocess.py.
    """
    top, bottom, left, right = crop
    debug = ref_frame.copy()
    cv2.rectangle(debug, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(debug, sample_id, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(save_path), debug)


def main():
    """
    Loop over all videos in stoich_dataset/{DATASET_NAME}/videos_raw/,
    collect diffraction bounding boxes from sampled frames,
    compute the union global crop, save it to global_crop_{DATASET_NAME}.txt,
    and generate debug overlay images for manual verification.

    Run this script once per dataset before batch_preprocess.py.
    Always verify the debug images in debug_global_crop_{DATASET_NAME}/ before proceeding.
    """
    random.seed(RANDOM_SEED)

    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Videos folder not found: {VIDEOS_DIR}")

    video_files = sorted(VIDEOS_DIR.glob("*.avi"))
    if not video_files:
        raise FileNotFoundError(f"No .avi files in {VIDEOS_DIR}")

    print(f"\nDataset: {DATASET_NAME}")
    print(f"Found {len(video_files)} videos in {VIDEOS_DIR}\n")
    print(f"Settings: percentile={PERCENTILE}, min_area={MIN_CONTOUR_AREA}, "
          f"max_bbox_frac={MAX_BBOX_FRAC}, padding={PADDING}\n")

    DEBUG_DIR.mkdir(exist_ok=True)
    for f in DEBUG_DIR.glob("*.png"):
        f.unlink()

    all_bboxes     = []
    per_video_data = []
    video_width    = None
    video_height   = None

    for i, video_path in enumerate(video_files, start=1):
        sample_id = video_path.stem
        print(f"[{i:2d}/{len(video_files)}] {sample_id}", end=" ... ")

        try:
            bboxes, ref_frame, w, h = sample_video_bboxes(video_path)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        if not bboxes:
            print("no valid bright region detected, skipping")
            continue

        if video_width is None:
            video_width, video_height = w, h

        all_bboxes.extend(bboxes)
        per_video_data.append((sample_id, ref_frame, bboxes))

        avg_w = sum(b[2] for b in bboxes) / len(bboxes)
        avg_h = sum(b[3] for b in bboxes) / len(bboxes)
        print(f"got {len(bboxes)} bboxes (avg {avg_w:.0f}x{avg_h:.0f})")

    if not all_bboxes:
        raise RuntimeError("No bounding boxes collected from any video.")

    print(f"\nTotal bboxes: {len(all_bboxes)}")
    print(f"Video dimensions: {video_width} x {video_height}")

    crop = compute_global_crop(all_bboxes, video_height, video_width)
    top, bottom, left, right = crop

    print("\n" + "=" * 50)
    print(f"GLOBAL CROP — {DATASET_NAME}")
    print("=" * 50)
    print(f"  top    = {top}")
    print(f"  bottom = {bottom}")
    print(f"  left   = {left}")
    print(f"  right  = {right}")
    print(f"  size:  {bottom - top} x {right - left}")
    print(f"  frac:  {((bottom-top)*(right-left))/(video_height*video_width):.1%}")
    print("=" * 50)

    # Save to dataset-specific crop file so STO and CSO never overwrite each other
    crop_file = OUTPUT_DIR / f"global_crop_{DATASET_NAME}.txt"
    with open(crop_file, "w") as f:
        f.write(f"# Global crop computed across all {DATASET_NAME} videos\n")
        f.write("# Format: [top, bottom, left, right]\n")
        f.write(f"GLOBAL_CROP = [{top}, {bottom}, {left}, {right}]\n")
    print(f"\nSaved crop to: {crop_file}")

    print(f"\nGenerating debug images in {DEBUG_DIR}...")
    for sample_id, ref_frame, bboxes in per_video_data:
        draw_debug_image(ref_frame, crop, sample_id, DEBUG_DIR / f"{sample_id}.png")
    print(f"Saved {len(per_video_data)} debug images.")
    print(f"\nNext: verify green rectangle contains streaks in EVERY debug image.")
    print(f"Then run batch_preprocess.py with DATASET_NAME = '{DATASET_NAME}'\n")


if __name__ == "__main__":
    main()