"""
Script 2: batch preprocess RHEED videos for a given dataset.

Edit DATASET_NAME below to switch between datasets.
Loops over videos in stoich_dataset/{DATASET_NAME}/videos_raw/,
runs pre_processing on each with global crop, saves per-video H5 files
to stoich_dataset/{DATASET_NAME}/preprocessed_h5/.
"""

import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ML"
sys.path.insert(0, str(ML_DIR))

from pre_processing import pre_processing  # noqa: E402



# -----------------------------
# DATASET SELECTION — edit this to switch datasets
# -----------------------------
DATASET_NAME = "CSO"   # change to "STO" to run the original dataset
# -----------------------------
# PATHS
# -----------------------------
VIDEOS_DIR = PROJECT_ROOT / "stoich_dataset" / DATASET_NAME / "videos_raw"
OUTPUT_DIR = PROJECT_ROOT / "stoich_dataset" / DATASET_NAME / "preprocessed_h5"
CROP_FILE  = Path(__file__).resolve().parent / f"global_crop_{DATASET_NAME}.txt"


# -----------------------------
# LOAD GLOBAL CROP
# -----------------------------
def load_global_crop():
    """
    Read the global crop coordinates from global_crop.txt.
    The crop was computed once across all 31 videos by find_global_crop.py
    to ensure every video is cropped to the same diffraction region.

    Returns: [top, bottom, left, right] as a list of integers
    Raises ValueError if the GLOBAL_CROP line cannot be parsed.
    """
    text = CROP_FILE.read_text()
    match = re.search(r"GLOBAL_CROP\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
    if not match:
        raise ValueError(f"Could not parse GLOBAL_CROP from {CROP_FILE}")
    return [int(x) for x in match.groups()]


# -----------------------------
# PREPROCESSING SETTINGS
# -----------------------------
# ── Preprocessing settings applied to all 31 videos ───────────────────
# GLOBAL_CROP is loaded from global_crop.txt computed by find_global_crop.py.
# Bkgrnd_Crop and Bkgrnd_Each are enabled to subtract the per-frame
# background from each video independently.
# All other corrections (scar removal, RSS, drift) are off for this dataset.
GLOBAL_CROP = load_global_crop()

SETTINGS = dict(
    Frame_Period=1,
    Blank_Thresh=10,
    Kill_Br=False,
    Scar_Removal=False,
    Scar_Scope=(20, 20, 150, 25),
    Passes=10,
    Crop_Edges=True,
    Crop=GLOBAL_CROP,
    Fake_Hdr=False,
    Hdr_Type="Gamma",
    Hdr_Rescale=False,
    Gamma=2,
    Sigma=15,
    Bkgrnd_Crop=True,
    Bkgrnd_Rescale=False,
    Bkgrnd_Each=True,
    Bkgrnd_Thresh=50,
    Translate=False,
    Tshift=(0, 0),
    Drift=False,
    D_Amount=(0, 0),
    D_Type="Linear",
    Jump_Size=(5, 1),
    Rss=False,
    First_Only=False,
    Vertical=False,
    Rss_Scope=(10, 10),
    Rss_Crop=(20, 20, 20, 20),
    Invert=False,
)


# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Loop over all .avi files in stoich_dataset/videos_raw/ and run
    pre_processing() on each using the shared global crop and settings.
    Saves one H5 file per video to stoich_dataset/preprocessed_h5/.
    Tracks and reports successes and failures at the end.
    """
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Videos folder not found: {VIDEOS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(VIDEOS_DIR.glob("*.avi"))
    if not video_files:
        raise FileNotFoundError(f"No .avi files in {VIDEOS_DIR}")

    print(f"\nFound {len(video_files)} videos.")
    print(f"Global crop: {GLOBAL_CROP}")
    print(f"Output folder: {OUTPUT_DIR}\n")

    succeeded = []
    failed = []

    for i, video_path in enumerate(video_files, start=1):
        sample_id = video_path.stem
        print(f"\n[{i:2d}/{len(video_files)}] {sample_id}")
        print("-" * 60)

        try:
            out_h5 = pre_processing(
                Input_File=str(video_path),
                Out_Path=str(OUTPUT_DIR),
                Out_Name=sample_id,
                **SETTINGS,
            )
            succeeded.append(sample_id)
            print(f"  -> saved: {out_h5}")
        except Exception as e:
            failed.append((sample_id, str(e)))
            print(f"  -> FAILED: {e}")

    print("\n" + "=" * 60)
    print("BATCH PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Succeeded: {len(succeeded)}/{len(video_files)}")
    print(f"Failed:    {len(failed)}")
    if failed:
        print("\nFailures:")
        for sid, err in failed:
            print(f"  {sid}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()