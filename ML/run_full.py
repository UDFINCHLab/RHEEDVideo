import os

from pre_processing import pre_processing
from pca import pca
from k_means import k_means
from plot_pca import plot_pca
from plot_kmeans import plot_k_means


####################### Preprocessing Settings #######################

FRAME_PERIOD = 1
BLANK_THRESH = 10

KILL_BR = False

SCAR_REMOVAL = False
SCAR_SCOPE = (20, 20, 150, 25)  # (top, bottom, left, right)
PASSES = 10

CROP_EDGES = True
CROP = [670, 1706, 806, 2038] # [top, bottom, left, right]

FAKE_HDR = False
HDR_TYPE = "Gamma"  # 'CLAHE' or 'Gamma' or 'Gamma2' or 'Invert'
HDR_RESCALE = False
GAMMA = 2
SIGMA = 15

BKGRND_CROP = False
BKGRND_RESCALE = False
BKGRND_EACH = False
BKGRND_THRESH = 50

TRANSLATE = False
TSHIFT = (100, 100)  # (x, y)

DRIFT = False
D_AMOUNT = (25, 25)  # (x, y)
D_TYPE = "Linear"  # 'Linear' or 'Jumpy'
JUMP_SIZE = (5, 1)  # (x, y)

RSS = False
FIRST_ONLY = False
VERTICAL = False
RSS_SCOPE = (10, 10)  # (left, right)
RSS_CROP = [20, 20, 20, 20]  # [top, bottom, left, right] all positive

INVERT = False

####################### PCA Settings #######################
PCA_COMPONENTS = 6

####################### k-means Settings #######################
RUNS = 100
CLUSTERS = (4, 4)  # Can be a single value, a tuple, or a list of values

####################### PCA Plot Settings #######################
PCA_FIG_SIZE = (3.375, 6.75)
NUM_VECTORS = -1  # -1 for all, otherwise a positive integer

####################### k-means Plot Settings #######################
MEANS_FIG_SIZE = (3.375, 3.375)


def main() -> None:
    # ----- Resolve important paths -----
    project_root = os.path.dirname(os.path.dirname(__file__))

    captures_dir = os.path.join(project_root, "captures")
    preproc_dir = os.path.join(project_root, "results", "pre_processing")
    pca_plot_dir = os.path.join(project_root, "results", "pca_plots")
    kmeans_plot_dir = os.path.join(project_root, "results", "kmeans_plots")

    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(pca_plot_dir, exist_ok=True)
    os.makedirs(kmeans_plot_dir, exist_ok=True)

    # ----- 1. Find latest video in captures/ -----
    all_videos = [
        os.path.join(captures_dir, f)
        for f in os.listdir(captures_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not all_videos:
        raise FileNotFoundError("❌ No video files found in captures/")

    latest_video = max(all_videos, key=os.path.getctime)
    base_name = os.path.splitext(os.path.basename(latest_video))[0]

    print("\n============================================")
    print("🎬 Full RHEED ML Pipeline")
    print("============================================")
    print(f"📥 Latest video: {latest_video}")
    print(f"📂 H5 output will be in: {preproc_dir}")
    print(f"🖼 PCA plots  → {pca_plot_dir}")
    print(f"🖼 KMeans plots → {kmeans_plot_dir}")
    print("============================================\n")

    # ----- 2. Pre-processing -----
    print("🔧 Running pre-processing...")
    pre_processing_location = pre_processing(
        Input_File=latest_video,
        Out_Path=preproc_dir,
        Out_Name=base_name,
        Frame_Period=FRAME_PERIOD,
        Blank_Thresh=BLANK_THRESH,
        Kill_Br=KILL_BR,
        Scar_Removal=SCAR_REMOVAL,
        Scar_Scope=SCAR_SCOPE,
        Passes=PASSES,
        Crop_Edges=CROP_EDGES,
        Crop=CROP,
        Fake_Hdr=FAKE_HDR,
        Hdr_Type=HDR_TYPE,
        Hdr_Rescale=HDR_RESCALE,
        Gamma=GAMMA,
        Sigma=SIGMA,
        Bkgrnd_Crop=BKGRND_CROP,
        Bkgrnd_Rescale=BKGRND_RESCALE,
        Bkgrnd_Each=BKGRND_EACH,
        Bkgrnd_Thresh=BKGRND_THRESH,
        Translate=TRANSLATE,
        Tshift=TSHIFT,
        Drift=DRIFT,
        D_Amount=D_AMOUNT,
        D_Type=D_TYPE,
        Jump_Size=JUMP_SIZE,
        Rss=RSS,
        First_Only=FIRST_ONLY,
        Vertical=VERTICAL,
        Rss_Scope=RSS_SCOPE,
        Rss_Crop=RSS_CROP,
        Invert=INVERT,
    )

    print(f"✅ Pre-processing complete → {pre_processing_location}\n")

    # ----- 3. PCA -----
    print("🧠 Running PCA...")
    pca(pre_processing_location, PCA_COMPONENTS)
    print("✅ PCA complete.\n")

    # ----- 4. K-Means -----
    print("📊 Running k-means clustering...")
    k_means(Input_File=pre_processing_location, Runs=RUNS, Clusters=CLUSTERS)
    print("✅ k-means complete.\n")

    # ----- 5. Plot K-Means -----
    print("🖼 Generating k-means plots...")
    plot_k_means(
        Input_File=pre_processing_location,
        Out_Path=kmeans_plot_dir,
        Fig_size=MEANS_FIG_SIZE,
        show=True,
    )
    print("✅ k-means plotting complete.\n")

    # ----- 6. Plot PCA -----
    print("🖼 Generating PCA plots...")
    plot_pca(
        Input_File=pre_processing_location,
        Out_Path=pca_plot_dir,
        Fig_Size=PCA_FIG_SIZE,
        Num_Vectors=NUM_VECTORS,
        show=True,
    )
    print("✅ PCA plotting complete.\n")

    print("🎉 FULL PIPELINE FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
