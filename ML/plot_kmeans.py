import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py as h5

from os import makedirs

from typing import Optional, cast, Any



INPUT_FILE = ''
OUT_PATH = ''


FIG_SIZE = (7, 3)


def plot_k_means(Input_File: str, 
                 Out_Path: str,
                 Fig_size: tuple[float,float]=(7,3), 
                 show: bool=False) -> None:

    plt.rc('font', family='Times New Roman', weight='bold', size=8)
    
    
    if Out_Path[-1] != '/':
        Out_Path += '/'
        
    makedirs(Out_Path, exist_ok=True)
    
    labels: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    
    with h5.File(Input_File, 'r') as file:
        # Access the kmeans group
        kmeans_group: h5.Group = file['kmeans']
                
        # Load the times
        data_group = file['data']
        times = data_group['times'][:]
        
        # Determine the keys for each k value
        cluster_keys = list(kmeans_group.keys())
        
        # Load the cluster labels for each k value
        for key in cluster_keys:
            current_labels = kmeans_group[key]['labels'][:]
            labels.append(current_labels)
            current_centroids = kmeans_group[key]['centroids'][:]
            centroids.append(current_centroids)

    plot_all_clusters(labels, times, Fig_size, Out_Path, show)
    
    plot_all_centroids(centroids, Out_Path, show)
    

   
    return



def plot_all_centroids(centroids_list: list[np.ndarray], out_path: str, show: bool=False) -> None:
    
    for i, centroids in enumerate(centroids_list):
        k = centroids.shape[-1]
        path = f'{out_path}/k={k}/'
        makedirs(path, exist_ok=True)
        plot_centroids(centroids, path=path, show=show)

    return


def plot_centroids(centroids: np.ndarray, path: str, fig_size: Optional[tuple[float, float]]=None, show: bool=False) -> None:
    
    
    num_clusters = centroids.shape[-1]
    
    for i in range(num_clusters):
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        ax.imshow(centroids[:,:,i], cmap=centroid_color_map(), vmin=0, vmax=255)
        
        ax.set_axis_off()
        
        if show:
            plt.show()
    
        fig.savefig(f'{path}Cluster {i+1}.png', dpi=1000)
    
    return


def plot_all_clusters(labels_list: list[np.ndarray], times: np.ndarray, fig_size: tuple[float, float], save_path: str, show: bool=False) -> None:
    if save_path[-1] != '/':
        save_path += '/'
        
    for i, labels in enumerate(labels_list):
        k = np.unique(labels).size
        k_save_path = f'{save_path}k={k}/'
        makedirs(k_save_path, exist_ok=True)
        plot_kmeans(labels, times, fig_size, k_save_path, show)
        
    return



def plot_kmeans(labels: np.ndarray, times: np.ndarray, fig_size: tuple[float, float],
                save_path: str, show: bool=False) -> None:
        
    # If the labels start at 0, add 1 to all labels
    if min(labels) == 0:
        labels = labels + 1
        
    fig, ax = plt.subplots(figsize=fig_size)

    # --------------------------
    # Convert time to hours
    # --------------------------
    times, unit = scale_times(times)

    # --------------------------
    # Scatter plot
    # --------------------------
    scatter = ax.scatter(
        times, labels, 
        c=labels, s=8,
        cmap="viridis", 
        edgecolor="none"
    )

    # --------------------------
    # Axis labels (High quality)
    # --------------------------
    ax.set_xlabel("Time (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Cluster", fontsize=10, fontweight='bold')

    # --------------------------
    # Title (optional)
    # --------------------------
    ax.set_title(f"K-means Clustering (k={labels.max()})",
                 fontsize=11, fontweight='bold')

    # --------------------------
    # Ticks
    # --------------------------
    ax.set_yticks(np.arange(labels.min(), labels.max() + 1, 1))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(labels.min() - 0.3, labels.max() + 0.3)

    # --------------------------
    # Grid style
    # --------------------------
    ax.grid(True, color="gray", alpha=0.4, linewidth=0.7)

    # --------------------------
    # Add colorbar (optional)
    # --------------------------
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster Index", fontsize=9, fontweight="bold")

    # --------------------------
    # Save output file
    # --------------------------
    fig.tight_layout()
    fig.savefig(f"{save_path}Clustering.png", dpi=600)

    if show:
        plt.show()

    plt.close(fig)




















def sort_clusters(cluster_labels: np.ndarray) -> np.ndarray:

    # Dictionary to store the first occurrence index of each label
    first_occurrence = {}
    
    # Traverse the array to record the first occurrence of each label
    for i, label in enumerate(cluster_labels):
        if label not in first_occurrence:
            first_occurrence[label] = len(first_occurrence) + 1
    
    # Create a new array with reordered labels
    reordered_arr = np.array([first_occurrence[label] for label in cluster_labels])
    
    return reordered_arr

    
def scale_times(times: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Always convert time to hours for consistency with reference plots.
    The raw `times` from the H5 file are in seconds.
    """
    return times / 3600.0, 'hours'

    
    









def adjustments(fig_size: tuple[float,float])-> list[float]:
    #              L,     B,     R,     T,     WS,    HS,    YT,     XT
    if fig_size == (7, 2):
        adjusts = [0.06,  0.2,   0.99,  0.88,  0.2,   0.2,   -0.035, 0.00]
    elif fig_size == (3.5, 2):
        adjusts = [0.12,  0.2,   0.95,  0.88,  0.2,   0.2,   -0.065, 0.00]
    elif fig_size == (7, 3):
        adjusts = [0.06,  0.2,   0.99,  0.88,  0.2,   0.2,   -0.035, 0.00]
    elif fig_size == (3.375, 1):
        adjusts = [0.08,  0.3,   0.99,  0.97,  0.00,  0.00,  -0.05,  -0.3]
    elif fig_size == (2, 1):
        adjusts = [0.15,  0.3,   0.99,  0.97,  0.00,  0.00,  -0.10,  -0.3]
    elif fig_size == (1.687, 1):
        adjusts = [0.15,  0.3,   0.99,  0.97,  0.00,  0.00,  -0.10,  -0.3]
    else:
        adjusts = [0.00,  0.00,  1.00,  1.00,  0.00,  0.00,  -0.10,  0.00]
        
    return adjusts








def au_color_map() -> LinearSegmentedColormap:
    # Define the RGB values for the start and end colors
    start_color = (11/255, 35/255, 65/255)  # Blue
    end_color = (242/255, 107/255, 5/255)    # Orange

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color], N=256)
    
    return cmap


def rwb_color_map() -> LinearSegmentedColormap:
    # Define the RGB values for the start and end colors
    start_color = (0, 0, 1)  # Blue
    middle_color = (1, 1, 1) # White
    end_color = (1, 0, 0)    # Red

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=256)
    
    return cmap


def centroid_color_map() -> str:
    # Define the RGB values for the start and end colors
    start_color = (0, 0, 0)  # Black
    middle_color = (0, 1, 0) # Green
    end_color = (1, 1, 1)    # White

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=256)
    
    return cmap


def train_color() -> str:
    return '#0b2341'

def val_color() -> str:
    return '#e86100'

def test_color() -> str:
    return '#215834'

def background_color() -> str:
    return '#ced3d9'

















if __name__ == "__main__":
    import os

    # 1. Folder containing H5 files generated by preprocessing
    preproc_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "pre_processing"
    )

    # collect all .h5 files
    all_h5 = [
        os.path.join(preproc_dir, f)
        for f in os.listdir(preproc_dir)
        if f.lower().endswith(".h5")
    ]

    if not all_h5:
        raise FileNotFoundError("❌ No .h5 files found in results/pre_processing/")

    # 2. Pick the latest H5 file
    latest_h5 = max(all_h5, key=os.path.getmtime)
    print(f"\n📌 Latest H5 File Detected: {latest_h5}\n")

    # 3. Output folder for cluster plots only
    kmeans_plot_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "kmeans_plots"
    )
    os.makedirs(kmeans_plot_dir, exist_ok=True)

    # 4. Run k-means plotting (plots only, no H5 writing)
    plot_k_means(
        Input_File=latest_h5,
        Out_Path=kmeans_plot_dir,
        Fig_size=FIG_SIZE,
        show=False
    )

    print("\n✅ K-Means plots generated!")
    print(f"📁 Saved in: {kmeans_plot_dir}")
