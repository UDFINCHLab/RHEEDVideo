import h5py as h5
import numpy as np
from sklearn.cluster import KMeans







INPUT_FILE = '' # Path/File.h5
RUNS = 100
CLUSTERS = (4, 4) # a tuple of integers



def k_means(Input_File: str, Runs: int=100, Clusters: tuple[int, int]=(1,10)) -> None:
    ############################ Load Data ############################

    # Print the settings
    print_settings(Runs, Clusters, Input_File)
    
    # Load the data
    try:
        print('Loading The Data')
        with h5.File(Input_File, 'r') as f:
            pca_group: h5.Group = f['pca']
            
            # Load the eigenvalues
            eigen_values = pca_group['eigenvalues'][:]

            # Load the original data
            data_group = f['data']
            images = data_group['image_data'][:]
            
        print('Data Loaded')
        
    except Exception as e:
        raise Exception(f'Error loading the data: {e}')
    
    
    
    
    ############################ Perform k-means ############################
    
    # Determine the number of clusters to run PCA on based on the shape 
    # of the clusters tuple
    if len(Clusters) == 1: # If one value is given, run from 1 to that value
        cluster_nums = np.arange(1, Clusters[0]+1)
    elif len(Clusters) == 2: # If two values are given, run from the first to the second
        cluster_nums = np.arange(Clusters[0], Clusters[1]+1)
    else: # If more than two values are given, use all values
        cluster_nums = list(Clusters)

    
    kmeans_list: list[KMeans] = []
    labels_list: list[np.ndarray] = []
    
    print(f'Starting k-menas clustering')
    for i, clusters in enumerate(cluster_nums):
        # Create the k-means model
        kmeans = KMeans(n_clusters=clusters, n_init=Runs, max_iter=500)
    
        # Fit the model
        kmeans.fit(eigen_values)
        
        kmeans_list.append(kmeans)
        
        labels_list.append(kmeans.labels_)
        
        print(f'k={clusters} completed')
        
    print('k-means clustering completed')
    
    ############################ Sort Clusters ############################
    
    for ldx, labels in enumerate(labels_list):
        labels_list[ldx] = sort_clusters(labels)
    
    
    ############################ Calculate Centroids ############################

    centroids_list = calculate_centroids(images, labels_list)

    
    ############################ Save Results ############################
        
    
    with h5.File(Input_File, 'r+') as file:
        # Create the kmeans group
        group_name = 'kmeans'
        if group_name in file:
            del file[group_name]
        kmeans_group = file.create_group('kmeans')

        # Save the kmeans models
        for i, kmeans in enumerate(kmeans_list):
            
            # Get the cluster number
            cluster_num = cluster_nums[i]
            
            # Create the sub group for k=cluster_num
            cluster_group = kmeans_group.create_group(f'k={cluster_num}')
            
            # Save the results as datasets in the sub group
            cluster_group.create_dataset('labels', data=labels_list[i])
            cluster_group.create_dataset('cluster_centers', data=kmeans.cluster_centers_)
            cluster_group.create_dataset('inertia', data=kmeans.inertia_)
            cluster_group.create_dataset('n_iter', data=kmeans.n_iter_)
            cluster_group.create_dataset('centroids', data=centroids_list[i])
            
            
            # Save the settings as attributes to the sub group
            cluster_group.attrs['runs'] = Runs
            cluster_group.attrs['clusters'] = cluster_num
            
    return      

        
        
        
   
    
    
def sanitize_output_path(path:str, f_name: str) -> tuple[str, str]:
    '''
    Sanitizes the output path by ensuring there is a '/' at the end of the path.
    And remove any extensions from the file name.
    '''
    if path[-1] != '/':
        path = path + '/'
        
    f_name = f_name.split('.')[0]
    
    return path, f_name


def print_settings(runs, clusters, input_file):
    '''
    Prints the settings specified by user
    '''
    print('Settings:')
    print(f'Runs: {runs}')
    print(f'Clusters: {clusters}')
    print(f'Input File: {input_file}')
    print('\n\n')
    
    return


def calculate_centroids(images: np.ndarray, label_list: list[np.ndarray]) -> np.ndarray:
    
    centroids_list = []
    
    for i, labels in enumerate(label_list):
        
        cluster_nums = np.unique(labels)
        
        centroid = np.zeros((images.shape[0], images.shape[1], len(cluster_nums)))
        
        for cluster_num in cluster_nums:
            cluster_indices = np.where(labels == cluster_num)[0]
            
            cluster_images = images[:,:,cluster_indices]
            
            cluster_mean = np.mean(cluster_images, axis=2)
            
            centroid[:,:,cluster_num] = cluster_mean
            
        centroids_list.append(centroid)
        
    return centroids_list
        





def sort_clusters(cluster_labels: np.ndarray) -> np.ndarray:

    # Dictionary to store the first occurrence index of each label
    first_occurrence = {}
    
    # Traverse the array to record the first occurrence of each label
    for i, label in enumerate(cluster_labels):
        if label not in first_occurrence:
            first_occurrence[label] = len(first_occurrence)
    
    # Create a new array with reordered labels
    reordered_arr = np.array([first_occurrence[label] for label in cluster_labels])
    
    return reordered_arr




if __name__ == "__main__":
    import os

    # 1. Folder containing pre-processing output files (.h5)
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

    # 2. Pick the latest .h5 file
    latest_h5 = max(all_h5, key=os.path.getmtime)
    print(f"\n📌 Latest H5 File Detected: {latest_h5}\n")

    # 3. Run K-Means (writes inside H5 file — no folder needed)
    k_means(latest_h5, RUNS, CLUSTERS)

    print("\n✅ K-means clustering complete!")
    print(f"K-means results written inside H5 file: {latest_h5}")
