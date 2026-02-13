import h5py as h5
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Optional





INPUT_FILE = '' # Path/File.h5
PCA_COMPONENTS = 6

def pca(Input_File: str, PCA_Components: int) -> None:
    '''
    Runs PCA on the data provided in the input file. The PCA will reduce the data
    to the number of components specified by the user.
    
    Parameters:
                Input_File: str - The path to the input file
                PCA_Components: int - The number of components to reduce the data to
                
    Returns:
                result: bool - True if the PCA was successful, False otherwise
    
    '''
    ############################ Load Data ############################
    
    # Print the settings
    print_settings(Input_File, PCA_Components)
    
    # Load the data
    try:
        print('Loading The Data')
        with h5.File(Input_File, 'r') as file:
            data_group: h5.Group = file['data']
            
            dataset: h5.Dataset = data_group['image_data']
            
            # Load the image data into an array
            input_data = dataset[:]
            
            # Load the attributes into a dictionary
            attrs = {key: dataset.attrs[key] for key in dataset.attrs}
            
            times = data_group['times'][:]
            
        print('Data Loaded')
        
    except Exception as e:
        raise Exception(f'Error loading the data: {e}')
    
    
    # Convert the image data to grayscale
    image_data = convert_video_to_grayscale(input_data)
    
    # Load some of the attributes into variables
    try:
        img_width = attrs['x_res']
        img_height = attrs['y_res']
        total_frames = attrs['total_frames']
    except KeyError as e:
        raise KeyError(f'Key not found in the attributes: {e}')
    
    
    # Trim the data
    image_data, empty_cols = trim_data(image_data)
    
    
    
    
    ############################ Perform PCA ############################
    print('Performing PCA')
    
    reshaped_data = np.reshape(image_data, (img_height*img_width, total_frames)).T
    
    # Normalize the data
    normalized_image_data, scalar = normalize_data2(reshaped_data)

    # Perform PCA
    pca = PCA(PCA_Components)
    
    # Fit PCA to the reshaped video data
    eigenvalues = pca.fit_transform(normalized_image_data)
    
    variances = pca.explained_variance_ratio_
    total_variance = np.sum(variances)
    
    # Get the eigenvectors
    eigenvectors = pca.components_.T.reshape((img_height, img_width, PCA_Components))
    
    print('PCA Complete')

    ############################ Save the results ############################
    
    with h5.File(Input_File, 'r+') as file:
        
        ########### Save the PCA results ###########
        # create a group to hold the results
        group_name = 'pca'
        if group_name in file:
            del file[group_name]
            
        results_group = file.create_group('pca')
        
        # Construct datasets inside the results group for the eigenvectors and eigenvalues
        results_group.create_dataset('eigenvectors', data=eigenvectors)
        values = results_group.create_dataset('eigenvalues', data=eigenvalues)
        results_group.create_dataset('explained_variance', data=variances)
        results_group.create_dataset('means', data=pca.mean_)
        results_group.create_dataset('scalar_mean', data=scalar.mean_)
        results_group.create_dataset('scalar_std', data=scalar.scale_)
        results_group.create_dataset('scalar_var', data=scalar.var_)
        
        # Add attributes to the eigenvalues dataset
        values.attrs['explained_variance'] = total_variance
        values.attrs['n_components'] = PCA_Components
        
    
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



def convert_video_to_grayscale(image_data: np.ndarray) -> np.ndarray:
    '''
    Converts the image data to grayscale by removing the R and B channels if
    they are empty. If they are not empty, the image is converted to grayscale
    using the formula: (R + G + B) / 3
    
    Parameters:
                image_data: np.ndarray - A 3D or 4D array of image data
                                        with shape: [y_res, x_res, channels, frames]
                                        or shape: [y_res, x_res, frames] 
    
    Output:
                np.ndarray - The image data converted to grayscale
    '''
    
    if len(image_data.shape) == 3:
        gray = image_data
        
    elif len(image_data.shape) not in (3, 4):
        shape = str(image_data.shape)
        raise ValueError(f'image_data has an invalid shape: {shape}. Expected 3 or 4 dimensions')
        
    else:
        blue, green, red = image_data[:, :, 0, :], image_data[:, :, 1, :], image_data[:, :, 2, :]
        
        if np.mean(blue) == 0 < 50 and np.mean(red) == 0 < 50:
            gray = green
            
        else:
            gray = (red + green + blue) / 3
            
        # Remove the 3rd dimension
        gray = np.squeeze(gray)
    
    return gray


def print_settings(input_file: str, components: int) -> None:
    '''
    Prints the settings specified by user
    '''
    print('Settings:')
    print(f'PCA Components: {components}')
    print(f'Input File: {input_file}')
    print('\n\n')
    
    return




def trim_data(image_data: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    '''
    Removes any empty columns from the image data. 
    This is done by checking the sum of the columns. If the sum is 0, the column
    is removed.
    
    Parameters:
                image_data: np.ndarray - The image data to trim
                
    Returns:
                np.ndarray - The trimmed image data
                np.ndarray - The indices of the columns removed
    '''

    # Find the columns that are empty
    empty_cols = np.where(np.sum(image_data, axis=0) == 0)[0]
    
    # Remove the empty columns
    trimmed_data = np.delete(image_data, empty_cols, axis=1)
    
    return trimmed_data, empty_cols



def repad_data(image_data: np.ndarray, empty_cols: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    '''
    Repads the image data by adding empty columns back into the data. The empty columns
    are added back in the same location they were removed from.
    
    Parameters:
                image_data: np.ndarray - The image data to repad
                empty_cols: np.ndarray - The indices of the columns removed
                img_width: int - The width of the original image
                img_height: int - The height of the original image
                
    Returns:
                np.ndarray - The repadded image data
    '''
    
    # Create an array of zeros with the same height as the image data
    empty_col = np.zeros((img_height, 1))
    
    # Create a list to store the repadded data
    repadded_data = []
    
    # Iterate over the empty columns and add them back into the data
    for i in range(image_data.shape[1] + len(empty_cols)):
        if i in empty_cols:
            repadded_data.append(empty_col)
            
        else:
            repadded_data.append(image_data[:, i])
    
    # Stack the repadded data
    repadded_data = np.hstack(repadded_data)
    
    return repadded_data



def normalize_data(image_data: np.ndarray) -> tuple[np.ndarray, float, float] :
    '''
    Normalizes the image data by subtracting the mean and dividing by the standard deviation
    to result in a mean of 0 and standard deviation of 1.
    
    Parameters:
                image_data: np.ndarray - The image data to normalize
                
    Returns:
                np.ndarray - The normalized image data
                float - The mean of the image data
                float - The standard deviation of the image data
    '''
    
    # Calculate the mean and standard deviation
    mean = np.mean(image_data, axis=0)
    std = np.std(image_data, axis=0)
    
    # Normalize the data
    normalized_data = (image_data - mean) / std
    
    return normalized_data, mean, std


def denormalize_data(normalized_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    '''
    Denormalizes the image data by multiplying by the standard deviation and adding the mean
    back to the data.
    
    Parameters:
                normalized_data: np.ndarray - The normalized image data
                mean: float - The mean of the original image data
                std: float - The standard deviation of the original image data
                
    Returns:
                np.ndarray - The denormalized image data
    '''
    
    # Denormalize the data
    denormalized_data = (normalized_data * std) + mean
    
    return denormalized_data



def normalize_data2(image_data: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    '''
    Normalizes the image data by subtracting the mean and dividing by the standard deviation
    to result in a mean of 0 and standard deviation of 1.
    
    Parameters:
                image_data: np.ndarray - The image data to normalize
                
    Returns:
                np.ndarray - The normalized image data
                StandardScaler - The StandardScaler object used to normalize the data
    '''
    
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    # Fit the scaler to the data
    scaler.fit(image_data)
    
    # Normalize the data
    normalized_data = scaler.transform(image_data)
    
    return normalized_data, scaler


def denormalize_data2(normalized_data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    '''
    Denormalizes the image data by multiplying by the standard deviation and adding the mean
    back to the data.
    
    Parameters:
                normalized_data: np.ndarray - The normalized image data
                scaler: StandardScaler - The StandardScaler object used to normalize the data
                
    Returns:
                np.ndarray - The denormalized image data
    '''
    
    # Denormalize the data
    denormalized_data = scaler.inverse_transform(normalized_data)
    
    return denormalized_data












def plot_values_and_vectors(eigenvectors: np.ndarray, eigenvalues: np.ndarray, n_components: int, scalar: Optional[StandardScaler], shape: Optional[tuple[int, int]]):
    '''
    Plots the eigenvalues and eigenvectors of the PCA
    
    Parameters:
                eigenvectors: np.ndarray - The eigenvectors of the PCA
                eigenvalues: np.ndarray - The eigenvalues of the PCA
    '''
    
    if scalar is not None:
        if shape is None:
            raise ValueError('If a scalar is provided, the shape of the original data must be provided')
        
        eigenvectors = scalar.inverse_transform(eigenvectors)
    
        eigenvectors = eigenvectors.T.reshape(shape[0], shape[1], n_components)
        
    for i in range(n_components):
        cv.imshow(f'Eigenvector {i}', eigenvectors[:,:,i].astype(np.uint8))
        cv.waitKey(0)
        '''
        plt.figure()
        plt.imshow(eigenvectors[:,:,i], cmap='gray')
        plt.title(f'Eigenvector {i}')
        plt.show()
        '''
    
    for i in range(n_components):
        plt.figure()
        plt.plot(eigenvalues[i])
        plt.title(f'Eigenvalues {i}')
        plt.show()

    return

def plot_values_and_vectors2(eigenvectors: np.ndarray, eigenvalues: np.ndarray, n_components: int):
    '''
    Plots the eigenvalues and eigenvectors of the PCA
    
    Parameters:
                eigenvectors: np.ndarray - The eigenvectors of the PCA
                eigenvalues: np.ndarray - The eigenvalues of the PCA
    '''

    for i in range(n_components):
        plt.figure()
        plt.imshow(eigenvectors[:,:,i], cmap='gray')
        plt.title(f'Eigenvector {i}')
        plt.show()
    
    x = np.linspace(0, eigenvalues.shape[0], eigenvalues.shape[0])
    for i in range(n_components):
        plt.figure()
        plt.plot(x, eigenvalues[:, i])
        plt.title(f'Eigenvalues {i}')
        plt.show()

    return




def plot_image(image: np.ndarray, title: str=''):
    '''
    Plots an image
    
    Parameters:
                image: np.ndarray - The image to plot
                title: str - The title of the plot
    '''
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    
    return








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

    # 3. Run PCA (writes inside H5 file — no folder needed)
    pca(latest_h5, PCA_COMPONENTS)

    print("\n✅ PCA processing complete!")
    print(f"PCA results written inside H5 file: {latest_h5}")
