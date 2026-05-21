"""
RHEED video preprocessing pipeline — pre_processing.py

Converts a raw RHEED video file into a preprocessed HDF5 file ready for
PCA and K-Means analysis. Applies a configurable sequence of image corrections
per frame including cropping, scar removal, background subtraction, HDR
simulation, RSS spot alignment, drift correction, and inversion.

The output H5 file stores the processed frame stack, timestamps, and all
settings used as attributes so results are fully reproducible.

Main entry point: pre_processing()
Run directly to process the latest video in the captures/ folder.

Dependencies: OpenCV, NumPy, h5py
"""
# Imports
import os
import numpy as np
import cv2 as cv
import h5py as h5
from typing import Optional
import zipfile as zf
from math import floor


# ── Default settings (used when running this file directly) ───────────
# These are overridden when pre_processing() is called from run_full.py
# or any other pipeline script that passes its own parameters.
# See pre_processing() docstring for full parameter descriptions.
IN_FILE = ''
OUT_PATH = ''
OUT_NAME = '' # Will be an h5 file 

FRAME_PERIOD = 25
BLANK_THRESH = 10

KILL_BR = False

SCAR_REMOVAL = False
SCAR_SCOPE = (10, 10, 150, 25) # (top, bottom, left, right)
PASSES = 10

CROP_EDGES = False
CROP = [670, 1706, 806, 2038] # [top, bottom, left, right]

FAKE_HDR = False
HDR_TYPE = 'Invert' # 'CLAHE' or 'Gamma' or 'Gamma2' or 'Invert'
HDR_RESCALE = False
GAMMA = 2
SIGMA = 15

BKGRND_CROP = False
BKGRND_RESCALE = False
BKGRND_EACH = False
BKGRND_THRESH = 50

TRANSLATE = False
TSHIFT = (100, 100) # (x, y)

DRIFT = False
D_AMOUNT = (25, 25) # (x, y)
D_TYPE = 'Linear' # 'Linear' or 'Jumpy'
JUMP_SIZE = (5, 1) # (x, y)

RSS = False
FIRST_ONLY = False
VERTICAL = False
RSS_SCOPE = (10, 10) # (left, right)
RSS_CROP = [50, 50, 50, 50] # [top, bottom, left, right] all positive
    
    
    
    
    

def pre_processing(
    Input_File: str,
    Out_Path: str,
    Out_Name: str,
    Frame_Period: int = -1,
    Blank_Thresh: int = 3,
    Kill_Br: bool = False,
    Scar_Removal: bool = False,
    Scar_Scope: tuple[int, int, int, int] = (0, 0, 0, 0),
    Passes: int = 1,
    Crop_Edges: bool = False,
    Crop: tuple[int, int, int, int] = (0, 0, 0, 0),
    Fake_Hdr: bool = False,
    Hdr_Type: str = 'Gamma',
    Hdr_Rescale: bool = False,
    Gamma: float = 1.5,
    Sigma: float = 15,
    Bkgrnd_Crop: bool = False,
    Bkgrnd_Rescale: bool = False,
    Bkgrnd_Each: bool = False,
    Bkgrnd_Thresh: float = 50,
    Translate: bool = False,
    Tshift: tuple[int, int] = (0, 0),
    Drift: bool = False,
    D_Amount: tuple[int, int] = (0, 0),
    D_Type: str = 'Linear',
    Jump_Size: tuple[int, int] = (5, 1),
    Rss: bool = False,
    First_Only: bool = False,
    Vertical: bool = False,
    Rss_Scope: tuple[int, int] = (10, 10),
    Rss_Crop: tuple[int, int, int, int] = (0, 0, 0, 0),
    Invert: bool = False
) -> str:
    
    '''
    Preprocesses a video file to prepare it for PCA
    
    Parameters:
                Input_File: str - the path to the video file to preprocess
                Out_Path: str - the path to save the output file to
                Out_Name: str - the name of the output file
                Frame_Period: int - the period to sample the frames at
                Blank_Thresh: int - the threshold defining a blank frame
                Kill_Br: bool - whether to zero the blue and red channels
                Scar_Removal: bool - whether to remove scars from the image
                Scar_Scope: tuple[int, int, int, int] - the scope of the scars to remove
                Passes: int - the number of passes to remove scars
                Crop_Edges: bool - whether to crop the edges of the image
                Crop: tuple[int, int, int, int] - the amount to crop the image [top, bottom, left, right]
                Fake_Hdr: bool - whether to fake an HDR image
                Hdr_Type: str - the type of HDR to fake
                Hdr_Rescale: bool - whether to rescale the HDR image
                Gamma: float - the gamma value for the HDR image
                Sigma: float - the sigma value for the HDR image
                Bkgrnd_Crop: bool - whether to remove the background
                Bkgrnd_Rescale: bool - whether to rescale the background
                Bkgrnd_Each: bool - whether to calculate the background for each frame
                Bkgrnd_Thresh: float - the threshold for the background
                Translate: bool - whether to translate the image
                Tshift: tuple[int, int] - the amount to translate the image by
                Drift: bool - whether to drift the image
                D_Amount: tuple[int, int] - the amount to drift the image by
                D_Type: str - the type of drift to apply
                Jump_Size: tuple[int, int] - the size of the jumps for jumpy drift
                Rss: bool - whether to find the RSS spot
                First_Only: bool - whether to only find the RSS spot in the first frame
                Vertical: bool - whether to only find the RSS spot in the vertical direction
                Rss_Scope: tuple[int, int] - the scope to find the RSS spot in
                Rss_Crop: tuple[int, int, int, int] - the crop to apply to the RSS spot
                Invert: bool - whether to invert the image
                
    Returns:
                path: str - the location of the output file
    '''
    # ── Setup output path and initialize video source ──────────────────
    out_path, out_name = sanitize_output_path(Out_Path, Out_Name)
    
    # Construct the output file
    out_name = out_path + out_name + '.h5'

    out_fps = 1/Frame_Period


    # Make sure the output directory exists
    try:
        os.makedirs(Out_Path)
    except FileExistsError:
        pass
    

    
    file_name = Input_File.split('/')[-1]
    


    ########################### Print the start string ###########################
    print('\n\n\n')
    print('###############################################')
    print(f'Processing file: {file_name}' ) 
    print('###############################################')
    print('\n\n\n')
    
    
    
    ############################# Initilze video source #############################

    # Open the video file
    video = cv.VideoCapture(Input_File)# type: ignore
    
    _, first_frame = video.read()
    
    native_x_res, native_y_res, _ = first_frame.shape
    
    
    if Crop_Edges:
        first_frame = first_frame[Crop[0]:Crop[1], Crop[2]:Crop[3]]
    
    y_res, x_res, _ = first_frame.shape
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))# type: ignore
    
    # Get the native frame rate of the video
    native_fps = video.get(cv.CAP_PROP_FPS)# type: ignore

    if out_fps > native_fps:
        out_fps = native_fps
        
    first_frame_idx = find_first_frame(video)
        
    frames_to_sample, _ = frame_rate_sampeling(native_fps, out_fps, total_frames, first_frame_idx)
    
    out_frames = len(frames_to_sample)
    
    
    
    
    # Create an array to store the image data
    # This array will be saved to the output file at the end
    # ── Allocate output arrays ─────────────────────────────────────────
    # image_data holds the processed frame stack (y, x, frames) as uint8.
    # times holds the camera timestamp in seconds for each sampled frame.
    image_data = np.empty((y_res, x_res, out_frames), dtype=np.uint8)
    
    times = np.empty(out_frames, dtype=np.float64)
    
    
    
    ############################# Edit precalculations #############################

    # ── Pre-calculate values needed before the main frame loop ─────────
    # Background threshold, RSS center, and drift shift arrays are computed
    # once here so they are ready when the per-frame loop runs below.
    if Bkgrnd_Crop:
        
        # Get the first of the video
        video.set(cv.CAP_PROP_POS_FRAMES, first_frame_idx)
        ret, first_frame = video.read()
        if Crop_Edges:
            first_frame = first_frame[Crop[0]:Crop[1], Crop[2]:Crop[3]]
        
        bkgrnd_thresh = calc_bacground_thresh(first_frame, BKGRND_THRESH)
        
        
    if Rss:
        
        # Get the first frame of the video
        video.set(cv.CAP_PROP_POS_FRAMES, first_frame_idx)
        ret, first_frame = video.read()
        
        if Crop_Edges:
            first_frame = first_frame[Crop[0]:Crop[1], Crop[2]:Crop[3]]
        
        old_center = find_center(first_frame, crop=Rss_Crop)
        
        frame_center = (x_res//2, y_res//2)
        
        initial_shift = (frame_center[0] - old_center[0], frame_center[1] - old_center[1])
        

        
    if Drift:
        if D_Type == 'Linear':
            h_shifts = np.linspace(0, D_Amount[0], out_frames, dtype=int)
            v_shifts = np.linspace(0, D_Amount[1], out_frames, dtype=int)
        
        elif D_TYPE == 'Jumpy':
            h_steps = D_Amount[0] // Jump_Size[0]
            v_steps = D_Amount[1] // Jump_Size[1]
            h_shifts = np.linspace(0, h_steps, out_frames, dtype=int) * Jump_Size[0]
            v_shifts = np.linspace(0, v_steps, out_frames, dtype=int) * Jump_Size[1]
            
        d_shifts = list(map(lambda x, y: (x, y), h_shifts, v_shifts))
        
        
    video.set(cv.CAP_PROP_POS_FRAMES, first_frame_idx)
    _, first_frame = video.read()





    ############################# Perform Edits #############################

    # ── Main per-frame processing loop ────────────────────────────────
    # Each sampled frame is loaded, all enabled corrections are applied
    # in order, then stored into image_data. Progress is printed every 1%.
    for fdx, i in enumerate(frames_to_sample):
        video.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if Crop_Edges:
            frame = frame[Crop[0]:Crop[1], Crop[2]:Crop[3]]
            
        if Scar_Removal:
            frame = scar_removal(frame, Passes, Scar_Scope)
            
            
        if Fake_Hdr:
            if Hdr_Type == 'CLAHE':
                frame[:,:,1] = fake_hdr(frame[:,:,1])
            elif Hdr_Type == 'Gamma':
                frame[:,:,1] = fake_hdr_gamma(frame[:,:,1], Gamma)
            elif Hdr_Type == 'Invert':
                frame[:,:,1] = fake_hdr_invert(frame[:,:,1], Sigma)
            elif Hdr_Type == 'Gamma2':
                frame[:,:,1] = fake_hdr_gamma2(frame[:,:,1], Gamma)
                
                
            if Hdr_Rescale:
                frame[:,:,1] = rescale(frame[:,:,1], 20)
        
        if Bkgrnd_Crop:
            if Bkgrnd_Each:
                bkgrnd_thresh = calc_bacground_thresh(frame, Bkgrnd_Thresh)
            frame[:,:,1] = remove_and_subtract_background(frame[:,:,1], bkgrnd_thresh, Bkgrnd_Rescale)
            
        if Rss:
            if First_Only:
                frame = translate(frame, initial_shift)
                
            else:
                scope = (old_center[0] - Rss_Scope[0], old_center[0] + Rss_Scope[1])
                new_center = find_center(frame[:,:,1], scope, Rss_Crop)
                
                # Add on the initial shift
                new_shift = (frame_center[0] - new_center[0], frame_center[1] - new_center[1])
                
                # Set vertical shift to zero if not desired
                if not Vertical:
                    new_shift = (new_shift[0], initial_shift[1])
                    
                frame = translate(frame, new_shift)
                
                old_center = new_center

            
        elif Translate:
            frame = translate(frame, Tshift)
        
        elif Drift:
            frame = translate(frame, d_shifts[fdx])
            
        if Invert:
            frame = invert(frame)
        
            
        if Kill_Br:
            frame[:,:,0] = 0
            frame[:,:,2] = 0
        
        # Save the frame to the image data array
        image_data[:, :, fdx] = frame[:,:,1]
        # Save the time in ms to the times array
        times[fdx] =  video.get(cv.CAP_PROP_POS_MSEC)/1000
        
        
        # Print the progress every 1% of the way through
        progress = (fdx + 1) / out_frames * 100
        if (fdx + 1) % (np.ceil(out_frames / 100)) == 0:
            print(f"Progress: {int(progress)}%")

        
        
    # Open the output file
    # ── Write output HDF5 file ─────────────────────────────────────────
    # Saves the processed frame stack, timestamps, video metadata,
    # and all preprocessing settings as dataset attributes.
    # Settings are saved so any future run can be reproduced exactly.
    with h5.File(out_name, 'w') as out_file:
        
        data_group = out_file.create_group('data')
        
        # Save the image data as a dataset in the output file
        d_set = data_group.create_dataset('image_data', data=image_data)
        time_set = data_group.create_dataset('times', data=times)
        
        # Save the other information need for pca, k-means, and post processing in the attrs
        d_set.attrs['fps'] = out_fps
        d_set.attrs['total_frames'] = out_frames
        d_set.attrs['x_res'] = x_res
        d_set.attrs['y_res'] = y_res
        d_set.attrs['first_frame_num'] = first_frame_idx
        d_set.attrs['last_frame_num'] = frames_to_sample[-1]
        d_set.attrs['duration'] = out_frames / out_fps
        
        
        
        # Save information about the raw video
        d_set.attrs['native_fps'] = native_fps
        d_set.attrs['native_total_frames'] = total_frames
        d_set.attrs['native_x_res'] = native_x_res
        d_set.attrs['native_y_res'] = native_y_res
        d_set.attrs['native_duration'] = total_frames / native_fps
        
        
        
        # Save all the settings incase you want to know what was done

        d_set.attrs['bkgrnd_crop'] = Bkgrnd_Crop
        d_set.attrs['bkgrnd_rescale'] = Bkgrnd_Rescale
        d_set.attrs['bkgrnd_each'] = Bkgrnd_Each
        d_set.attrs['bkgrnd_thresh'] = Bkgrnd_Thresh
        d_set.attrs['hdr'] = Fake_Hdr
        d_set.attrs['hdr_type'] = Hdr_Type
        d_set.attrs['hdr_rescale'] = Hdr_Rescale
        d_set.attrs['gamma'] = Gamma
        d_set.attrs['sigma'] = Sigma
        d_set.attrs['crop_edges'] = Crop_Edges
        d_set.attrs['crop'] = Crop
        d_set.attrs['scar_removal'] = Scar_Removal
        d_set.attrs['scar_scope'] = Scar_Scope
        d_set.attrs['passes'] = Passes
        d_set.attrs['translate'] = Translate
        d_set.attrs['tshift'] = Tshift
        d_set.attrs['drift'] = Drift
        d_set.attrs['d_amount'] = D_Amount
        d_set.attrs['d_type'] = D_Type
        d_set.attrs['jump_size'] = Jump_Size
        d_set.attrs['rss'] = Rss
        d_set.attrs['first_only'] = First_Only
        d_set.attrs['vertical'] = Vertical
        d_set.attrs['rss_scope'] = Rss_Scope
        d_set.attrs['rss_crop'] = Rss_Crop
        d_set.attrs['invert'] = Invert
        d_set.attrs['kill_br'] = Kill_Br

        time_set.attrs['unit'] = 's'
    
    
    return out_name
























def get_all_file_names(path, extension) -> list[tuple[str, str]]:
    '''
    Returns a list of the file names in a directory with a specified extension.

    Parameters: 
                path: str - the path to the directory to get the file names from
                extension: str - the extension of the files to get the names of
    Returns:    
                file_names: list[str] - all files in dir and sub dirs with extension
    Raises:     
                Exception - if the file path does not exist
    '''
    try:

        all_files = os.listdir(path)

    except FileNotFoundError:

        raise Exception(
            'The file path you specified does not exist. Try again.')

    file_names = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                file_names.append((root, file))
    return file_names





def remove_existing(files: list[tuple[str,str]], file_path: str) -> list[tuple[str,str]]:
    '''
    Removes files that already exist in the specified directory.

    Parameters: 
                files: list[tuple[str,str]] - the list of files to check for
    Returns:    
                return_files: list[tuple[str,str]] - the list of files that do not already exist
    '''
    return_files = []
    
    is_path = os.path.isdir(file_path)
    
    if is_path:
    
        files_in_path = os.listdir(file_path)
        
        for file in files:
            
            base = file[1][:-8]
            
            if (base + '.h5') not in files_in_path:
                    
                return_files.append(file)
        
    return return_files






def extract_zip(zip_file_name: str, zip_path: str, out_path: str, out_name: str) -> str:
    '''
    Extract the contents of a zip file to a specified directory. 
    
    Parameters:
                zip_file_name: str - the name of the zip file to extract
                zip_path: str - the path to the zip file
                out_path: str - the path to extract the contents to
                out_name: str - the name of the directory to extract the contents to
    Returns:
                extract_path: str - the path to the extracted contents
    '''
    
    zip_file = zf.ZipFile(zip_path + '/' + zip_file_name, 'r')
    
    zip_file.extractall( out_path + '/' + out_name)
    
    extract_path = out_path + '/' + out_name 
    
    return extract_path





def frame_rate_sampeling(native_rate: float, desired_rate: float, total_frames: int, start_frame: int=0) -> tuple[list[int], float]:
    '''
    Returns a list of the frames to read from a video based on the desired frame rate.

    Parameters: native_rate: float - the native frame rate of the video
                desired_rate: float - the desired frame rate of the video
                total_frames: int - the total number of frames in the video
                start_frame: int - the frame number to start reading from
    Returns:    
                frames_to_read: list[int] - the list of frame numbers to read from the video
    '''
    if native_rate < desired_rate:
        
        period = 1
        
        fps = native_rate
        
    else:
        
        period = int(floor(native_rate / desired_rate))
        
        fps = desired_rate

    frames_to_read = range(start_frame, total_frames, period)

    return frames_to_read, fps




def remove_white(frame: np.ndarray, thresh: float) -> np.ndarray:
    '''
    Removes white from a frame of video.

    Parameters: 
                frame: np.ndarray - the frame to remove white from
                thresh: int - the threshold for white
    Returns:    
                frame: np.ndarray - the frame with white removed
    '''
    
    new_frame = frame.copy()
    
    new_frame[(new_frame[:,:,0] > thresh) & (new_frame[:,:,2] > thresh)] = 0
    
    return new_frame




def filter_high_intensity(frame: np.ndarray, thresh: float) -> np.ndarray:
    """
    Zero out all pixel values above the given threshold.
    Used to suppress saturated bright spots before center detection.

    Args:
        frame: Grayscale numpy array
        thresh: Pixel value threshold — values above this are set to 0

    Returns: numpy array with high-intensity pixels zeroed
    """
    
    new_frame = frame.copy()
    
    new_frame[new_frame > thresh] = 0
    
    return new_frame





def remove_background(frame: np.ndarray, thresh: float) -> np.ndarray:
    '''
    Removes the background from a frame of video.

    Parameters: 
                frame: np.ndarray - the grey scale frame to remove the background from
                thresh: int - the threshold for background
    Returns:    
                frame: np.ndarray - the frame with the background removed
    '''
    
    new_frame = frame.copy()
    
    new_frame[new_frame < thresh] = 0
    
    return new_frame



def remove_and_subtract_background(frame: np.ndarray, thresh: float, rescale: bool=False) -> np.ndarray:
    '''
    Removes the background from a frame of video and subtracts the background from the frame.

    Parameters: 
                frame: np.ndarray - the grey scale frame to remove the background from
                thresh: int - the threshold for background
    Returns:    
                frame: np.ndarray - the frame with the background removed
    '''
    
    new_frame = frame.copy()
    
     #show('new_frame', new_frame)
    
    new_frame[new_frame < thresh] = 0
    
    # show('after <thresh', new_frame)
    
    new_frame[new_frame >= thresh] = new_frame[new_frame >= thresh] - thresh
    
    # show('after >=thresh', new_frame)
    
    if rescale:
        new_frame = new_frame / (255 - thresh)
        # show('after rescale', new_frame1)
        new_frame = (new_frame * 255).astype(np.uint8)
        # show('after rescale2', new_frame, 1)
        
        
    
    return new_frame



def rescale(image: np.ndarray, blank_thresh: int) -> np.ndarray:
    '''
    Rescales an image to 0-255 after removing the background based on a threshold.
    
    Parameters:
                image: np.ndarray - the image to rescale
                blank_thresh: int - the threshold for the background
                
    Returns:
                scaled: np.ndarray - the rescaled image
    
    '''
    
    new_frame = image.copy()
    values = np.reshape(new_frame, (1, -1))
    non_blanks = values[values > blank_thresh]
    new_min = np.min(non_blanks)
    new_max = np.max(non_blanks)
    new_frame[new_frame < new_min] = new_min
    
    s1 = new_frame - new_min
    s2 = new_max - new_min
    s3 = 255/s2
    scaled = s3*s1
    
    return scaled.astype(np.uint8)
    


def translate(frame: np.ndarray, translation: tuple[int, int]) -> np.ndarray:
    '''
    Translates a frame of video by a specified amount.

    Parameters: 
                frame: np.ndarray - the frame to translate
                translation: tuple(int, int) - the amount to translate the frame by (x, y)
    Returns:    
                frame: np.ndarray - the translated frame
    '''

    x, y = translation
    
    new_frame = frame.copy()
    
    new_frame = np.roll(new_frame, x, axis=1)
    new_frame = np.roll(new_frame, y, axis=0)
    
    return new_frame



def find_first_frame(video: cv.VideoCapture) -> np.ndarray:
    '''
    Finds the first frame of a video that is not blank.

    Parameters: 
                video: cv.VideoCapture - the video to find the first frame of
                
    Returns:    
                frame_idx: int - the index of the first frame that is not blank
    '''
    
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))# type: ignore
    
    for i in range(total_frames):
            
        video.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        # frame = frame[:,:,1]
        # frame = filter_high_intensity(frame, 255*.9)
        frame = remove_white(frame, 255*.5)
        # Take just the green
        # show('frame', frame, 1)
        frame = frame[:,:,1]
        
        
        if np.mean(frame) > 15:
            # show('frame', frame)
            break
        #else:
            #print('Frame ' + str(i) + ' is ' + str(np.mean(frame)))
            
            

    return i



def find_center(frame: np.ndarray, scope: tuple[int, int]=0, crop: Optional[tuple[int, int, int, int]]=None) -> tuple[int, int]:
    '''
    Finds the center of a frame of video.

    Parameters: 
                frame: np.ndarray - the frame to find the center of
    Returns:    
                center: tuple(int, int) - the center of the frame (x,y)
    '''

    if len(frame.shape) == 3:
        filtered_frame = remove_white(frame, 255*0.7)
        filtered_frame = filtered_frame[:,:,1]
    else:
        filtered_frame = filter_high_intensity(frame, 255*0.95)
        
    # show('filtered_frame', filtered_frame)
        
    filtered_frame = remove_background(filtered_frame, 255*0.8)
            
    # show('filtered_frame', filtered_frame)
    

    center = find_rss_spot(filtered_frame, scope, image_range=crop)

    
    return center



def find_rss_spot(image: np.ndarray, scope: tuple[int, int]=0, image_range: Optional[tuple[int,int,int,int]] = None) -> tuple[int, int]:
    '''
    Returns the center of the image based on the mirrored residual sum of squares and vertical line max.

    Parameters: 
                image: np.ndarray - the image to find the RSS spot in
                scope: tuple[int, int] - the range of x values to calculate the RSS for
    Returns:    
                center: tuple(int, int) - the center of the image (x,y)
    '''
    in_scope = scope
    if scope == 0:

        scope = [0, image.shape[1]-1]
        
    if image_range is None:
        image_range = [0, 0, 0, 0]
    else:
        # Shift the scope to account for the cropped image
        scope = (scope[0]-image_range[2], scope[1]- image_range[2])
        
    height = image.shape[0]
    width = image.shape[1]
    
    # Check if the scope is to large for the cropped image
    if 0 > scope[0]:# Check left bound
        scope = (0, scope[1])
        
    if image.shape[1] - image_range[3] -  image_range[2] < scope[1]:# Check right bound
        scope = (scope[0], width - image_range[3]- image_range[2] - 1)
        
    
    
            
    cropped_image = image[image_range[0]:height-image_range[1], image_range[2]:width-image_range[3]]
        
    residuals = find_rss(cropped_image, scope)

    # Find min of residuals
    mirror_point = find_residual_min(residuals) 
    # Correct for scope shift
    mirror_point = mirror_point + scope[0]
    
    test_image = cropped_image.copy()
    test_image[:, mirror_point] = 255
    
    # show('image', test_image)

    
    # Approximate the vertical center using line_max
    l_max = find_thick_line_max(cropped_image[:, mirror_point-9:mirror_point+10])

    # Use the line_max to crop to just the center spot 50x20 pixels (y,x)
    y_croped_image = cropped_image[l_max-50:l_max+50, mirror_point-9:mirror_point+10]
    
    # Use the RSS to find the vertical residuals
    y_residuals = find_rss(y_croped_image, axis=1)
    
    # Find the min of the residuals aka the vertical center
    # and correct for the shift caused by the crop
    y_max = find_residual_min(y_residuals) + l_max - 50
    
    # Correct for the cropped image
    y_max = y_max + image_range[0]
    mirror_point = mirror_point + image_range[2]

    return mirror_point, y_max 





def find_rss(image: np.ndarray, scope: tuple[int, int]=0, axis: int=0) -> list[float]:
    '''
    Calculates the mirrored residual sum of squares for each x in scope.

    Parameters: image: np.ndarray - the image to calculate the RSS for
                scope: tuple[int, int] - the range of x values to calculate the RSS for   
    Returns:    residuals: list[float] - the RSS for each x in scope
    '''

    if scope == 0:
        if axis == 0:
            scope = [0, image.shape[1]]
        else:
            scope = [0, image.shape[0]]

    x_marginal = np.sum(image, axis=axis)

    residuals = np.zeros(scope[1] - scope[0])

    for x in range(scope[0], scope[1]):

        lower = np.flip(x_marginal[0:x])
        upper = np.flip(x_marginal[x+1:])

        if lower.shape > upper.shape:

            difference = lower.shape[0] - upper.shape[0]
            upper = np.append(np.zeros(difference), upper)
            unmirrored = np.append(x_marginal, np.zeros(difference))

        elif upper.shape > lower.shape:

            difference = upper.shape[0] - lower.shape[0]
            lower = np.append(lower, np.zeros(difference, dtype=int))
            unmirrored = np.append(np.zeros(difference), x_marginal)

        else:

            unmirrored = x_marginal

        y = np.asarray(x_marginal[x])
        mirror = np.append(np.append(upper, [y]), lower)

        residuals[x - scope[0]] = (1/x_marginal.shape[0]) * \
            np.sum(np.square(unmirrored - mirror))

    return residuals



def find_line_max(vector: np.ndarray, mean=False) -> int:
    '''
    Find the median point of the maximum values in a vector.

    Parameters: 
                vector: np.ndarray - the vector to find the median point of the maximum values for
    Returns:    
                max_point: int - the median or mean point of the maximum values
    '''

    max = np.amax(vector)

    max_points = np.empty(0, dtype=int)

    for x in range(0, vector.shape[0]):
        if vector[x] == max:
            max_points = np.append(max_points, x)
            
    if mean:
        max_point = int(np.floor(np.mean(max_points)))
    else:

        theFloor = np.floor(max_points.shape[0]/2)

        max_point = max_points[int(theFloor)]

    return max_point


def find_thick_line_max(image: np.ndarray) -> int:
    '''
    Find the median point of the maximum values in a vector.

    Parameters: 
                image: np.ndarray - the image to find the mean vertical point of the maximum values for
    Returns:    
                center: int - the mean point of the maximum values
    '''

    frame = image.copy()
    max = np.max(frame)
    
    # Get the indices of the foreground pixels
    foreground_indices = np.argwhere(image == max)
    
    heights = foreground_indices[:,0]
    
    # Calculate the centroid as the mean of the foreground pixel indices
    center = heights.mean(axis=0, dtype=int)

    return int(center)



def show(name: str, frame: np.ndarray, wait: int=0):
    '''
    Displays a frame of video in a window.

    Parameters: 
                name: str - the name of the window
                frame: np.ndarray - the frame to display
                wait: int - the amount of time to wait before closing the window
    '''
    cv.imshow(name, frame)
    cv.waitKey(wait)
    cv.destroyAllWindows()
    return


def remove_output_files(files: list[tuple[str,str]]) -> list[tuple[str,str]]:
    '''
    Removes files that are output files.

    Parameters: 
                files: list[tuple[str,str]] - the list of files to check for
    Returns:    
                return_files: list[tuple[str,str]] - the list of files that are not output files
    '''
    return_files = []
    
    for file in files:
        
        if '_e_' not in file[1]:
            
            return_files.append(file)
        
    return return_files


def calc_median(frame: np.ndarray, channel: int) -> int:
    '''
    Calculates the median of a single channel of a frame of video.

    Parameters: 
                image: np.ndarray - the image to calculate the median of
    Returns:    
                median: int - the median of the vector
    '''
    frame = remove_white(frame, 255*0.5)
    
    
    green_channel = frame[:,:,channel]
    
    g_array = np.reshape(green_channel, (1, -1))
    g_nonblank = g_array[g_array > BLANK_THRESH]
    g_median = np.median(g_nonblank)
    
    return g_median


def calc_bacground_thresh(image, thresh):
    '''
    Calculates the background threshold for a frame of video.

    Parameters: 
                image: np.ndarray - the image to calculate the background threshold for
                thresh: float - the threshold for the background in %
    Returns:    
                background_thresh: float - the background threshold
    '''
    image = remove_white(image, 255*0.5)
    image = image[:,:,1]
    vector = np.reshape(image, (1, -1))
    non_zeros = vector[vector > BLANK_THRESH]
    value = np.percentile(non_zeros, thresh)
    
    return value


def find_residual_min(residuals: np.ndarray, ) -> int:
    '''
    Finds the minimum value in a vector of residuals.

    Parameters: 
                residuals: np.ndarray - the residuals to find the minimum of
    
    Returns:    
                min_point: int - the point of the minimum value
    '''

    rmin = np.amin(residuals)
    min_points = np.empty(0, dtype=int)

    for r in range(0, residuals.shape[0]):

        if residuals[r] == rmin:

            min_points = np.append(min_points, r)

    ordered_points = np.sort(min_points)
    mirror_point = ordered_points[int(
        floor(ordered_points.shape[0]/2))]
    
    return mirror_point


def scale_image(image, Z):
    """
    Scales a 2D array by a factor of Z.

    Parameters:
    image (2D array): The input array to be scaled.
    Z (int): The scaling factor.

    Returns:
    2D array: The scaled image.
    """
    # Repeat elements along the row axis (axis=0)
    scaled_image = np.repeat(image, Z, axis=0)
    # Repeat elements along the column axis (axis=1)
    scaled_image = np.repeat(scaled_image, Z, axis=1)
    
    return scaled_image




def fake_hdr(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    local contrast in a grayscale frame, simulating an HDR effect.

    Args:
        image: Grayscale uint8 numpy array

    Returns: Contrast-enhanced uint8 numpy array
    """
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    hdr_image = clahe.apply(image)
    
    return hdr_image



def fake_hdr_gamma(image, gamma=1.5):
    """
    Apply power-law gamma correction to simulate HDR enhancement.
    Values are normalized to [0, 1], raised to the gamma power, then rescaled.

    Args:
        image: Grayscale uint8 numpy array
        gamma: Gamma exponent — values > 1 darken, values < 1 brighten

    Returns: Gamma-corrected uint8 numpy array
    """
    
    # Apply gamma correction
    gamma_corrected_image = np.power(image / 255.0, gamma)
    gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)
    
    return gamma_corrected_image


def fake_hdr_gamma2(image: np.ndarray, gamma: float=1.5):
    """
    Simulates HDR effect on a grayscale image using gamma correction.

    Parameters:
    image_path (str): The file path to the input grayscale image.
    gamma (float): The gamma value for gamma correction.

    Returns:
    numpy.ndarray: The resulting HDR image.
    """

    # Apply gamma correction
    inverted = np.power(1 - (image / 255.0), gamma)
    non_inverted = np.abs(1 - inverted)
    gamma_corrected_image = (non_inverted * 255).astype(np.uint8)

    return gamma_corrected_image


def fake_hdr_invert(image, sigma=15):
    """
    Simulates HDR effect on a grayscale image using image inversion and desaturation.

    Parameters:
    image_path (str): The file path to the input grayscale image.
    sigma (int): The sigma value for Gaussian blur.

    Returns:
    numpy.ndarray: The resulting HDR image.
    """

    
    # Invert the image
    inverted_image = cv.bitwise_not(image)
    
    # Apply Gaussian Blur to the inverted image
    blurred_image = cv.GaussianBlur(inverted_image, (0, 0), sigma)
    
    # Invert the blurred image
    inverted_blurred_image = cv.bitwise_not(blurred_image)
    
    # Combine the original image with the inverted blurred image
    hdr_image = cv.addWeighted(image, 0.5, inverted_blurred_image, 0.5, 0)

    
    return hdr_image




def scar_removal(image: np.ndarray, passes: int, scope: tuple[int,int,int,int]) -> np.ndarray:
    
    '''
    Removes the blue and red scards from an image.
    
    Parameters:
                image: np.ndarray - the image to remove the scars from
                passes: int - the number of passes to make
                scope: tuple[int,int,int,int] - the scope of the image to remove the scars from
                                                [top, bottom, left, right]
    
    Returns:    
                result: np.ndarray - the image with the scars removed
    '''
    #scope = (153, 262, 363, 252)
    og_image = image.copy()
    
    #crop = (282, 289, 451, 249)
    
    # cropped = image[crop[0]:-crop[1], crop[2]:-crop[3],:]
    # show('cropped', scale_image(cropped,6))
    
    top = scope[0]
    bottom = image.shape[0] - scope[1]
    left = scope[2]
    right = image.shape[1] - scope[3]
    
    blue = og_image[:,:,0]
    green = og_image[:,:,1]
    red = og_image[:,:,2]

    
    blue_coords = np.argwhere(((blue > 10) & (red < 100)) )
    red_coords = np.argwhere(((red > 10) & (blue < 50)) | (red > 50) & (green < 50))
    
    test_blue = np.zeros_like(blue)
    test_red = np.zeros_like(red)
    
    '''
    b_thresh = 50
    blue_test = blue.copy()
    blue_test[(blue_test > b_thresh) & (red < 50)] = 255
    blue_test[(blue_test <= b_thresh) | (red >= 50)] = 0
    show('blue', blue_test)
    
    r_thresh = 50
    red_test = red.copy()
    red_test[(red_test > r_thresh) & (blue < 50)] = 255
    red_test[(red_test <= r_thresh) | (blue >= 50)] = 0
    show('red', red_test)
    '''
    
    for i in range(0, passes):
        for coord in blue_coords:
            y,x = coord
            
            if y < top or y > bottom or x < left or x > right:
                continue
            
            blue[y,x] = 0
            red[y,x] = 0
            green[y,x] = 0
            test_blue[y,x] = 255
            
            # average green value for surrounding pixels
            neighbors = green[y-1:y+2, x-1:x+2]
            non_blank = neighbors[neighbors > 25]
            if non_blank.size == 0:
                continue
            average = np.mean(non_blank)
            green[y,x] = average
            
        
            
            
        
        for coord in red_coords:
            y,x = coord
        
            
            if y < top or y > bottom or x < left or x > right:
                continue
            
            red[y,x] = 0
            blue[y,x] = 0
            green[y,x] = 0
            test_red[y,x] = 255
            
            # average green value for surrounding pixels
            neighbors = green[y-1:y+2, x-1:x+2]
            non_blank = neighbors[neighbors > 25]
            if non_blank.size == 0:
                continue
            average = np.mean(non_blank)
            green[y,x] = average
            
    new_coords = np.zeros_like(blue)
            
    
    for x in range(scope[2], image.shape[1] - scope[3]):
        for y in range(scope[0], image.shape[0] - scope[1]):
            neighbors = green[y-1:y+2, x-1:x+2]
            non_blank = neighbors[neighbors > 25]
            if non_blank.size == 0:
                continue
            average = np.mean(non_blank)
            
            if green[y,x] < average*.75:
                new_coords[y-1:y+2, x-1:x+2] = 1
    
    
    for i in range(0, PASSES):
        for coord in np.argwhere(new_coords):
            y,x = coord
            red[y,x] = 0
            blue[y,x] = 0
            green[y,x] = 0
            
            neighbors = green[y-1:y+2, x-1:x+2]
            non_blank = neighbors[neighbors > 25]
            if non_blank.size == 0:
                continue
            average = np.mean(non_blank)
            green[y,x] = average

    result = np.dstack((blue, green, red))
    
    
    '''
    show('blue', scale_image(test_blue[crop[0]:-crop[1], crop[2]:-crop[3]],6) )
    show('result0', scale_image(result[crop[0]:-crop[1], crop[2]:-crop[3], 0],6))
    
    show('red', scale_image(test_red[crop[0]:-crop[1], crop[2]:-crop[3]],6))
    show('result2', scale_image(result[crop[0]:-crop[1], crop[2]:-crop[3], 2],6))
    
    show('result', scale_image(result[crop[0]:-crop[1], crop[2]:-crop[3], 1],6))
    
    test = result[crop[0]+67:-(crop[1]+4), crop[2]+6:-(crop[3]+29),:]
    
    show('test', scale_image(test,6))
    '''
    
    return result


def invert(frame: np.ndarray) -> np.ndarray:
    '''
    Inverts a frame of video.

    Parameters: 
                frame: np.ndarray - the frame to invert
    Returns:    
                frame: np.ndarray - the inverted frame
    '''
    return cv.bitwise_not(frame)


def sanitize_output_path(path:str, f_name: str) -> tuple[str, str]:
    '''
    Sanitizes the output path by ensuring there is a '/' at the end of the path.
    And remove any extensions from the file name.
    '''
    if path[-1] != '/':
        path = path + '/'
        
    f_name = f_name.split('.')[0]
    
    return path, f_name




if __name__ == "__main__":

    # 1. Find the latest video in captures/
    captures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")

    all_videos = [os.path.join(captures_dir, f) 
                  for f in os.listdir(captures_dir) 
                  if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    if not all_videos:
        raise FileNotFoundError("❌ No video files found in captures/ folder.")

    latest_video = max(all_videos, key=os.path.getmtime)
    print(f"\n📌 Latest video detected: {latest_video}\n")

    # 2. Output folder for pre-processing results
    out_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "results", "pre_processing")
    os.makedirs(out_folder, exist_ok=True)

    # 3. Output name (same as input but .h5 instead of .mp4)
    out_name = os.path.splitext(os.path.basename(latest_video))[0]

    # 4. Run preprocessing
    pre_processing(
        latest_video,
        out_folder,
        out_name,
        FRAME_PERIOD,
        BLANK_THRESH,
        KILL_BR,
        SCAR_REMOVAL,
        SCAR_SCOPE,
        PASSES,
        CROP_EDGES,
        CROP,
        FAKE_HDR,
        HDR_TYPE,
        HDR_RESCALE,
        GAMMA,
        SIGMA,
        BKGRND_CROP,
        BKGRND_RESCALE,
        BKGRND_EACH,
        BKGRND_THRESH,
        TRANSLATE,
        TSHIFT,
        DRIFT,
        D_AMOUNT,
        D_TYPE,
        JUMP_SIZE,
        RSS,
        FIRST_ONLY,
        VERTICAL,
        RSS_SCOPE,
        RSS_CROP
    )
