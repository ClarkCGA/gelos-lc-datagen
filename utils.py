import numpy as np
import tifffile as tiff
import scipy.misc
import rioxarray

def get_count(window, axis=None, **kwargs):
    window = window.compute()
    ret = np.zeros((window.shape[0], window.shape[2]))
    for i in range(0, window.shape[0]):
        for j in range(0, window.shape[2]):
            ret[i, j] = (np.unique(window[i, :, j, :])).shape[0]
    return ret


def save_array_as_tif(array, file_path):
    """
    Save a NumPy array as a TIFF file.

    Parameters:
        array (numpy.ndarray): The NumPy array to save.
        file_path (str): Path to save the TIFF file, including the .tif extension.

    Example:
        save_array_as_tif(my_array, "output_file.tif")
    """
    try:
        tiff.imwrite(file_path, array)
    except Exception as e:
        print(f"An error occurred while saving the array: {e}")


def gen_chips(s2_array, lc_array, index):

    
    lc_path = f"/home/benchuser/data/lc_{index:04}.tif"
    s2_path = f"/home/benchuser/data/s2_{index:04}.tif"
    
    s2_array.rio.to_raster(s2_path)
    lc_array.rio.to_raster(lc_path)
