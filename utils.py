import numpy as np
import tifffile as tiff
import scipy.misc
import rioxarray

def gen_chips(s2_array, lc_array, index):

    lc_path = f"/home/benchuser/data/lc_{index:04}.tif"
    s2_path = f"/home/benchuser/data/s2_{index:04}.tif"
    
    s2_array.rio.to_raster(s2_path)
    lc_array.rio.to_raster(lc_path)
