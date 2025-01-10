import numpy as np
import tifffile as tiff
import scipy.misc
import rioxarray

def gen_chips(s2_array, lc_array, index):

    lc_path = f"/home/benchuser/data/lc_{index:05}.tif"
    s2_path = f"/home/benchuser/data/s2_{index:05}.tif"
    try:
        s2_array.rio.to_raster(s2_path)
        lc_array.rio.to_raster(lc_path)
        gen_status = True
    except:
        gen_status = False


    return gen_status
