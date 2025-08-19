import numpy as np
import pandas as pd 
from PIL import Image
import os

def mask_nodata(band, nodata_values=(-999,)):
    '''
    Mask nodata to nan
    :param band
    :param nodata_values:nodata values in chips is -999
    :return band
    '''
    band = band.astype(float)
    for val in nodata_values:
        band[band == val] = np.nan
    return band

def normalize(band):
    '''
    Normalize a band to 0-1 range(float)
    :param band (ndarray)
    return normalize band (ndarray); when max equals min, returns zeros.
    '''
    if np.nanmean(band) >= 4000:
        band = band / 6000
    elif np.max(band) < 1:
        band = band * 3
    else:
        band = band / 4000
    band = np.clip(band, None, 1)

    return band


def save_thumbnails(array, root_path, index):
    '''
    Read array, process and save png thumbnails.
    :param array: xr.DataArray
    :param root_path: directory to save thumbnails
    :return
    '''

    for i, dt in enumerate(array.time.values):
        ts = pd.to_datetime(str(dt)) 
        filename = f"{array.name}_{index:06}_{i}_{ts.strftime('%Y%m%d')}.png"
        file_path = os.path.join(root_path, filename)
        
        blue  = array.isel(time = i,band=1).values.astype(float)
        green = array.isel(time = i,band=2).values.astype(float)
        red   = array.isel(time = i,band=3).values.astype(float)
    
        # mask and normalize
        blue = normalize(mask_nodata(blue))
        green = normalize(mask_nodata(green))
        red   = normalize(mask_nodata(red))

        # stack and convert to 8-bit
        rgb = np.dstack((red, green, blue))
        rgb_8bit = (rgb * 255).astype(np.uint8)
    
        pil_img = Image.fromarray(rgb_8bit)
        pil_img.save(file_path, format="PNG")

   
def save_multitemporal_chips(array, root_path, index):
    dts = []
    for i, dt in enumerate(array.time.values):
        ts = pd.to_datetime(str(dt)) 
        dest_path = f"{root_path}/{array.name}_{index:06}_{i}_{ts.strftime('%Y%m%d')}.tif"
        array.sel(time = dt).squeeze().rio.to_raster(dest_path)
        dts.append(ts.strftime('%Y%m%d'))
    return dts

