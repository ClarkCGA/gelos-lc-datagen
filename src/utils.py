import numpy as np
import stackstac 
import pandas as pd 
from datetime import datetime, timedelta
import pystac
from PIL import Image
import os
def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)

def missing_values(array, chip_size, sample_size):
    """Check if the given S2/LC stacked array contains NaN values over the central sample area."""
    array_trimmed = array.isel(x = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2)), 
                               y = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2))
                              )
    has_nan = array_trimmed.isnull().any()
    zero_array = array_trimmed.max() == 0
    missing_values = has_nan or zero_array
    return missing_values
    
def save_multitemporal_chips(array, root_path, index):
    dts = []
    for i, dt in enumerate(array.time.values):
        ts = pd.to_datetime(str(dt)) 
        dest_path = f"{root_path}/{array.name}_{index:06}_{i}_{ts.strftime('%Y%m%d')}.tif"
        array.sel(time = dt).squeeze().rio.to_raster(dest_path)
        dts.append(ts.strftime('%Y%m%d'))
    return dts

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
        
        blue  = array.isel(time = i,band=0).values.astype(float)
        green = array.isel(time = i,band=1).values.astype(float)
        red   = array.isel(time = i,band=2).values.astype(float)
    
        # mask and normalize
        blue = normalize(mask_nodata(blue))
        green = normalize(mask_nodata(green))
        red   = normalize(mask_nodata(red))

        # stack and convert to 8-bit
        rgb = np.dstack((red, green, blue))
        rgb_8bit = (rgb * 255).astype(np.uint8)
    
        pil_img = Image.fromarray(rgb_8bit)
        pil_img.save(file_path, format="PNG")

def gen_chips(s2_array, s1_array, landsat_array, lc_array, dem_array, index, root_path):

    lc_path = f"{root_path}/lc_{index:06}.tif"
    dem_path = f"{root_path}/dem_{index:06}.tif"
    s2_dts, s1_dts, landsat_dts = [], [], []
    try:
        s2_dts = save_multitemporal_chips(s2_array, root_path, index)
        s1_dts = save_multitemporal_chips(s1_array, root_path, index)
        landsat_dts = save_multitemporal_chips(landsat_array, root_path, index)

        save_thumbnails(s2_array, root_path, index)
        save_thumbnails(landsat_array, root_path, index)
    
        lc_array.rio.to_raster(lc_path)
        dem_array.rio.to_raster(dem_path)
        gen_status = True
    except Exception as e:
        print(e)
        gen_status = False

    return gen_status, s2_dts, s1_dts, landsat_dts

def get_continent(point, continents_path):
    """Returns the continent name for a given shapely Point (in (lon, lat) order)."""
    continents = gpd.read_file(continents_path)

    for _, row in continents.iterrows():
        if row['geometry'].contains(point):
            return row['CONTINENT']
    return "Unknown"