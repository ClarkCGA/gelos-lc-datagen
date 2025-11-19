import numpy as np
import pandas as pd 
from PIL import Image
import os
import xarray as xr

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

def scale(array: np.array):
    """Scales a numpy array to 0-1 according to maximum value."""
    if array.max() > 1.0:
        array_scaled = array / 4000
    else:
        array_scaled = array * 5

    array_norm = np.clip(array_scaled, 0, 1)
    return array_norm


def s1_norm(linear_data, band='VV'):
    # Convert linear power to dB
    db_data = 10 * np.log10(linear_data)

    # Define fixed clipping ranges for VV and VH bands in dB
    if band == 'VV':
        min_db, max_db = -25, 0
    elif band == 'VH':
        min_db, max_db = -30, -5

    # Clip dB values
    clipped_db = np.clip(db_data, min_db, max_db)

    # Scale clipped values to [0,1]
    scaled_data = (clipped_db - min_db) / (max_db - min_db)

    return scaled_data
    
def create_s1_rgb_composite(s1_array: xr.DataArray):
    """
    Creates an RGB image from a Sentinel-1 xarray.DataArray.
    
    Mapping based on physical scattering properties:
    - Red: VV (Co-Pol) -> Sensitive to surface roughness.
    - Green: VH (Cross-Pol) -> Sensitive to volume scattering.
    - Blue: VV/VH Ratio -> Helps differentiate smooth surfaces like water.

    :param s1_array: xr.DataArray with 'vv' and 'vh' bands.
    :return: 8-bit RGB image as a numpy array.
    """
    vv = s1_array.isel(band=0).values.astype(float)
    vh = s1_array.isel(band=1).values.astype(float)

    # Normalize each polarization to enhance contrast
    vv_norm = s1_norm(vv, 'VV')
    vh_norm = s1_norm(vh, 'VH')

    # Calculate the ratio for the blue channel. Add epsilon to avoid division by zero.
    ratio = vv_norm / (vh_norm + 1e-6)

    # Stack bands into an RGB image
    # R: VV -> Red indicates strong surface scattering (rough surfaces)
    # G: VH -> Green indicates strong volume scattering (vegetation, buildings)
    # B: Ratio -> Blue indicates low volume scattering (water, roads)
    rgb = np.dstack((vv_norm, vh_norm, ratio))
    
    # Convert to 8-bit integer for image display
    rgb_8bit = (rgb * 255).astype(np.uint8)
    
    return rgb_8bit

def save_thumbnails(array, root_path, index):
    '''
    Read array, process and save png thumbnails for Sentinel 1 SAR data.
    :param array: xr.DataArray
    :param root_path: directory to save thumbnails
    :return
    '''

    for i, dt in enumerate(array.time.values):
        ts = pd.to_datetime(str(dt)) 
        filename = f"{array.name}_{index:06}_{i}_{ts.strftime('%Y%m%d')}.png"
        file_path = os.path.join(root_path, filename)

        if array.name == 'sentinel_1':
            rgb_8bit = create_s1_rgb_composite(array.isel(time=i))
        else:
            blue  = array.isel(time = i,band=1).values.astype(float)
            green = array.isel(time = i,band=2).values.astype(float)
            red   = array.isel(time = i,band=3).values.astype(float)
    
            # mask and normalize
            blue = scale(blue)
            green = scale(green)
            red   = scale(red)

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
    dts = ','.join(dts)
    return dts

