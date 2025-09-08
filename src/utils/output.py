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

# def normalize(band):
#     '''
#     Normalize a band to 0-1 range(float)
#     :param band (ndarray)
#     return normalize band (ndarray); when max equals min, returns zeros.
#     '''
#     if np.nanmean(band) >= 4000:
#         band = band / 6000
#     elif np.max(band) < 1:
#         band = band * 3
#     else:
#         band = band / 4000
#     band = np.clip(band, None, 1)

#     return band

def normalize(array):
    """Normalizes a numpy array to 0-1, stretching to 2nd and 98th percentiles."""
    # Mask out invalid values (NaNs) for percentile calculation
    valid_pixels = array[~np.isnan(array)]
    if valid_pixels.size == 0:
        return np.zeros_like(array) # Return an all-zero array if no valid data
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    # Clip to prevent extreme values from dominating the image
    array_norm = np.clip((array - p2) / (p98 - p2), 0, 1)
    # Fill NaNs with 0 for visualization
    return np.nan_to_num(array_norm)


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
    vv_norm = normalize(vv)
    vh_norm = normalize(vh)

    # Calculate the ratio for the blue channel. Add epsilon to avoid division by zero.
    ratio = vv / (vh + 1e-6)
    ratio_norm = normalize(ratio)

    # Stack bands into an RGB image
    # R: VV -> Red indicates strong surface scattering (rough surfaces)
    # G: VH -> Green indicates strong volume scattering (vegetation, buildings)
    # B: Ratio -> Blue indicates low volume scattering (water, roads)
    rgb = np.dstack((vv_norm, vh_norm, ratio_norm))
    
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
            rgb_8bit = create_s1_rgb_composite(array.iself(time=i))
        else:
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

def save_fire_chips(stack, aoi_index, aoi, chip_index, crop_idx, time_series_type, metadata_df, epsg, out_path):
    for dt in stack.time.values:
        print(f"Processing chip ID {chip_index} for {time_series_type} date {dt}")
        ts = pd.to_datetime(str(dt))
        if os.path.exists(out_path):
            print(f"Skipping chip ID {chip_index}_{crop_idx} for chip {chip_index} date: {ts.strftime('%Y%m%d')} — file already exists")
            continue
        print(f"Saving chip ID {chip_index}_{crop_idx} for chip {chip_index} date: {ts.strftime('%Y%m%d')}")
        stack.sel(time=dt).squeeze().rio.to_raster(out_path)
        metadata_df = pd.concat([pd.DataFrame([[f"{chip_index:08}",
                                                    aoi_index, 
                                                    ts.strftime('%Y%m%d'),
                                                    f"{time_series_type}",
                                                    f"{aoi["source"]}",
                                                    stack.name,
                                                    stack.x[int(len(stack.x)/2)].data,
                                                    stack.y[int(len(stack.y)/2)].data,
                                                    epsg,
                                                    f"{aoi["pre_date"]}",
                                                    f"{aoi["post_date"]}"]
                                                ],
                                                columns=metadata_df.columns
                                            ),
                                    metadata_df],
                                    ignore_index=True
                                )
    return metadata_df