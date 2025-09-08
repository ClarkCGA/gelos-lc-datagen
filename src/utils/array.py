from datetime import datetime
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, box
import geopandas as gpd
    
def process_array( 
            stack, 
            epsg: int,
            coords: tuple[float, float],
            array_name: str,
            chip_size: int,
            sample_size: int,
            resolution: int,
            fill_na: bool = True,
            na_value: int = -999,
            dtype = str,
            ):

    x, y = coords
    # get dimensions of chip in pixels
    sample_size = int(sample_size / resolution)
    chip_size = int(chip_size / resolution)
    
    # get indices of stack for the area of the chip
    min_x_index = (x) * sample_size - int((chip_size - sample_size)/2)
    max_x_index = (x + 1) * sample_size + int((chip_size - sample_size)/2)
    min_y_index = (y) * sample_size - int((chip_size - sample_size)/2)
    max_y_index = (y + 1) * sample_size + int((chip_size - sample_size)/2)
    x_indices = slice(min_x_index, max_x_index)
    y_indices = slice(min_y_index, max_y_index)    

    # get stack at indices of the valid sample area
    array = stack.isel(x = x_indices, y = y_indices)
    
    # write the crs
    array.rio.write_crs(f"epsg:{epsg}", inplace=True)
    
    # get values from the valid sample area 
    array = array.where((array.x >= stack.x[(x) * sample_size]) &
                              (array.x < stack.x[(x + 1) * sample_size]) & 
                              (array.y <= stack.y[(y) * sample_size]) &
                              (array.y > stack.y[(y + 1) * sample_size])
                             )

    # fill na values
    if fill_na:
        array = array.fillna(na_value)
        array = array.rio.write_nodata(na_value)

    # cast to dtype and rename
    array = array.astype(np.dtype(dtype))
    array = array.rename(array_name)
    
    # raise an exception if any values are missing
    if missing_values(array, chip_size, sample_size):
        raise ValueError(f"{array_name} missing values")
    
    # Create a GeoSeries from the array's bounding box with its native CRS
    native_footprint = gpd.GeoSeries([box(*array.rio.bounds())], crs=array.rio.crs)
    
    # Reproject the GeoSeries to EPSG:4326 and get the geometry
    footprint = native_footprint.to_crs("EPSG:4326").iloc[0]
    
    return array, footprint.wkt

def missing_values(array, chip_size, sample_size):
    """Check if the given S2/LC stacked array contains NaN values over the central sample area."""
    array_trimmed = array.isel(x = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2)), 
                               y = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2))
                              )
    has_nan = array_trimmed.isnull().any()
    all_zero_row = (array_trimmed == 0).all(dim='y').any()
    all_zero_col = (array_trimmed == 0).all(dim='x').any()
    missing_values = has_nan or all_zero_row or all_zero_col
    return missing_values
 
def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)

def harmonize_to_old(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    if "time" not in data.dims:
        # static composite --> nothing to do
        return data
    if "time" not in data.dims:
        if "time" in data:
            data = data.expand_dims("time")  # convert scalar to 1D dimension
        else:
            raise ValueError("Variable 'time' not found in dataset.")
                
    data = data.set_index(time="time")

    cutoff = datetime(2022, 1, 25)
    offset = 1000
    bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
        
    return xr.concat([old, new], dim="time")


def select_burnt_chips(stack, burn_mask, config):
    """Return list[(chip_stack, (y0,y1,x0,x1))] filtered by pct-burn."""
    chip_slices = []
    for y0 in range(0, burn_mask.shape[0] - config.chips.chip_size + 1,
                config.chips.chip_size):
        for x0 in range(0, burn_mask.shape[1] - config.chips.chip_size + 1,
                    config.chips.chip_size):
            window = burn_mask[y0:y0 + config.chips.chip_size,
                            x0:x0 + config.chips.chip_size]
            if window.mean() >= 0.30:
                chip_slices.append((y0, y0 + config.chips.chip_size,
                                    x0, x0 + config.chips.chip_size))

    chips = [
        (stack.isel(y=slice(y0, y1), x=slice(x0, x1), drop=False),
            (y0, y1, x0, x1))
            for y0, y1, x0, x1 in chip_slices
        ]
    return chips

def extract_quarterly_fire_chips(stack, chip_slice):
    y0, y1, x0, x1 = chip_slice
    chip_quarter_data = []
    for quarter_idx, stack in enumerate(stack):
        if stack is None:
            break
        chip = stack.isel(y=slice(y0, y1), x=slice(x0, x1), drop=False)
        try:
            chip = chip.compute()
        except Exception:
            break
        if chip.shape[2:] != (224, 224):
            break
        if missing_values(chip, 224, 224):
            break
        chip_quarter_data.append((chip, quarter_idx))
    return chip_quarter_data
