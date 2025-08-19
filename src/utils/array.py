import numpy as np
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
    zero_array = array_trimmed.max() == 0
    missing_values = has_nan or zero_array
    return missing_values
 
def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)