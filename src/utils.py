import numpy as np
import stackstac 
import pandas as pd 
from datetime import datetime, timedelta
import pystac
from PIL import Image
import os

def search_s2_scenes(aoi, date_range, catalog, config):
    """
    Searches for Sentinel-2 scenes within the AOI and date range.
    Adds debugging info to identify missing data issues.
    """
    s2_search = catalog.search(
        collections=config["sentinel_2"]["collection"],
        bbox=aoi['geometry'].bounds, 
        datetime=date_range,
        query = [f"s2:nodata_pixel_percentage<{config["sentinel_2"]["nodata_pixel_percentage"]}",
                 f"eo:cloud_cover<{config["sentinel_2"]["cloud_cover"]}"
                ],
        sortby=["+properties.eo:cloud_cover"],
        max_items=1,
    )
    items = s2_search.item_collection()
    return items

def search_s1_scenes(aoi, s2_datetime, catalog, config):
    """
    Searches for Sentinel-1 scenes within the AOI that are closest to the specified datetime.
    """
    delta = timedelta(days=6)
    start = s2_datetime - delta
    end = s2_datetime + delta
    datetime_range = f"{start.isoformat(timespec='seconds').replace('+00:00', 'Z')}/{end.isoformat(timespec='seconds').replace('+00:00', 'Z')}"
    s1_search = catalog.search(
        collections=config["sentinel_1"]["collection"],
        bbox=aoi['geometry'].bounds, 
        datetime=datetime_range,
        max_items=10,
    )
    # Convert to list and sort by closeness to s2_datetime
    s1_collection = s1_search.item_collection()
    if len(s1_collection) == 0:
        return s1_collection
    sorted_items = sorted(s1_collection.items, key=lambda item: abs(item.datetime - s2_datetime))
    s1_collection = pystac.item_collection.ItemCollection([sorted_items[0]])

    return s1_collection
    
def mask_cloudy_pixels(item_stack, platform):
    if platform == "landsat":
        qa = item_stack.sel(band="qa_pixel").astype("uint16")
        
        # Define bitmask for cloud-related flags
        mask_bitfields = [1, 2, 3, 4]
        bitmask = sum(1 << b for b in mask_bitfields)
        clear_mask = (qa & bitmask) == 0
        
        # Broadcast the clear_mask to match stack shape
        clear_mask = clear_mask.broadcast_like(item_stack)
        
        # Apply the mask
        stack = item_stack.where(clear_mask)
    elif platform == "sentinel_2":
        scl = item_stack.sel(band="SCL")
        cloud_mask = scl.isin([3, 8, 9, 10])
        item_stack = item_stack.where(~cloud_mask)
    else:
        print(f"attempting to cloud mask invalid platform: {platform}")
        return None
    return item_stack

def search_landsat_scenes(aoi, s2_datetime, catalog, config):
    """
    Searches for Landsat scenes within the AOI that are within 6 days of of the specified datetime and returns the least cloudy scene
    """
    delta = timedelta(days=6)
    start = s2_datetime - delta
    end = s2_datetime + delta
    datetime_range = f"{start.isoformat(timespec='seconds').replace('+00:00', 'Z')}/{end.isoformat(timespec='seconds').replace('+00:00', 'Z')}"
    landsat_search = catalog.search(
        collections=config["landsat"]["collection"],
        bbox=aoi['geometry'].bounds, 
        datetime=datetime_range,
        query = {
            "platform": {"in": config["landsat"]["platforms"]},
            "eo:cloud_cover": {"lt": config["landsat"]["cloud_cover"]},
        },
        sortby=["+properties.eo:cloud_cover"],
        max_items=1
    )
    # Convert to list and sort by closeness to s2_datetime
    landsat_collection = landsat_search.item_collection()

    return landsat_collection

def search_dem_scene(bbox, catalog, config):
    lc_search = catalog.search(
        collections=config["dem"]["collection"],
        bbox=bbox,
        datetime=config["dem"]["year"]
    )
    items = lc_search.item_collection()
    return items

def search_lc_scene(bbox, catalog, config):
    lc_search = catalog.search(
        collections=config["land_cover"]["collection"],
        bbox=bbox,
        datetime=config["land_cover"]["year"]
    )
    items = lc_search.item_collection()
    return items

def stack_data(items, platform, config, epsg=None, bbox=None):
    if bbox == None:
        bbox = items[0].bbox
    
    if config[platform]["native_crs"] == True:
        try:
            epsg = items[0].properties["proj:epsg"]
        except:
            epsg = int(items[0].properties["proj:code"].split(":")[-1])
            
    try:
        item_stack = stackstac.stack(
            items,
            assets=config[platform]["bands"],
            epsg=epsg,
            resolution=config[platform]["resolution"],
            fill_value=np.nan,
            bounds_latlon = bbox
        )
        if platform in ['sentinel_2', 'landsat']:
            item_stack = mask_cloudy_pixels(item_stack, platform)
            item_stack = item_stack.drop_sel(band=config[platform]['cloud_band'])
            item_stack = item_stack.chunk(chunks={"band": len(config[platform]["bands"]) - 1, "x": -1, "y": "auto"})
        else: 
            item_stack = item_stack.chunk(chunks={"band": len(config[platform]["bands"]), "x": -1, "y": "auto"})
            
        return item_stack
    except Exception as e:
        print(f"Error stacking {platform} data: {e}")
        return None

def stack_dem_data(items, config, epsg=None, bbox=None):
    if not items:
        print("No dem data found.")
        return None
    if config["dem"]["native_crs"] == True:
        try:
            epsg = items[0].properties["proj:epsg"]
        except:
            epsg = int(items[0].properties["proj:code"].split(":")[-1])
    try:
        stacked_data = stackstac.stack(
            items,
            epsg=epsg,
            resolution=config["dem"]["resolution"],
            bounds_latlon=bbox,
        ).median("time", skipna=True).squeeze()
        stacked_data = stacked_data.chunk(chunks={"x": -1, "y": "auto"})
        return stacked_data
        
    except Exception as e:
        print(f"Error stacking dem data: {e}")
        return None

def stack_lc_data(lc_items, config, epsg, bbox):
    if not lc_items:
        print("No Land Cover data found.")
        return None
    try:
        stacked_data = stackstac.stack(
            lc_items,
            epsg=epsg,
            resolution=config["sentinel_2"]["resolution"],
            bounds_latlon=bbox,
        ).median("time", skipna=True).squeeze()
        stacked_data = stacked_data.chunk(chunks={"x": -1, "y": "auto"})
        return stacked_data
        
    except Exception as e:
        print(f"Error stacking Land Cover data: {e}")
        return None

def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)

def missing_values(array, chip_size, sample_size):
    """Check if the given S2/LC stacked array contains NaN values over the central sample area."""
    array_trimmed = array.isel(x = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2)), 
                               y = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2))
                              )
    has_nan = array_trimmed.isnull().any()
    return has_nan
    
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