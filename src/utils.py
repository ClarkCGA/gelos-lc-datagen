import numpy as np
import stackstac 
import pandas as pd 

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


def search_lc_scene(bbox, catalog, config):
    lc_search = catalog.search(
        collections=config["land_cover"]["collection"],
        bbox=bbox,
        datetime=config["land_cover"]["year"]
    )
    items = lc_search.item_collection()
    return items

def stack_s2_data(s2_items, config):
    try:
        epsg = s2_items[0].properties["proj:epsg"]
    except:
        epsg = int(s2_items[0].properties["proj:code"].split(":")[-1])
        
    try:
        s2_stack = stackstac.stack(
            s2_items,
            assets=config["sentinel_2"]["bands"],
            epsg=epsg,
            resolution=config["sentinel_2"]["resolution"],
            fill_value=np.nan,
            bounds_latlon = s2_items[0].bbox
        )
        # s2_stack_resampled = s2_stack.median("time", skipna=True)
        s2_stack = s2_stack.chunk(chunks={"band": len(config["sentinel_2"]["bands"]), "x": -1, "y": "auto"})

        return s2_stack
    except Exception as e:
        print(f"Error stacking Sentinel-2 data: {e}")
        return None


def stack_lc_data(lc_items, epsg, s2_bbox, config):
    if not lc_items:
        print("No Land Cover data found.")
        return None
    try:
        stacked_data = stackstac.stack(
            lc_items,
            epsg=epsg,
            resolution=config["sentinel_2"]["resolution"],
            bounds_latlon=s2_bbox,
        ).median("time", skipna=True).squeeze()
        stacked_data = stacked_data.chunk(chunks={"x": -1, "y": "auto"})
        return stacked_data
        
    except Exception as e:
        print(f"Error stacking Land Cover data: {e}")
        return None

def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)

def missing_values(array, chip_size, sample_size):
    """Check if the given S2/LC stacked array contains NaN values over the cenrtal sample area."""
    array_trimmed = array.isel(x = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2)), 
                               y = slice(int((chip_size - sample_size) / 2), int((chip_size + sample_size) / 2))
                              )
    has_nan = array_trimmed.isnull().any()
    return has_nan
    
    
def gen_chips(s2_array, lc_array, index):

    lc_path = f"/home/benchuser/data/lc_{index:06}.tif"
    dts = []
    try:
        for dt in s2_array.time.values:
            ts = pd.to_datetime(str(dt)) 
            s2_path = f"/home/benchuser/data/s2_{index:06}_{ts.strftime('%Y%m%d')}.tif"
            s2_array.sel(time = dt).squeeze().rio.to_raster(s2_path)
            dts.append(ts.strftime('%Y%m%d'))
        lc_array.rio.to_raster(lc_path)
        gen_status = True
    except:
        gen_status = False


    return gen_status, dts