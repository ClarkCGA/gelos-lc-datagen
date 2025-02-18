import numpy as np
import stackstac 

def search_s2_scenes(aoi, date_range, catalog, config):
    """
    Searches for Sentinel-2 scenes within the AOI and date range.
    Adds debugging info to identify missing data issues.
    """
    # catalog = pystac_client.Client.open(
    #     "https://planetarycomputer.microsoft.com/api/stac/v1",
    #     modifier=planetary_computer.sign_inplace,
    # )
    s2_search = catalog.search(
        collections=config["sentinel_2"]["collection"],
        bbox=aoi['geometry'].bounds, 
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": config["sentinel_2"]["cloud_cover"]}},
        sortby=["+properties.eo:cloud_cover"],
        max_items=1,
    )
    items = s2_search.item_collection()
    return items


def search_lc_scene(bbox, catalog, config):
    # catalog = pystac_client.Client.open(
    #     "https://planetarycomputer.microsoft.com/api/stac/v1",
    #     modifier=planetary_computer.sign_inplace,
    # )
    lc_search = catalog.search(
        collections=config["land_cover"]["collection"],
        bbox=bbox,
        datetime=config["land_cover"]["year"]
    )
    items = lc_search.item_collection()
    return items

def stack_s2_data(s2_items, config):
    print("\nChecking available assets in Sentinel-2 items...")
    valid_bands = [band for band in config["sentinel_2"]["bands"] if all(band in item.assets for item in s2_items)]
    try:
        s2_stack = stackstac.stack(
            s2_items,
            assets=valid_bands,
            # epsg=s2_items[0].properties["proj:epsg"],
            epsg=int(s2_items[0].properties["proj:code"].split(":")[-1]),
            resolution=config["sentinel_2"]["resolution"],
            fill_value=np.nan,
            bounds_latlon = s2_items[0].bbox
        )
        # s2_stack_resampled = s2_stack.median("time", skipna=True)
        s2_stack = s2_stack.chunk(chunks={"band": len(valid_bands), "x": -1, "y": "auto"})
        
        print(f"Stacked Sentinel-2 bands: {list(s2_stack.coords['band'].values)}")
        print(f"Number of time steps: {len(s2_stack.time)}")
        return s2_stack
    except Exception as e:
        print(f"Error stacking Sentinel-2 data: {e}")
        return None


def stack_lc_data(lc_items, s2_epsg, s2_bbox, config):
    if not lc_items:
        print("No Land Cover data found.")
        return None
    try:
        print("Stacking Land Cover images...")
        stacked_data = stackstac.stack(
            lc_items,
            dtype=np.ubyte,
            fill_value=255,
            sortby_date=False,
            epsg=s2_epsg,
            resolution=config["sentinel_2"]["resolution"],
            bounds_latlon=s2_bbox,
        ).squeeze()
        stacked_data = stacked_data.chunk(chunks={"x": -1, "y": "auto"})
        print("Stacked LC data shape:", stacked_data.shape)
        #print(f"Chunk sizes: {stacked_data.chunks}")   # Uncomment for big chunk size 
        return stacked_data
        
    except Exception as e:
        print(f"Error stacking Land Cover data: {e}")
        return None

def unique_class(window, axis=None, **kwargs):
    return np.all(window == window[0, 0], axis=axis)

def missing_values(array):
    """Check if the given Dask DataArray contains NaN values and print only when necessary."""
    has_nan = array.isnull().any().compute()
    if has_nan:
        print("Warning: Missing values detected in the chip!")
    return has_nan
    
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
