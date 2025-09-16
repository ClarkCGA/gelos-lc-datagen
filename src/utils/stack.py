import stackstac
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape

def pystac_itemcollection_to_gdf(item_collection):
    geometries = []
    properties = []
    for item in item_collection:
        # Create box geometry from bbox
        geometries.append(shape(item.geometry))
        props = item.properties.copy()
        props['collection'] = item.collection_id
        properties.append(props)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
    
    return gdf

def stack_data(
    items,
    platform,
    native_crs,
    resolution,
    bands,
    cloud_band=None,
    epsg=None,
    bbox=None,
    bbox_is_latlon=True,
):

    if bbox is None:
        bbox = items[0].bbox
    
    if native_crs:
        try:
            epsg = items[0].properties["proj:epsg"]
        except:
            epsg = int(items[0].properties["proj:code"].split(":")[-1])
    
    bounds_param = "bounds_latlon" if bbox_is_latlon else "bounds"
    if not bbox_is_latlon:
        bbox = adjust_bbox_to_resolution(bbox, resolution)
    bounds_kwargs = {bounds_param: bbox}

    stack = stackstac.stack(
        items,
        assets=bands,
        epsg=epsg,
        resolution=resolution,
        fill_value=np.nan,
       **bounds_kwargs
    )
    if platform in ['sentinel_2', 'landsat']:
        stack = mask_cloudy_pixels(stack, platform)

    # if platform not in ['sentinel_2']:
    #     quarter_times = stack.time.groupby("time.quarter").first()
    #     stack = stack.groupby("time.quarter").first(skipna=True)
    #     stack['quarter'] = quarter_times.values
    #     stack = stack.rename({'quarter': 'time'})
 
    if len(stack.band) != len(bands):
        raise ValueError(f"{platform} unexpected number of bands")
    if platform in ['sentinel_2', 'landsat']:
        stack = stack.drop_sel(band=cloud_band)
    
    return stack

def stack_dem_data(items, native_crs, resolution, epsg=None, bbox=None, bbox_is_latlon=False):
    if not items:
        print("No dem data found.")
        return None
    if native_crs:
        try:
            epsg = items[0].properties["proj:epsg"]
        except:
            epsg = int(items[0].properties["proj:code"].split(":")[-1])
    bounds_param = "bounds_latlon" if bbox_is_latlon else "bounds"
    if not bbox_is_latlon:
        bbox = adjust_bbox_to_resolution(bbox, resolution)
    bounds_kwargs = {bounds_param: bbox}
    stack = stackstac.stack(
        items,
        epsg=epsg,
        resolution=resolution,
       **bounds_kwargs
    ).mean(dim="time").squeeze()
    
    return stack

def stack_land_cover_data(items, native_crs, resolution, epsg, bbox, bbox_is_latlon=False):
    if not items:
        print("No Land Cover data found.")
        return None
    if native_crs:
        try:
            epsg = items[0].properties["proj:epsg"]
        except:
            epsg = int(items[0].properties["proj:code"].split(":")[-1])
    bounds_param = "bounds_latlon" if bbox_is_latlon else "bounds"
    if not bbox_is_latlon:
        bbox = adjust_bbox_to_resolution(bbox, resolution)
    bounds_kwargs = {bounds_param: bbox}
    stack = stackstac.stack(
        items,
        epsg=epsg,
        resolution=resolution,
       **bounds_kwargs
    ).mean(dim="time").squeeze()
    return stack

def adjust_bbox_to_resolution(bbox, resolution):
    '''Adjusts bbox from rioxarray.rio output so stackstac snaps to grid correctly'''
    # this function gets the bbox which intersects the centers of all pixels we want from stackstac
    r = resolution / 2
    minx, miny, maxx, maxy = bbox
    bbox_adjusted = (minx + r, miny, maxx, maxy - r)
    return bbox_adjusted
   
def mask_cloudy_pixels(stack, platform):
    if platform == "landsat":
        qa = stack.sel(band="qa_pixel").astype("uint16")
        
        # Define bitmask for cloud-related flags
        mask_bitfields = [1, 2, 3, 4]
        bitmask = sum(1 << b for b in mask_bitfields)
        clear_mask = (qa & bitmask) == 0
        
        # Broadcast the clear_mask to match stack shape
        clear_mask = clear_mask.broadcast_like(stack)
        
        # Apply the mask
        stack = stack.where(clear_mask)
    elif platform == "sentinel_2":
        scl = stack.sel(band="SCL")
        cloud_mask = scl.isin([3, 8, 9, 10])
        stack = stack.where(~cloud_mask)
    else:
        print(f"attempting to cloud mask invalid platform: {platform}")
        return None
    return stack


