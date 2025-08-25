import numpy as np
import pandas as pd
import pdb
from datetime import timedelta, datetime
import pystac
import geopandas as gpd
from shapely.geometry import shape

landsat_wrs_path = '/home/benchuser/data/WRS2_descending_0.zip'
landsat_wrs_url = 'https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip'

try:
    landsat_wrs_gdf = gpd.read_file(landsat_wrs_path).to_crs(3857)
except:
    landsat_wrs_gdf = gpd.read_file(landsat_wrs_url).to_crs(3857)
 
def get_landsat_wrs_path(aoi):
    aoi = gpd.GeoSeries([shape(aoi)], crs=4326).to_crs(3857)
    intersecting_gdf = landsat_wrs_gdf[landsat_wrs_gdf.intersects(aoi.iloc[0])].copy()
    intersecting_gdf['intersection_area'] = intersecting_gdf.geometry.intersection(aoi.iloc[0]).area
    sorted_gdf = intersecting_gdf.sort_values(by='intersection_area', ascending=False)
    best_footprint = sorted_gdf.iloc[0]

    return int(best_footprint['PATH'])
    
def search_s2_scenes(aoi, overall_date_range, catalog, collection, nodata_pixel_percentage, cloud_cover, s2_mgrs_tile):
    """
    Searches for Sentinel-2 scenes within the AOI and date range.
    When passed a Sentinel-2 scene ID, will only return chips matching this scene ID
    """
    query = [
        f"s2:nodata_pixel_percentage<{nodata_pixel_percentage}",
        f"eo:cloud_cover<{cloud_cover}"
    ]
    if s2_mgrs_tile:
        query.append(f"s2:mgrs_tile={s2_mgrs_tile}")
    search = catalog.search(
        collections=collection,
        intersects = aoi,
        query = query,
        datetime=overall_date_range,
        sortby=["+properties.eo:cloud_cover"],
        max_items=1,
    )

    items = search.item_collection()
    if not items:
        return items, s2_mgrs_tile
    s2_mgrs_tile = items[0].properties['s2:mgrs_tile']
    return items, s2_mgrs_tile

def search_s1_scenes(aoi, center_datetime, overall_date_range, delta_days, catalog, collection, relative_orbit):
    """
    Searches for Sentinel-1 scenes within the AOI that are closest to the specified datetime.
    """
    datetime_range = get_clipped_datetime_range(center_datetime, overall_date_range, delta_days)

    if not relative_orbit:
        orbit_search = catalog.search(
            collections = collection,
            intersects = aoi,
            datetime = datetime_range
        )
        orbit_search_items_gdf = gpd.GeoDataFrame.from_features(orbit_search.item_collection().to_dict(), crs=4326).to_crs(3857)
        aoi_gs = gpd.GeoSeries([shape(aoi)], crs=4326).to_crs(3857)
        orbit_search_orbits_gdf = orbit_search_items_gdf.dissolve(by='sat:relative_orbit')
        intersecting_gdf = orbit_search_orbits_gdf[orbit_search_orbits_gdf.intersects(aoi_gs.iloc[0])].copy()
        intersecting_gdf['intersection_area'] = intersecting_gdf.geometry.intersection(aoi_gs.iloc[0]).area
        intersecting_gdf['distance_to_center_datetime'] = (pd.to_datetime(intersecting_gdf['datetime']) - center_datetime).abs()
        sorted_gdf = intersecting_gdf.sort_values(by=['intersection_area', 'distance_to_center_datetime'], ascending=[False, True])
        best_footprint = sorted_gdf.iloc[0]
        relative_orbit = best_footprint.name

    search = catalog.search(
        collections = collection,
        intersects = aoi,
        query = [f'sat:relative_orbit={relative_orbit}'],
        datetime = datetime_range,
        max_items = 50,
    )
    items = search.item_collection()
    if not items:
        return items, relative_orbit

    # Find the scene closest to the center datetime
    closest_item = min(items, key=lambda item: abs(item.datetime - center_datetime))
    best_date = closest_item.datetime.date()

    # Filter for all scenes on that same date
    scenes_on_best_date = [
        item for item in items if item.datetime.date() == best_date
    ]

    return pystac.ItemCollection(scenes_on_best_date), relative_orbit
 
 
 
def search_landsat_scenes(aoi, center_datetime, overall_date_range, delta_days, catalog, collection, platforms, cloud_cover, landsat_wrs_path):
    """
    Searches for Landsat scenes within the AOI. Finds the least cloudy scene
    and returns all scenes from that same date for compositing.
    """
    datetime_range = get_clipped_datetime_range(center_datetime, overall_date_range, delta_days)
    query = {
        "platform": {"in": platforms},
        "eo:cloud_cover": {"lt": cloud_cover},
    }
    if landsat_wrs_path:
        query["landsat:wrs_path"] = {"eq": str(landsat_wrs_path).zfill(3)}
       
    search = catalog.search(
        collections = collection,
        intersects = aoi,
        datetime = datetime_range,
        query = query,
        sortby = ["+properties.eo:cloud_cover"],
        max_items = 50 
    )

    items = search.item_collection()
    if not items:
        return pystac.ItemCollection([])
   
    # Get the date of the scene with the least cloud cover
    best_item = items[0]
    best_date = best_item.datetime.date()

    # Filter for all scenes on that same date
    scenes_on_best_date = [
        item for item in items if item.datetime.date() == best_date
    ]
    
    return pystac.ItemCollection(scenes_on_best_date)

def search_annual_scene(aoi, year, catalog, collection):
    """Search for annual data such as Landsat and DEM"""
    search = catalog.search(
        collections=collection,
        intersects = aoi,
        datetime=year
    )
    items = search.item_collection()
    return items

def get_clipped_datetime_range(center_datetime, overall_date_range, delta_days):
    """
    Creates a datetime range string centered around a specific datetime,
    clipped by an overall date range.
    """
    delta = timedelta(days=delta_days)
    start = center_datetime - delta
    end = center_datetime + delta

    # Clip the search window to the overall date_range
    range_start_str, range_end_str = overall_date_range.split('/')
    range_start_str += 'T00:00:00Z'
    range_end_str += 'T23:59:59Z'
    range_start = datetime.fromisoformat(range_start_str.replace('Z', '+00:00'))
    range_end = datetime.fromisoformat(range_end_str.replace('Z', '+00:00'))

    start = max(start, range_start)
    end = min(end, range_end)

    return f"{start.isoformat(timespec='seconds').replace('+00:00', 'Z')}/{end.isoformat(timespec='seconds').replace('+00:00', 'Z')}"

def count_unique_dates(item_collection: pystac.ItemCollection) -> int:
    """Counts the number of unique dates in an item collection."""
    if not item_collection:
        return 0
    dates = {item.datetime.date() for item in item_collection}
    return len(dates)