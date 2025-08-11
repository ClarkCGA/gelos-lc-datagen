import numpy as np
from datetime import timedelta, datetime
import pystac
 
def search_s2_scenes(aoi, overall_date_range, catalog, collection, nodata_pixel_percentage, cloud_cover):
    """
    Searches for Sentinel-2 scenes within the AOI and date range.
    Adds debugging info to identify missing data issues.
    """
    search = catalog.search(
        collections=collection,
        intersects = aoi,
        datetime=overall_date_range,
        query = [f"s2:nodata_pixel_percentage<{nodata_pixel_percentage}",
                 f"eo:cloud_cover<{cloud_cover}"
                ],
        sortby=["+properties.eo:cloud_cover"],
        max_items=1,
    )
    items = search.item_collection()
    return items

def search_s1_scenes(aoi, center_datetime, overall_date_range, delta_days, catalog, collection):
    """
    Searches for Sentinel-1 scenes within the AOI that are closest to the specified datetime.
    """
    datetime_range = get_clipped_datetime_range(center_datetime, overall_date_range, delta_days)

    search = catalog.search(
        collections = collection,
        intersects = aoi,
        datetime = datetime_range,
        max_items = 50,
    )
    all_items = search.item_collection()
    if not all_items:
        return pystac.ItemCollection([])

    # Find the scene closest to the center datetime
    closest_item = min(all_items, key=lambda item: abs(item.datetime - center_datetime))
    best_date = closest_item.datetime.date()

    # Filter for all scenes on that same date
    scenes_on_best_date = [
        item for item in all_items if item.datetime.date() == best_date
    ]

    return pystac.ItemCollection(scenes_on_best_date)
 
 
 
def search_landsat_scenes(aoi, center_datetime, overall_date_range, delta_days, catalog, collection, platforms, cloud_cover):
    """
    Searches for Landsat scenes within the AOI. Finds the least cloudy scene
    and returns all scenes from that same date for compositing.
    """
    datetime_range = get_clipped_datetime_range(center_datetime, overall_date_range, delta_days)
    search = catalog.search(
        collections = collection,
        intersects = aoi,
        datetime = datetime_range,
        query = {
            "platform": {"in": platforms},
            "eo:cloud_cover": {"lt": cloud_cover},
        },
        sortby = ["+properties.eo:cloud_cover"],
        max_items = 50 
    )

    all_items = search.get_all_items()
    if not all_items:
        return pystac.ItemCollection([])

    # Find the scene closest to the center datetime
    best_item = all_items[0]
    best_date = best_item.datetime.date()

    # Filter for all scenes on that same date
    scenes_on_best_date = [
        item for item in all_items if item.datetime.date() == best_date
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