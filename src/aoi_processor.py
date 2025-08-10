from src.downloader import Downloader
from src.chip_generator import ChipGenerator
import pystac
import pandas as pd
from .utils.search import search_s2_scenes, search_s1_scenes, search_landsat_scenes, search_annual_scene, count_unique_dates
from .utils.stack import stack_data, stack_dem_data, stack_lc_data, pystac_itemcollection_to_gdf
from functools import reduce


class AOI_Processor:
    """Responsible for processing one AOI, managed by Downloader"""
    def __init__(self, aoi_index, aoi, downloader: Downloader):
        self.downloader = downloader
        self.aoi_index = aoi_index
        self.aoi = aoi
        self.aoi_bounds = aoi['geometry'].bounds

    def process_aoi(self):
        """Process one AOI by searching and stacking data sources"""
        print(f"\nProcessing AOI at index {self.aoi_index}")

        s2_items = pystac.item_collection.ItemCollection([])

        for date_range in self.downloader.config.sentinel_2.time_ranges:
            print(f"Searching Sentinel-2 scenes for {date_range}")
            s2_items_season = search_s2_scenes(
                self.aoi,
                date_range,
                self.downloader.catalog,
                self.downloader.config.sentinel_2.collection,
                self.downloader.config.sentinel_2.nodata_pixel_percentage,
                self.downloader.config.sentinel_2.cloud_cover,
            )
            s2_items += s2_items_season

        if len(s2_items)<4:
            raise ValueError(f"S2 scenes missing")

        try:
            epsg = s2_items[0].properties["proj:epsg"]
        except:
            epsg = int(s2_items[0].properties["proj:code"].split(":")[-1])
        bbox_latlon = s2_items[0].bbox

        s1_items = pystac.item_collection.ItemCollection([])
        landsat_items = pystac.item_collection.ItemCollection([])

        for s2_item, date_range in zip(s2_items, self.downloader.config.sentinel_2.time_ranges):
            center_datetime = s2_item.datetime
            print(f"searching sentinel-1 and landsat scenes close to {center_datetime} within {date_range}")
            s1_item = search_s1_scenes(
                self.aoi,
                center_datetime,
                date_range,
                self.downloader.config.sentinel_1.delta_days,
                self.downloader.catalog,
                self.downloader.config.sentinel_1.collection,
            )
            s1_items += s1_item
            landsat_item = search_landsat_scenes(
                self.aoi,
                center_datetime,
                date_range,
                self.downloader.config.landsat.delta_days,
                self.downloader.catalog,
                self.downloader.config.landsat.collection,
                self.downloader.config.landsat.platforms,
                self.downloader.config.landsat.cloud_cover,
            )
            landsat_items += landsat_item

        if count_unique_dates(landsat_items) < 4:
            raise ValueError(f"landsat scenes missing")

        if count_unique_dates(s1_items) < 4:
            raise ValueError(f"s1 scenes missing")
                
        print("searching land cover data...")
        lc_items = search_annual_scene(
            self.aoi,
            self.downloader.config.land_cover.year,
            self.downloader.catalog,
            self.downloader.config.land_cover.collection,
        )
        if not lc_items:
            raise ValueError(f"lc data missing")

        print("searching dem data...")
        dem_items = search_annual_scene(
            self.aoi,
            self.downloader.config.dem.year,
            self.downloader.catalog,
            self.downloader.config.dem.collection,
        )
        if not dem_items:
            raise ValueError(f"dem data missing")

            # first, get area of overlap of all item bboxes
        itemcollections = [s2_items, s1_items, landsat_items, lc_items, dem_items]
        bbox_gdf = pd.concat([pystac_itemcollection_to_gdf(items) for items in itemcollections])
        combined_geoms = bbox_gdf.groupby('collection')['geometry'].apply(lambda x: x.unary_union)
        overlap = reduce(lambda x, y: x.intersection(y), combined_geoms)
        overlap_bounds = overlap.bounds

        print("stacking landsat data...")
        self.landsat_stack = stack_data(
            landsat_items,
            "landsat",
            self.downloader.config.landsat.native_crs,
            self.downloader.config.landsat.resolution,
            self.downloader.config.landsat.bands,
            self.downloader.config.landsat.cloud_band,
            epsg,
            overlap_bounds,
            bbox_is_latlon = True
        )

        overlap_bbox = self.landsat_stack.rio.bounds()

        print("stacking sentinel-2 data...")
        self.s2_stack = stack_data(
            s2_items,
            "sentinel_2",
            self.downloader.config.sentinel_2.native_crs,
            self.downloader.config.sentinel_2.resolution,
            self.downloader.config.sentinel_2.bands,
            self.downloader.config.sentinel_2.cloud_band,
            epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        print("stacking dem data...")
        self.dem_stack = stack_dem_data(
            dem_items, 
            self.downloader.config.dem.resolution, 
            epsg, 
            overlap_bbox
        )

        print("stacking land cover data...")
        self.lc_stack = stack_lc_data(
            lc_items, 
            self.downloader.config.land_cover.resolution, 
            epsg, 
            overlap_bbox
        )

        print("stacking sentinel-1 data...")
        self.s1_stack = stack_data(
            s1_items,
            "sentinel_1",
            self.downloader.config.sentinel_1.native_crs,
            self.downloader.config.sentinel_1.resolution,
            self.downloader.config.sentinel_1.bands,
            None,  # No cloud band for Sentinel-1
            epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        chip_generator = ChipGenerator(self)
        chip_generator.generate()
        


