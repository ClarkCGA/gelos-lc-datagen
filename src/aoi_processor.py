import pdb
import xarray as xr
import rioxarray as rxr
from src.gelos_config import GELOSConfig
from src.chip_generator import ChipGenerator
import pystac
import pandas as pd
import geopandas as gpd

from .utils.search import search_s2l2a_scenes, search_s1rtc_scenes, search_lc2l2_scenes, search_annual_scene, count_unique_dates, get_lc2l2_wrs_path
from .utils.stack import stack_data, stack_dem_data, stack_lulc_data, pystac_itemcollection_to_gdf
from functools import reduce

class AOI_Processor:
    """Responsible for processing one AOI, managed by Downloader"""
    def __init__(self, aoi_index, aoi, chip_index, working_directory, catalog, config: GELOSConfig):
        self.config = config
        self.catalog = catalog
        self.aoi_index = aoi_index
        self.aoi = aoi
        self.chip_index = chip_index
        self.working_directory = working_directory
        self.stacks = {}
        self.s2l2a_scene_id = None
        self.lc2l2_wrs_path = None
        self.s1rtc_relative_orbit = None


    def process_aoi(self):
        """Process one AOI by searching and stacking data sources"""
        print(f"\nProcessing AOI at index {self.aoi_index}")

        s2l2a_items = pystac.item_collection.ItemCollection([])
        for date_range in self.config.s2l2a.time_ranges:
            print(f"Searching Sentinel-2 scenes for {date_range}")
            s2l2a_items_season, self.s2l2a_scene_id = search_s2l2a_scenes(
                self.aoi.geometry,
                date_range,
                self.catalog,
                self.config.s2l2a.collection,
                self.config.s2l2a.nodata_pixel_percentage,
                self.config.s2l2a.cloud_cover,
                self.s2l2a_scene_id,
            )
            if not s2l2a_items_season:
                raise ValueError("s2l2a scenes missing")
            s2l2a_items += s2l2a_items_season

        if len(s2l2a_items)<4:
            raise ValueError(f"s2l2a scenes missing")

        try:
            self.epsg = s2l2a_items[0].properties["proj:epsg"]
        except:
            self.epsg = int(s2l2a_items[0].properties["proj:code"].split(":")[-1])
        self.s2l2a_bbox = s2l2a_items[0].geometry
        
        self.lc2l2_wrs_path = get_lc2l2_wrs_path(self.s2l2a_bbox)

        s1rtc_items = pystac.item_collection.ItemCollection([])
        lc2l2_items = pystac.item_collection.ItemCollection([])
        
        for s2l2a_item, date_range in zip(s2l2a_items, self.config.s2l2a.time_ranges):
            center_datetime = s2l2a_item.datetime
            print(f"searching s1rtc and lc2l2 scenes close to {center_datetime} within {date_range}")
            s1rtc_item, self.s1rtc_relative_orbit = search_s1rtc_scenes(
                self.s2l2a_bbox,
                center_datetime,
                date_range,
                self.config.s1rtc.delta_days,
                self.catalog,
                self.config.s1rtc.collection,
                self.s1rtc_relative_orbit,
            )
            if not s1rtc_item:
                raise ValueError("s1rtc scenes missing")
            s1rtc_items += s1rtc_item

            lc2l2_item = search_lc2l2_scenes(
                self.s2l2a_bbox,
                center_datetime,
                date_range,
                self.config.lc2l2.delta_days,
                self.catalog,
                self.config.lc2l2.collection,
                self.config.lc2l2.platforms,
                self.config.lc2l2.cloud_cover,
                self.lc2l2_wrs_path,
            )
            if not lc2l2_item:
                raise ValueError("lc2l2 scenes missing")
            lc2l2_items += lc2l2_item

        if count_unique_dates(lc2l2_items) < 4:
            raise ValueError(f"lc2l2 scenes missing")

        if count_unique_dates(s1rtc_items) < 4:
            raise ValueError(f"s1rtc scenes missing")
                
        print("searching lulc data...")
        lulc_items = search_annual_scene(
            self.s2l2a_bbox,
            self.config.lulc.year,
            self.catalog,
            self.config.lulc.collection,
        )
        if not lulc_items:
            raise ValueError(f"lulc data missing")

        print("searching dem data...")
        dem_items = search_annual_scene(
            self.s2l2a_bbox,
            self.config.dem.year,
            self.catalog,
            self.config.dem.collection,
        )
        if not dem_items:
            raise ValueError(f"dem data missing")

            # first, get area of overlap of all item bboxes
        self.itemcollections = {
            "s2l2a": s2l2a_items,
            "s1rtc": s1rtc_items,
            "lc2l2": lc2l2_items,
            "lulc": lulc_items,
            "dem": dem_items
        }
        bbox_gdf = pd.concat([pystac_itemcollection_to_gdf(items) for items in self.itemcollections.values()])
        bbox_gdf.to_file(self.working_directory / f"{self.aoi_index}_stac_items.json", driver="GeoJSON")
        
        # group scenes which share a collection and date
        # bbox_gdf['date'] = bbox_gdf.apply(lambda x: x.datetime.date())
        bbox_gdf['datetime'] = pd.to_datetime(bbox_gdf['datetime'], format="mixed")
        bbox_gdf['date'] = bbox_gdf['datetime'].dt.date
        combined_geoms = bbox_gdf.groupby(['collection', 'date'])['geometry'].apply(lambda x: x.unary_union)

        # get the intersection of all data sources as the bounding box for stacks
        overlap = reduce(lambda x, y: x.intersection(y), combined_geoms)
        self.overlap_bounds = overlap.bounds
        
        self.scene_ids = {
            f"{platform}_scene_ids": ','.join([item.id for item in items]) for platform, items in self.itemcollections.items()
        }
        
        print("stacking lc2l2 data...")
        self.stacks['lc2l2'] = stack_data(
            lc2l2_items,
            "lc2l2",
            self.config.lc2l2.native_crs,
            self.config.lc2l2.resolution,
            self.config.lc2l2.bands,
            self.config.lc2l2.cloud_band,
            self.epsg,
            self.overlap_bounds,
            bbox_is_latlon = True
        )
        
        overlap_bbox = self.stacks['lc2l2'].rio.bounds()

        print("stacking dem data...")
        self.stacks['dem'] = stack_dem_data(
            dem_items, 
            self.config.dem.native_crs,
            self.config.dem.resolution, 
            self.epsg, 
            overlap_bbox,
            bbox_is_latlon=False
        )

        print("stacking land cover data...")
        self.stacks['lulc'] = stack_lulc_data(
            lulc_items, 
            self.config.lulc.native_crs,
            self.config.lulc.resolution, 
            self.epsg, 
            overlap_bbox,
            bbox_is_latlon=False
        )


        print("stacking s1rtc data...")
        self.stacks['s1rtc'] = stack_data(
            s1rtc_items,
            "s1rtc",
            self.config.s1rtc.native_crs,
            self.config.s1rtc.resolution,
            self.config.s1rtc.bands,
            None,  # No cloud band for Sentinel-1
            self.epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        print("stacking s2l2a data...")
        self.stacks['s2l2a'] = stack_data(
            s2l2a_items,
            "s2l2a",
            self.config.s2l2a.native_crs,
            self.config.s2l2a.resolution,
            self.config.s2l2a.bands,
            self.config.s2l2a.cloud_band,
            self.epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        chip_generator = ChipGenerator(self)
        chip_gdf = chip_generator.generate_from_aoi()
        return chip_gdf
        


