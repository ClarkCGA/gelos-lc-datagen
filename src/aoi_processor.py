import pdb
import xarray as xr
import rioxarray as rxr
from src.gelos_config import GELOSConfig
from src.chip_generator import ChipGenerator
import pystac
import pandas as pd
import geopandas as gpd

from .utils.search import search_s2_scenes, search_s1_scenes, search_landsat_scenes, search_annual_scene, count_unique_dates, get_landsat_wrs_path
from .utils.stack import stack_data, stack_dem_data, stack_land_cover_data, pystac_itemcollection_to_gdf
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
        self.s2_scene_id = None
        self.landsat_wrs_path = None
        self.s1_relative_orbit = None


    def process_aoi(self):
        """Process one AOI by searching and stacking data sources"""
        print(f"\nProcessing AOI at index {self.aoi_index}")

        s2_items = pystac.item_collection.ItemCollection([])
        for date_range in self.config.sentinel_2.time_ranges:
            print(f"Searching Sentinel-2 scenes for {date_range}")
            s2_items_season, self.s2_scene_id = search_s2_scenes(
                self.aoi.geometry,
                date_range,
                self.catalog,
                self.config.sentinel_2.collection,
                self.config.sentinel_2.nodata_pixel_percentage,
                self.config.sentinel_2.cloud_cover,
                self.s2_scene_id,
            )
            if not s2_items_season:
                raise ValueError("s2 scenes missing")
            s2_items += s2_items_season

        if len(s2_items)<4:
            raise ValueError(f"S2 scenes missing")

        try:
            self.epsg = s2_items[0].properties["proj:epsg"]
        except:
            self.epsg = int(s2_items[0].properties["proj:code"].split(":")[-1])
        self.s2_bbox = s2_items[0].geometry
        
        self.landsat_wrs_path = get_landsat_wrs_path(self.s2_bbox)

        s1_items = pystac.item_collection.ItemCollection([])
        landsat_items = pystac.item_collection.ItemCollection([])
        
        

        for s2_item, date_range in zip(s2_items, self.config.sentinel_2.time_ranges):
            center_datetime = s2_item.datetime
            print(f"searching sentinel_1 and landsat scenes close to {center_datetime} within {date_range}")
            s1_item, self.s1_relative_orbit = search_s1_scenes(
                self.s2_bbox,
                center_datetime,
                date_range,
                self.config.sentinel_1.delta_days,
                self.catalog,
                self.config.sentinel_1.collection,
                self.s1_relative_orbit,
            )
            if not s1_item:
                raise ValueError("s1 scenes missing")
            s1_items += s1_item

            landsat_item = search_landsat_scenes(
                self.s2_bbox,
                center_datetime,
                date_range,
                self.config.landsat.delta_days,
                self.catalog,
                self.config.landsat.collection,
                self.config.landsat.platforms,
                self.config.landsat.cloud_cover,
                self.landsat_wrs_path,
            )
            if not landsat_item:
                raise ValueError("landsat scenes missing")
            landsat_items += landsat_item

        if count_unique_dates(landsat_items) < 4:
            raise ValueError(f"landsat scenes missing")

        if count_unique_dates(s1_items) < 4:
            raise ValueError(f"s1 scenes missing")
                
        print("searching land cover data...")
        land_cover_items = search_annual_scene(
            self.s2_bbox,
            self.config.land_cover.year,
            self.catalog,
            self.config.land_cover.collection,
        )
        if not land_cover_items:
            raise ValueError(f"land_cover data missing")

        print("searching dem data...")
        dem_items = search_annual_scene(
            self.s2_bbox,
            self.config.dem.year,
            self.catalog,
            self.config.dem.collection,
        )
        if not dem_items:
            raise ValueError(f"dem data missing")

            # first, get area of overlap of all item bboxes
        self.itemcollections = {
            "sentinel_2": s2_items,
            "sentinel_1": s1_items,
            "landsat": landsat_items,
            "land_cover": land_cover_items,
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

        print("stacking landsat data...")
        self.stacks['landsat'] = stack_data(
            landsat_items,
            "landsat",
            self.config.landsat.native_crs,
            self.config.landsat.resolution,
            self.config.landsat.bands,
            self.config.landsat.cloud_band,
            self.epsg,
            self.overlap_bounds,
            bbox_is_latlon = True
        )
        
        overlap_bbox = self.stacks['landsat'].rio.bounds()

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
        self.stacks['land_cover'] = stack_land_cover_data(
            land_cover_items, 
            self.config.land_cover.native_crs,
            self.config.land_cover.resolution, 
            self.epsg, 
            overlap_bbox,
            bbox_is_latlon=False
        )


        print("stacking sentinel_1 data...")
        self.stacks['sentinel_1'] = stack_data(
            s1_items,
            "sentinel_1",
            self.config.sentinel_1.native_crs,
            self.config.sentinel_1.resolution,
            self.config.sentinel_1.bands,
            None,  # No cloud band for Sentinel-1
            self.epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        print("stacking sentinel_2 data...")
        self.stacks['sentinel_2'] = stack_data(
            s2_items,
            "sentinel_2",
            self.config.sentinel_2.native_crs,
            self.config.sentinel_2.resolution,
            self.config.sentinel_2.bands,
            self.config.sentinel_2.cloud_band,
            self.epsg,
            overlap_bbox,
            bbox_is_latlon=False
        )

        chip_generator = ChipGenerator(self)
        return chip_generator.generate_from_aoi()
        


