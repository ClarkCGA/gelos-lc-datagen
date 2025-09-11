import pdb
import xarray as xr
import rioxarray as rxr
from src.gelos_config import GELOSConfig
from src.chip_generator import ChipGenerator
import pystac
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape
from rasterio.features import rasterize

from .utils.search import search_s2_scenes, search_s1_scenes, search_landsat_scenes, search_annual_scene, count_unique_dates, get_landsat_wrs_path
from .utils.stack import stack_data, stack_dem_data, stack_land_cover_data, pystac_itemcollection_to_gdf
from .utils.array import select_burnt_chips
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

    def rasterize_aoi(self, aoi, stack):
        """
        Rasterize the AOI polygon into a burn mask
        """
        aoi_gdf = gpd.GeoDataFrame(
            {"geometry": [shape(aoi['geometry'])]},
            crs="EPSG:4326" # check that CRS matches across all Platforms -- Landsat, S1 and S2
        )
        
        aoi_proj = aoi_gdf.to_crs(stack.rio.crs)
        
        burn_mask = rasterize(
            [(mapping(aoi_proj['geometry'].iloc[0]), 1)],
            out_shape=(stack.sizes['y'], stack.sizes['x']), # specify the platform not just sentinel_2
            transform=stack.rio.transform(),  # specify the platform not just sentinel_2
            fill=0,
            dtype='uint8'
        )
        
        burn_mask_da = xr.DataArray(
            burn_mask,
            coords={"y": stack["y"], "x": stack["x"]},
            dims=("y", "x")
        )
        return burn_mask_da

    def stack_aoi(self, time_ranges=None):
        """Process one AOI by searching and stacking data sources"""
        print(f"\nProcessing AOI at index {self.aoi_index}")
        if not self.config.dataset.fire:
            time_ranges = self.config.sentinel_2.time_ranges
        else:
            time_ranges = time_ranges
        skip_pipeline = False

        s2_items = pystac.item_collection.ItemCollection([])
        for date_range in time_ranges:
            print(f"Searching Sentinel-2 scenes for {date_range}")
            s2_items_season, self.s2_scene_id = search_s2_scenes(
                self.aoi.geometry,
                date_range,
                self.catalog,
                self.config.sentinel_2.collection,
                self.config.sentinel_2.nodata_pixel_percentage,
                self.config.sentinel_2.cloud_cover,
                self.s2_scene_id
            )
            if not s2_items_season:
                print(f"Skipping enire loop: no S2 scenes for {date_range}")
                skip_pipeline = True
                break
                # raise ValueError("s2 scenes missing")
            s2_items += s2_items_season
        
        if skip_pipeline:
            print("skipping the rest of the processing pipeline due to a missing prerequisite Scene")
            return None

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

  

        for s2_item, date_range in zip(s2_items, time_ranges): 
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
            if not s1_item:  #and self.s1_relative_orbit is None:
                # raise ValueError("s1 scenes missing")
                print(f"Skipping enire loop: no S1 scenes for {date_range}")
                skip_pipeline = True
                break
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
                # raise ValueError("landsat scenes missing")
                print(f"Skipping enire loop: no Landsat scenes for {date_range}")
                skip_pipeline = True
                break

            landsat_items += landsat_item
        
        if skip_pipeline:
            print("skipping the rest of the processing pipeline due to a missing S1 Scene")
            return None

        if count_unique_dates(landsat_items) < 4:
            raise ValueError(f"landsat scenes missing")

        if count_unique_dates(s1_items) < 4:
            raise ValueError(f"s1 scenes missing")
            
        if not self.config.dataset.fire:
            print("searching land cover data...")
            land_cover_items = search_annual_scene(
                self.s2_bbox,
                self.config.land_cover.year,
                self.catalog,
                self.config.land_cover.collection,
            )
            if not land_cover_items:
                raise ValueError(f"land_cover data missing")
        
        if not self.config.dataset.fire:
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
        if not self.config.dataset.fire:
            self.itemcollections = {
                "sentinel_2": s2_items,
                "sentinel_1": s1_items,
                "landsat": landsat_items,
                "land_cover": land_cover_items,
                "dem": dem_items
            }
        else:
            self.itemcollections = {
                "sentinel_1": s1_items,
                "sentinel_2": s2_items,
                "landsat": landsat_items,
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

        if not self.config.dataset.fire:
            print("stacking dem data...")
            self.stacks['dem'] = stack_dem_data(
                dem_items, 
                self.config.dem.native_crs,
                self.config.dem.resolution, 
                self.epsg, 
                overlap_bbox,
                bbox_is_latlon=False
            )

        if not self.config.dataset.fire:
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

        return self.stacks

    def process_aoi(self):
        if getattr(self.config, "dataset", None) and getattr(self.config.dataset, "fire", False):
            return self.generate_time_series()
        else:
            chip_generator = ChipGenerator(self)
            return chip_generator.generate_from_aoi()
    
    def generate_time_series(self, event_date_ranges, control_date_ranges, metadata_df):
        event_stacks = self.stack_aoi(event_date_ranges)
        if event_stacks is None:
            print(f"Incomplete event stacks — skipping AOI: {self.aoi_index}")
            return None
        for key in event_stacks: 
            platform_event_stack = event_stacks[key]
            print(f"Extracting burn-rich chip areas from event stack for {key}")

            burn_mask_q1 = self.rasterize_aoi(self.aoi, platform_event_stack[0][0])
            event_chip_slices = select_burnt_chips(platform_event_stack[0][0], burn_mask_q1, self.config)
            print(f"Found {len(event_chip_slices)} burn-rich chip areas for {key}")

            for chip_id_num, (_, chip_slice) in enumerate(event_chip_slices):
                time_series_type = "event"
                chip_generator = ChipGenerator(self)
                metadata_df = chip_generator.generate_fire_chips(platform_event_stack,
                                                                key,
                                                                chip_slice,
                                                                chip_id_num,
                                                                self.aoi,
                                                                self.aoi_index,
                                                                self.config,
                                                                time_series_type,
                                                                metadata_df)
                print(metadata_df)
                # control chips
                for idx, time_range in enumerate(control_date_ranges):
                        time_series_type = "control"
                        ctrl_stacks = self.stack_aoi(time_range)
                        if ctrl_stacks is None:
                            print(f"Incomplete control stacks — skipping AOI: {self.aoi_index}")
                            continue
                        for key in ctrl_stacks: 
                            platform_ctrl_stacks = ctrl_stacks[key]
                            chip_generator = ChipGenerator(self)
                            metadata_df = chip_generator.generate_fire_chips(platform_ctrl_stacks,
                                                                            key,
                                                                            chip_slice,
                                                                            chip_id_num,
                                                                            self.aoi,
                                                                            self.aoi_index,
                                                                            self.config,
                                                                            time_series_type,
                                                                            metadata_df)

        self.chip_metadata_df = metadata_df
        return self.chip_metadata_df
        
    def _persist_progress(self, aoi_index, aoi_status):
        self.aoi_gdf.loc[aoi_index, "status"] = aoi_status
        # self.aoi_gdf.to_file(self.aoi_path, driver="GeoJSON")
        self.chip_metadata_df.to_csv(self.chip_metadata_path, index=False)


