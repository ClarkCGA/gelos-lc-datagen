import os
import pystac_client
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
import planetary_computer
import pandas as pd
import geopandas as gpd
from dask.distributed import Client, LocalCluster
import logging
from pathlib import Path
import shutil

from src.gelos_config import GELOSConfig
from src.aoi_processor import AOI_Processor
from src.utils.search import get_fire_date_ranges

class Downloader:
    """This class handles data selection and download for GELOS."""
    def __init__(self, config: GELOSConfig):
        self.config = config    
        self.working_directory = Path(self.config.directory.working) / self.config.dataset.version

        # start dask cluster
        self.cluster = LocalCluster(silence_logs=logging.ERROR)
        self.client = Client(self.cluster)

        # set retry policy for pystac catalog client
        retry = Retry(
            total=10, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
        )
        
        # initialize pystac client with retry policy
        stac_api_io = StacApiIO(max_retries=retry)
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io
        )
        
        # handle the case where the script is continuing an existing download operation
        if (self.working_directory / f'chip_metadata.csv').exists():
            if self.config.dataset.fire:
                self.aoi_path = (self.working_directory / 'fire_events_aoi.csv')
            else:
                self.aoi_path = (self.working_directory / 'aoi_metadata.geojson')
            self.aoi_gdf = gpd.read_file(self.aoi_path)
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'
            self.chip_metadata_df = pd.read_csv(self.chip_metadata_path)
            # drop aoi which already have chips generated
            self.aoi_gdf = self.aoi_gdf[self.aoi_gdf.index > self.chip_metadata_df['aoi_index'].max()]
            self.aoi_processing_gdf = self.aoi_gdf
            self.chip_index = self.chip_metadata_df['chip_index'].max() + 1
        
        # handle the case where the script is starting a new download operation for fire events
        elif self.config.dataset.fire and (self.working_directory / f'fire_events_aoi.csv').exists():
            self.aoi_path = (self.working_directory / 'fire_events_aoi.csv')
            self.aoi_gdf = gpd.read_file(self.aoi_path)
            self.aoi_processing_gdf = self.aoi_gdf
            self.chip_metadata_df = pd.DataFrame(columns=[ # fire chip metadata
                    "chip_index",
                    "aoi_index",
                    "date",
                    "type",
                    "source",
                    "platform",
                    "x_center",
                    "y_center",
                    "epsg",
                    "pre_date",
                    "post_date"
                    ])
            self.chip_index = 0
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'

        # handle the case where the script is starting a new download operation
        else:
            aoi_path = (f'/home/benchuser/code/data/map_{self.config.aoi.version}.geojson') 
            self.aoi_gdf = gpd.read_file(aoi_path)
            if self.config.aoi.exclude_indices:
                self.aoi_gdf = self.aoi_gdf.drop(self.config.aoi.exclude_indices)
            if self.config.aoi.include_indices:
                self.aoi_gdf = self.aoi_gdf.loc[self.config.aoi.include_indices]
            self.aoi_gdf['status'] = 'not processed'
            self.aoi_gdf.to_file(self.working_directory / 'aoi_metadata.geojson', driver = 'GeoJSON')
            self.aoi_processing_gdf = self.aoi_gdf
            self.chip_metadata_df = pd.DataFrame(columns=[
                        'chip_index',
                        'aoi_index',
                        'sentinel_2_dates',
                        'sentinel_1_dates',
                        'landsat_dates',
                        'land_cover',
                        'chip_footprint',
                        'epsg',
                        'status',
                ])
            self.chip_index = 0
            self.aoi_path = self.working_directory / 'aoi_metadata.geojson'
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'
    
    def download(self):
        """Route based on dataset type."""
        if getattr(self.config, "dataset", None) and getattr(self.config.dataset, "fire", False):
            return self.download_fire()
        else:
            return self.download_lc_chips()
    
    def download_lc_chips(self):
        """Download data for all AOIs that have not yet been processed from the AOI GeoJSON file"""
        for aoi_index, aoi in self.aoi_processing_gdf.iterrows():
            aoi_processor = AOI_Processor(
                aoi_index,
                aoi,
                self.chip_index,
                self.working_directory,
                self.catalog,
                self.config,
            )
            try:
                aoi_chip_df = aoi_processor.process_aoi()
                self.chip_metadata_df = pd.concat([self.chip_metadata_df, aoi_chip_df], ignore_index=True)
                self.chip_index += len(aoi_chip_df)
                aoi_status = 'success'
            except Exception as e:
                print(e)
                aoi_status = str(e)
            finally:
                self.aoi_gdf.loc[aoi_index, 'status'] = aoi_status
                self.aoi_gdf.to_file(self.aoi_path, driver='GeoJSON')
                self.chip_metadata_df.to_csv(self.chip_metadata_path, index=False)

    def download_fire(self):
        for aoi_index, aoi in self.aoi_processing_gdf.iterrows():
            print(f"\nProcessing AOI {aoi_index:02d}")
            event_date_ranges, control_date_ranges = get_fire_date_ranges(
                                                        aoi, 
                                                        n_control_years=getattr(self.config.fire, "n_control_years", 7)
                                                        )
            metadata_df = self.chip_metadata_df.copy()

            aoi_status = "not processed"

            aoi_processor = AOI_Processor(
                aoi_index=aoi_index,
                aoi=aoi,
                chip_index=self.chip_index,
                working_directory=self.working_directory,
                catalog=self.catalog,
                config=self.config,
                event_date_ranges = event_date_ranges, 
                control_date_ranges = control_date_ranges,
                metadata_df = metadata_df
            )
            chip_metadata_df = aoi_processor.generate_time_series()
            self.chip_metadata_df.to_csv(self.chip_metadata_path, index=False)
            aoi_status = "success"
            self._persist_progress(aoi_index, aoi_status)


    def _persist_progress(self, aoi_index, aoi_status):
        self.aoi_gdf.loc[aoi_index, "status"] = aoi_status
        self.aoi_gdf.to_file(self.aoi_path, driver="GeoJSON")
        self.chip_metadata_df.to_csv(self.chip_metadata_path, index=False)
