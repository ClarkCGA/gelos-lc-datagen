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
        if (self.working_directory / 'chip_metadata.csv').exists():
            self.aoi_path = (self.working_directory / 'aoi_metadata.geojson')
            self.aoi_gdf = gpd.read_file(self.aoi_path)
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'
            self.chip_metadata_df = pd.read_csv(self.chip_metadata_path)
            # drop aoi which already have chips generated
            self.aoi_processing_gdf = self.aoi_gdf[self.aoi_gdf.index > self.chip_metadata_df['aoi_index'].max()]
            self.chip_index = self.chip_metadata_df['chip_index'].max() + 1

        # handle the case where the script is starting a new download operation
        else:
            aoi_path = (f'/app/code/data/map_{self.config.aoi.version}.geojson')
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
                        's2l2a_dates',
                        's1rtc_dates',
                        'lc2l2_dates',
                        'lulc',
                        'chip_footprint',
                        'epsg',
                        'status',
                ])
            self.chip_index = 0
            self.aoi_path = self.working_directory / 'aoi_metadata.geojson'
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'

    
    def download(self):
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