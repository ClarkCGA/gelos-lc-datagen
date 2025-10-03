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

os.environ["CPL_VSIL_CURL_NUM_CONNECTIONS"] = "20"

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
            # Pick AOI file based on mode
            self.aoi_path = (self.working_directory / ('fire_events_aoi.geojson' if self.config.dataset.fire else 'aoi_metadata.geojson'))
            self.aoi_gdf = gpd.read_file(self.aoi_path, geometry='geometry')

            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'
            self.chip_metadata_df = pd.read_csv(self.chip_metadata_path)

            # Handle empty CSV safely
            if self.chip_metadata_df.empty or 'aoi_index' not in self.chip_metadata_df.columns:
                last_aoi = -1
            else:
                last_aoi = pd.to_numeric(self.chip_metadata_df['aoi_index'], errors='coerce').max()
                if pd.isna(last_aoi):
                    last_aoi = -1

            # Only skip AOIs if there are actually processed rows
            self.aoi_gdf = self.aoi_gdf[self.aoi_gdf.index > last_aoi]
            self.aoi_processing_gdf = self.aoi_gdf

            # Fire chips don't use a numeric chip counter; set to 0. LC keeps numeric.
            if getattr(self.config, "dataset", None) and getattr(self.config.dataset, "fire", False):
                self.chip_index = 0
            else:
                if self.chip_metadata_df.empty or 'chip_index' not in self.chip_metadata_df.columns:
                    self.chip_index = 0
                else:
                    last_chip = pd.to_numeric(self.chip_metadata_df['chip_index'], errors='coerce').max()
                    self.chip_index = (int(last_chip) + 1) if pd.notna(last_chip) else 0

        # handle the case where the script is starting a new download operation for fire events
        elif self.config.dataset.fire and (self.working_directory / f'fire_events_aoi.geojson').exists():
            self.aoi_path = (self.working_directory / 'fire_events_aoi.geojson')
            self.aoi_gdf = gpd.read_file(self.aoi_path)
            self.aoi_processing_gdf = self.aoi_gdf
            self.chip_metadata_df = pd.DataFrame(columns=[ # fire chip metadata # TODO: read in existing chip_metadata if exists
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
                    "post_date",
                    'chip_footprint',
                    'status',
                    ])
            self.chip_index = 0
            self.aoi_path = self.working_directory / 'aoi_metadata.geojson'
            self.chip_metadata_path = self.working_directory / 'chip_metadata.csv'

    def download(self):
        for aoi_index, aoi in self.aoi_processing_gdf.iterrows():
            event_date_ranges, control_date_ranges = get_fire_date_ranges(
                                                        aoi, 
                                                        n_control_years=getattr(
                                                        self.config.dataset.fire, 
                                                        "n_control_years", 
                                                        7)
                                                    )
            
            aoi_processor = AOI_Processor(
                aoi_index=aoi_index,
                aoi=aoi,
                chip_index=self.chip_index,
                working_directory=self.working_directory,
                catalog=self.catalog,
                config=self.config,
            )

            event_success = False

            try:
                print(f"Processing Event Chips for AOI {aoi_index:02d}")
                aoi_chip_df = aoi_processor.process_aoi("event", event_date_ranges, self.chip_metadata_df)
                self.chip_metadata_df = pd.concat([self.chip_metadata_df, aoi_chip_df], ignore_index=True)
                aoi_status = "success"
                event_success = True
            except Exception as e:
                print(f"[event-error] AOI {aoi_index:02d}: {e}")
                aoi_status = str(e)
            
            if not event_success:
                print(f"[skip-controls] AOI {aoi_index:02d}: event failed or produced no chips.")
                continue
            for ctrl_dates in control_date_ranges:
                try:
                    print(f"Processing Control Chips for AOI {aoi_index:02d}")
                    aoi_chip_df = aoi_processor.process_aoi("control", ctrl_dates, self.chip_metadata_df)
                    self.chip_metadata_df = pd.concat([self.chip_metadata_df, aoi_chip_df], ignore_index=True)
                    aoi_status = "success"
                except Exception as e:
                    print(f"[control-error] AOI {aoi_index:02d}: {e}")
                    aoi_status = str(e)
            
            self.aoi_gdf.loc[aoi_index, 'status'] = aoi_status
            self.aoi_gdf.to_file(self.aoi_path, driver='GeoJSON')
            self.chip_metadata_df.drop_duplicates(
                        subset=["chip_index","platform","date"], keep="last", inplace=True
                    )
            self.chip_metadata_df.to_csv(self.chip_metadata_path, index=False)