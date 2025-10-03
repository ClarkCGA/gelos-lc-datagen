# Generating Data for GFM-Bech

This repo contains the code to generate samples for GFM-Bench. 

To run the code:
```
docker run -it -p 8888:8888 -p 8787:8787 -v <PATH_TO_REPO>:/home/benchuser/code/:rw -v <PATH_TO_DATA>:/home/benchuser/data/ gfm-bench
```

The repo contains areas of interest for sample generation in GeoJSON files located under `data/`. Each version of the dataset has its own file. 

This dataset has multiple tracks. 
- LC Track: The main notebook to generate the data is `00_lc_generation.ipynb`, and the `01_lc_data_cleaning.ipynb` is used to clean the data.
- Fire Track: The main notebook to generate the data is `02_fire_generation.ipynb` which is currently a WIP. 

## Dataset versions

### LC Track
- v0.1: Initial version with low number of Built-up class. This version contains 224 x 224 chips that have a homogeneous LC class of size 100 x 100 in the middle. During chip generation, we also don't filter out chips that are all from the same tile and have the same LC class (this is causing oversampling in some regions, and will be updated in future versions). 
- v0.11: Updated v0.1 with extra AOIs for Built-up class but downsampled to ~26K samples (5K for each class other than built up).
- v0.20: Same as v0.11 but no downsampling.
- v0.30: New AOIs added. The homogenous LC sample is 64 x 64 pixels. There is a cap on the number of chips of the same LC class in each S-2 tile to make sure there is a global diversity. Resulting in 101,801 chips.
  
- v0.40: This version provides DEM, Landsat 8-9, Sentinel 2, and Sentinel 1 data for each chip. File names follow this convention for DEM: ```dem_{chip_id:06}.tif``` and this convension for mutlitemporal platforms (sentinel_1, sentinel_2, landsat): ```{platform}_{chip_id:06}_{date}.tif```. PNG visualization are included for multitemporal data sources wth the same naming convention, substituting ```png``` for ```tif```. The dataset consists of 77,547 chips. A chip tracker metadata GeoJSON named ```cleaned_df.geojson``` is included with the following fields:
  - chip_id (int): The chip's unique ID, starting from 0 and incrementing by 1
  - original_chip_id (int): The chip's ID from the chip generation pipeline, skipping failed chips
  - aoi_index (int): The Area of Interest index the chip derives from
  - sentinel_2_dates (str(list(int))): A list of 4 integer dates in format YYYYMMDD (no dashes) representing the days of each Sentinel 2 observation stored as a string. This can be read as a list using string literal eval.
  - sentinel_1_dates (str(list(int))): A list of 4 integer dates in format YYYYMMDD (no dashes) representing the days of each Sentinel 1 observation stored as a string.
  - landsat_dates (str(list(int))):: A list of 4 integer dates in format YYYYMMDD (no dashes) representing the days of each Landsat observation stored as a string.
  - land_cover (int): The land cover code of the chip, taken from the [10m annual LULC dataset](https://collections.sentinel-hub.com/impact-observatory-lulc-map/readme.html)
  - chip_footprint (shapely.Polygon): The footprint of the chip in EPSG 4326. This is the geometry field.
  - epsg (int): The EPSG of the raster data for the chip
  - status (str): The result of chip generation - all in the final dataset will be 'success'.
  - x_center (int): Chip centroid x coordinate in ESPG 4326
  - y_center (int): Chip centroid y coordinate in EPSG 4326

### Fire Track
- v0.10: 199 wildfire events from MTBS included (events that have a start date after 2023/01/01). Only "Wildfire" type is included. Chips include event and control events (control defined as the same season for the start date but from previous years). "Comment" was also filtered to make sure it is empty.
**Note**: MTBS data need to be added to the following folder: `data/mtbs`.