# Generating Data for GFM-Bech

This repo contains the code to generate samples for GELOS Land Cover. 

To run the code:
```
docker compose up --build -d
```

The notebook container uses Pixi with the project baked into the image at `/opt/pixi` and runs with `/app/` as the working directory. Live repo edits are mounted at `/app/`, while Pixi installs stay inside the image.

The repo contains areas of interest for sample generation in GeoJSON files located under `data/`. Each version of the dataset has its own file. 

## Dataset versions

- v0.1: Initial version with low number of Built-up class. This version contains 224 x 224 chips that have a homogeneous LC class of size 100 x 100 in the middle. During chip generation, we also don't filter out chips that are all from the same tile and have the same LC class (this is causing oversampling in some regions, and will be updated in future versions). 
- v0.11: Updated v0.1 with extra AOIs for Built-up class but downsampled to ~26K samples (5K for each class other than built up).
- v0.20: Same as v0.11 but no downsampling.
- v0.30: New AOIs added. The homogenous LC sample is 64 x 64 pixels. There is a cap on the number of chips of the same LC class in each S-2 tile to make sure there is a global diversity. Resulting in 101,801 chips.
  
- v0.40: This version provides dem, Landsat 8-9, Sentinel 2, and Sentinel 1 data for each chip.

- v0.50.1: This version adds new AOIs for better geospatial representation. File names follow this convention for dem: ```dem_{chip_id:06}.tif``` and this convention for mutlitemporal platforms (s1rtc, s2l2a, lc2l2): ```{platform}_{chip_id:06}_{date}.tif```. PNG visualization are included for multitemporal data sources wth the same naming convention, substituting ```png``` for ```tif```. The dataset consists of 78,585 chips. A chip tracker metadata GeoJSON named ```gelos_chip_tracker.geojson``` is included with the following fields:

original_id <class 'numpy.int32'> The chip's original ID, skipping chips which did not generate.
aoi_index <class 'numpy.int32'> The chip's source AOI, useful for grouping chips together.
s2l2a_dates <class 'str'> Dates of S2L2A observations. All dates formatted as "YYYYMMDD,YYYYMMDD,YYYYMMDD,YYYYMMDD".
s1rtc_dates <class 'str'> Dates of S1RTC observations.
lc2l2_dates <class 'str'> Dates of LC2L2 observations.
lulc <class 'str'> Numerical representation of LULC for the chip.
epsg <class 'numpy.int32'> EPSG of source S2L2A data which all other sources are reprojected to match.
status <class 'str'> Chip generation status - should always be "success" for valid chips.
s2l2a_scene_ids <class 'str'> STAC Scene IDS of S2L2A data. All scene IDS formatted as comma-delineated string.
s1rtc_scene_ids <class 'str'> STAC Scene IDS of S1RTC data.
lc2l2_scene_ids <class 'str'> STAC Scene IDS of LC2L2 data.
lulc_scene_ids <class 'str'> STAC Scene IDS of LULC data.
dem_scene_ids <class 'str'> STAC Scene IDS of DEM data.
id <class 'numpy.int32'> Unique ID for each valid chip, starting from 0 and incrementing by 1.
lat <class 'numpy.float64'> Latitude of chip center.
lon <class 'numpy.float64'> Longitude of chip center.
category <class 'str'> Category of chip land cover, e.g. "Trees" or "Rangeland".
color <class 'str'> Visualization color for the chip, based on category.
lc2l2_thumbs <class 'str'> Thumbnails of lc2l2 png visualizations for the chip, formatted as comma-delineated string.
s1rtc_thumbs <class 'str'> Thumbnails of s1rtc png visualizations for the chip
s2l2a_thumbs <class 'str'> Thumbnails of s2l2a png visualizations for the chip
lc2l2_paths <class 'str'> Paths to lc2l2 output tifs for the chip, formatted as comma-delineated string.
s1rtc_paths <class 'str'> Paths to s1rtc output tifs for the chip
s2l2a_paths <class 'str'> Paths to s2l2a output tifs for the chip
dem_paths <class 'str'> Paths to dem output tifs for the chip
geometry <class 'shapely.geometry.polygon.Polygon'> Chip footprint geometry
