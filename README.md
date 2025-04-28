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

### Fire Track
- v0.10: 199 wildfire events from MTBS included (events that have a start date after 2023/01/01). Only "Wildfire" type is included. Chips include event and control events (control defined as the same season for the start date but from previous years). "Comment" was also filtered to make sure it is empty.
**Note**: MTBS data need to be added to the following folder: `data/mtbs`.