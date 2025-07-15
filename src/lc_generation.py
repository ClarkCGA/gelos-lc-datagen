import pystac_client
import pystac
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
import planetary_computer
import warnings
import dask.distributed
import numpy as np
import rioxarray
import pandas as pd
import geopandas as gpd
from .utils import search_s2_scenes, search_s1_scenes, search_landsat_scenes, search_dem_scene, search_lc_scene 
from .utils import stack_data, stack_dem_data, stack_lc_data, unique_class, missing_values, gen_chips
import yaml
from dask.distributed import Client, LocalCluster
import logging
from pathlib import Path
import shutil
from functools import reduce
from shapely import box


def pystac_itemcollection_to_gdf(item_collection):
    geometries = []
    properties = []
    for item in item_collection:
        # Create box geometry from bbox
        bbox = item.bbox
        geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        geometries.append(geom)
        
        # Collect properties
        props = {
            'collection': item.collection_id,
        }
        properties.append(props)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
    
    return gdf


def generate_dataset(config_path):
    
    warnings.filterwarnings("ignore")
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("dask").setLevel(logging.ERROR)
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    version = config['dataset']['version']
    working_dir = Path(config['working_dir'])
    output_dir = Path(config['output_dir'])
    (working_dir / version).mkdir(exist_ok=True)
    metadata_filename = config['metadata']['file']
    aoi_version = config['aoi']['version']
    
    (working_dir / version).mkdir(exist_ok=True)
    shutil.copy(config_path, working_dir / version / "config.yaml")
    
    
    cluster = LocalCluster(silence_logs=logging.ERROR)
    client = Client(cluster)
    print(client.dashboard_link)
    
    retry = Retry(
        total=10, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
    )
    stac_api_io = StacApiIO(max_retries=retry)
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io
    )

    
    try:
        aoi_path = (working_dir / version / f'{aoi_version}.geojson')
        aoi_gdf = gpd.read_file(aoi_path)
        metadata_df = pd.read_csv(working_dir / version / metadata_filename)
        global_index = metadata_df['chip_id'].max() + 1
    except: 
        aoi_path = (f'/home/benchuser/code/data/map_{aoi_version}.geojson')
        aoi_gdf = gpd.read_file(aoi_path)
        aoi_gdf['processed'] = False
        aoi_gdf = aoi_gdf.drop(config['excluded_aoi_indices'])
        aoi_gdf.to_file(working_dir / version / f'{aoi_version}.geojson', driver = 'GeoJSON')
        metadata_df = pd.DataFrame(columns=["chip_id", "aoi_index", "s2_dates", "s1_dates", "landsat_dates", "lc", "x_center", "y_center", "epsg"])
        global_index = 0

    aoi_path = working_dir / version / f'{aoi_version}.geojson'
    
    for index, aoi in aoi_gdf.iterrows():
        if aoi_gdf.iloc[index]['processed']:
            print(f'AOI at index {index} already processed, continuing to next...')
            continue
        try:
            global_index, metadata_df = process_aoi(
                index,
                aoi,
                config,
                catalog,
                global_index,
                metadata_df,
                working_dir,
                version,
                metadata_filename
            )
        finally:
            aoi_gdf.loc[index, 'processed'] = True
            aoi_gdf.to_file(aoi_path, driver='GeoJSON')

def process_aoi(
    index,
    aoi,
    config,
    catalog,
    global_index,
    metadata_df,
    working_dir,
    version,
    metadata_filename
):
    
    print(f"\nProcessing AOI at index {index}")
    
    aoi_bounds = aoi['geometry'].bounds
    s2_items = pystac.item_collection.ItemCollection([])
    
    for date_range in config["sentinel_2"]["time_ranges"]:   
        print(f"Searching Sentinel-2 scenes for {date_range}")
        s2_items_season = search_s2_scenes(aoi, date_range, catalog, config)
        s2_items += s2_items_season

    if len(s2_items)<4:
        print(f"Missing Sentinel-2 scenes for AOI {aoi_bounds}")
        return global_index, metadata_df

    try:
        epsg = s2_items[0].properties["proj:epsg"]
    except:
        epsg = int(s2_items[0].properties["proj:code"].split(":")[-1])
    bbox_latlon = s2_items[0].bbox

    s1_items = pystac.item_collection.ItemCollection([])
    landsat_items = pystac.item_collection.ItemCollection([])

    for s2_item in s2_items:
        s2_datetime = s2_item.datetime
        print(f"Searching Sentinel-1 and Landsat scenes close to {s2_datetime}")
        s1_item = search_s1_scenes(aoi, s2_datetime, catalog, config)
        s1_items += s1_item
        landsat_item = search_landsat_scenes(aoi, s2_datetime, catalog, config)
        landsat_items += landsat_item
    
    if len(landsat_items) < 4:
        print(f"Missing Landsat Scenes for AOI {aoi_bounds}")
        return global_index, metadata_df

    if len(s1_items) < 4:
        print(f"Missing S1 scenes for AOI {aoi_bounds}")
        return global_index, metadata_df
            
    print("searching Land Cover data...")
    lc_items = search_lc_scene(aoi, catalog, config)
    if not lc_items:
        print(f"No Land Cover data found for AOI {aoi_bounds}")
        return global_index, metadata_df
    
    print("searching DEM data...")
    dem_items = search_dem_scene(aoi, catalog, config)
    if not dem_items:
        print(f"No DEM data found for AOI {aoi_bounds}")
        return global_index, metadata_df

        # first, get area of overlap of all item bboxes
    itemcollections = [s2_items, s1_items, landsat_items, lc_items, dem_items]
    bbox_gdf = pd.concat([pystac_itemcollection_to_gdf(items) for items in itemcollections])
    combined_geoms = bbox_gdf.groupby('collection')['geometry'].apply(lambda x: x.unary_union)
    overlap = reduce(lambda x, y: x.intersection(y), combined_geoms)
    overlap_bounds = overlap.bounds
    
    print("stacking Landsat data...")
    landsat_stack = stack_data(landsat_items, "landsat", config, epsg, overlap_bounds, bbox_is_latlon=True)
    if landsat_stack is None:
        print(f"Failed to stack Landsat bands for AOI {aoi_bounds}")
        return global_index, metadata_df
    
    overlap_bbox = landsat_stack.rio.bounds()

    print("stacking Sentinel-2 data...")
    s2_stack = stack_data(s2_items, "sentinel_2", config, epsg, overlap_bbox, bbox_is_latlon=False)
    if s2_stack is None:
        print(f"Failed to stack Sentinel-2 bands for AOI {aoi_bounds}")
        return global_index, metadata_df

    print("stacking DEM data...")
    dem_stack = stack_dem_data(dem_items, config,  epsg, overlap_bbox)
    if dem_stack is None:
        print(f"Failed to stack DEM data for AOI {aoi_bounds} and date range {date_range}")
        return global_index, metadata_df
    print("stacking Land Cover data...")
    lc_stack = stack_lc_data(lc_items, config, epsg, overlap_bbox)
    if lc_stack is None:
        print(f"Failed to stack Land Cover data for AOI {aoi_bounds} and date range {date_range}")
        return global_index, metadata_df

    print("stacking Sentinel-1 data...")
    s1_stack = stack_data(s1_items, "sentinel_1", config, epsg, overlap_bbox, bbox_is_latlon=False)
    if s1_stack is None:
        print(f"Failed to stack Sentinel-1 bands for AOI {aoi_bounds}")
        return global_index, metadata_df

    print("processing chips...")
    global_index, metadata_df = process_chips(s2_stack,
                                              s1_stack,
                                              landsat_stack,
                                              lc_stack,
                                              dem_stack,
                                              epsg,
                                              config,
                                              global_index,
                                              index,
                                              metadata_df,
                                              str(working_dir / version)
                                             )
    
    metadata_df.to_csv(working_dir / version / metadata_filename, index=False)
    return global_index, metadata_df
        
def process_array( 
            stack, 
            epsg: int,
            coords: tuple[float, float],
            array_name: str,
            config: dict,
            fill_na: bool = True,
            na_value: int = -999,
            dtype = np.int16,
            ):

    x, y = coords
    sample_size = int(config['chips']['sample_size'] / config[array_name]['resolution'])
    chip_size = int(config['chips']['chip_size'] / config[array_name]['resolution'])
    min_x_index = (x) * sample_size - int((chip_size - sample_size)/2)
    max_x_index = (x + 1) * sample_size + int((chip_size - sample_size)/2)
    min_y_index = (y) * sample_size - int((chip_size - sample_size)/2)
    max_y_index = (y + 1) * sample_size + int((chip_size - sample_size)/2)
    x_indices = slice(min_x_index, max_x_index)
    y_indices = slice(min_y_index, max_y_index)    

    array = stack.isel(x = x_indices, y = y_indices)
    array.rio.write_crs(f"epsg:{epsg}", inplace=True)
    array = array.where((array.x >= stack.x[(x) * sample_size]) &
                              (array.x < stack.x[(x + 1) * sample_size]) & 
                              (array.y <= stack.y[(y) * sample_size]) &
                              (array.y > stack.y[(y + 1) * sample_size])
                             )

    if fill_na:
        array = array.fillna(na_value)
        array = array.rio.write_nodata(na_value)
    array = array.astype(np.dtype(dtype))
    array = array.rename(array_name)
    if missing_values(array, chip_size, sample_size):
        return None
    return array

def process_chips(s2_stack, s1_stack, landsat_stack, lc_stack, dem_stack, epsg, config, global_index, aoi_index, metadata_df, root_path):

    print("Loading lc_stack")

    try:
        lc_stack = lc_stack.compute()
    except:
        print("skipping the AOI for no LC data")
        return global_index, metadata_df

    print("Loading s2_stack")
    
    try:
        s2_stack = s2_stack.compute()
    except:
        print("skipping the AOI for no S2 data")
        return global_index, metadata_df

    print("Loading s1_stack")
    
    try:
        s1_stack = s1_stack.compute()
    except:
        print("skipping the AOI for no S1 data")
        return global_index, metadata_df

    print("Loading dem_stack")
    
    try:
        dem_stack = dem_stack.compute()
    except:
        print("skipping the AOI for no dem data")
        return global_index, metadata_df

    try:
        landsat_stack = landsat_stack.compute()
    except:
        print("skipping the AOI for no landsat data")
        return global_index, metadata_df

    lc_sample_size = int(config['chips']['sample_size'] / config['land_cover']['resolution'])
    
    lc_uniqueness = lc_stack.coarsen(x = lc_sample_size,
                                     y = lc_sample_size,
                                     boundary = "trim"
                                    ).reduce(unique_class)
    lc_uniqueness[0:2, :] = False
    lc_uniqueness[-2:, :] = False
    lc_uniqueness[:, 0:2] = False
    lc_uniqueness[:, -2:] = False

    ys, xs = np.where(lc_uniqueness)

    # Following indices are added to limit the number of rangeland, bareground, and water chips per tile
    rangeland_index = 0
    bareground_index = 0
    water_index = 0
    tree_index = 0
    crops_index = 0
    for index in range(0, len(ys)):
        x = xs[index]
        y = ys[index]
        
        
        s2_array = process_array(
            stack = s2_stack, 
            epsg = epsg, 
            coords = (x, y),
            array_name = 'sentinel_2',
            config = config,
            fill_na = False, # so we can check for missing values
            na_value = None,
            dtype = np.int16,
        )

        if len(s2_array.time.values) < 4:
            print("Missing scenes in S2 array")
            continue
        
        if s2_array is None:
            print("Missing values in S2 array")
            continue    

        s1_array = process_array(
            stack = s1_stack, 
            epsg = epsg, 
            coords = (x, y),
            array_name = 'sentinel_1',
            config = config,
            fill_na = False,
            na_value = None,
            dtype = np.float32,
        )

        if s1_array is None:
            print("Missing values in S1 array")
            continue 

        if len(s2_array.time.values) < 4:
            print("Missing scenes in S1 array")
            continue

        landsat_array = process_array(
            stack = landsat_stack, 
            epsg = epsg, 
            coords = (x, y),
            array_name = 'landsat',
            config = config,
            fill_na = False,
            na_value = None,
            dtype = np.float32,
        )

        if landsat_array is None:
            print("Missing values in landsat array")
            continue 

        if len(s2_array.time.values) < 4:
            print("Missing scenes in landsat array")
            continue

        lc_array = process_array(
            stack = lc_stack, 
            epsg = epsg, 
            coords = (x, y),
            array_name = 'land_cover',
            config = config,
            fill_na = False,
            na_value = None,
            dtype = np.int8,
        )
        
        if lc_array is None:
            print("Missing values in land cover array")
            continue
            
        dem_array = process_array(
            stack = dem_stack, 
            epsg = epsg, 
            coords = (x, y),
            array_name = 'dem',
            config = config,
            fill_na = False,
            na_value = None,
            dtype = np.float32,
        )

        if dem_array is None:
            print("Missing values in dem array")
            continue
            
        if (np.isin(lc_array, [255, 130, 133])).any():
            raise ValueError('Wrong LC value')
        # Skipping Flooded Vegetation
        if (np.isin(lc_array, [4])).any():
            print("Skipping flooded vegetation")
            continue
        
        lc = np.unique(lc_array)
        if lc == 1:
            water_index += 1
            if water_index > 400:
                continue 
        elif lc == 8:
            bareground_index += 1
            if bareground_index > 400:
                continue
        elif lc == 11:
            rangeland_index += 1
            if rangeland_index > 400:
                continue
        elif lc == 2:
            tree_index += 1
            if tree_index > 400:
                continue
        elif lc == 5:
            crops_index += 1
            if crops_index > 400:
                continue
        print("Generating Chips...")
        gen_status, s2_dts, s1_dts, landsat_dts = gen_chips(s2_array, s1_array, landsat_array, lc_array, dem_array, global_index, root_path)
        if gen_status:
            metadata_df = pd.concat([pd.DataFrame([[global_index,
                                                    aoi_index,
                                                    s2_dts,
                                                    s1_dts,
                                                    landsat_dts,
                                                    np.unique(lc_array),
                                                    lc_stack.x[(x) * lc_sample_size + int(lc_sample_size / 2)].data,
                                                    lc_stack.y[(y) * lc_sample_size + int(lc_sample_size / 2)].data,
                                                    epsg]
                                                  ],
                                                  columns=metadata_df.columns
                                                 ),
                                     metadata_df],
                                    ignore_index=True
                                   )
            global_index += 1
    
    return global_index, metadata_df

def main():
    generate_dataset('/home/benchuser/code/config.yml')

if __name__ == '__main__':
    main()
    