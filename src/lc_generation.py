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
    
    log_errors = config['log_errors']
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
        aoi_gdf['error'] = None
        aoi_gdf = aoi_gdf.drop(config['excluded_aoi_indices'])
        aoi_gdf.to_file(working_dir / version / f'{aoi_version}.geojson', driver = 'GeoJSON')
        metadata_df = pd.DataFrame(columns=[
            "chip_id", 
            "aoi_index", 
            "s2_dates", 
            "s1_dates", 
            "landsat_dates", 
            "lc", 
            "x_center", 
            "y_center", 
            "epsg",
            "error_msg"]
            )
        global_index = 0

    aoi_path = working_dir / version / f'{aoi_version}.geojson'
    
    for index, aoi in aoi_gdf.iterrows():
        if aoi_gdf.iloc[index]['processed']:
            print(f'AOI at index {index} already processed, continuing to next...')
            continue
        try:
            global_index, metadata_df, failure_message = process_aoi(
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
        except Exception as e:
            failure_message = e
        finally:
            aoi_gdf.loc[index, 'processed'] = True
            aoi_gdf.loc[index, 'error'] = failure_message
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
        failure_message = 's2_scenes_missing'
        return global_index, metadata_df, failure_message

    try:
        epsg = s2_items[0].properties["proj:epsg"]
    except:
        epsg = int(s2_items[0].properties["proj:code"].split(":")[-1])
    bbox_latlon = s2_items[0].bbox

    s1_items = pystac.item_collection.ItemCollection([])
    landsat_items = pystac.item_collection.ItemCollection([])

    for s2_item in s2_items:
        s2_datetime = s2_item.datetime
        print(f"searching sentinel-1 and landsat scenes close to {s2_datetime}")
        s1_item = search_s1_scenes(aoi, s2_datetime, catalog, config)
        s1_items += s1_item
        landsat_item = search_landsat_scenes(aoi, s2_datetime, catalog, config)
        landsat_items += landsat_item

    if len(landsat_items) < 4:
        print(f"missing landsat scenes for aoi {aoi_bounds}")
        failure_message = 'landsat_scenes_missing'
        return global_index, metadata_df, failure_message

    if len(s1_items) < 4:
        print(f"missing s1 scenes for aoi {aoi_bounds}")
        failure_message = 's1_scenes_missing'
        return global_index, metadata_df, failure_message
            
    print("searching land cover data...")
    lc_items = search_lc_scene(aoi, catalog, config)
    if not lc_items:
        print(f"no land cover data found for aoi {aoi_bounds}")
        failure_message = 'lc_scenes_missing'
        return global_index, metadata_df, failure_message
    
    print("searching dem data...")
    dem_items = search_dem_scene(aoi, catalog, config)
    if not dem_items:
        print(f"no dem data found for aoi {aoi_bounds}")
        failure_message = 'dem_scenes_missing'
        return global_index, metadata_df, failure_message

        # first, get area of overlap of all item bboxes
    itemcollections = [s2_items, s1_items, landsat_items, lc_items, dem_items]
    bbox_gdf = pd.concat([pystac_itemcollection_to_gdf(items) for items in itemcollections])
    combined_geoms = bbox_gdf.groupby('collection')['geometry'].apply(lambda x: x.unary_union)
    overlap = reduce(lambda x, y: x.intersection(y), combined_geoms)
    overlap_bounds = overlap.bounds
    
    print("stacking landsat data...")
    landsat_stack = stack_data(landsat_items, "landsat", config, epsg, overlap_bounds, bbox_is_latlon=True)
    if landsat_stack is None:
        print(f"failed to stack landsat bands for aoi {aoi_bounds}")
        failure_message = 'landsat_stack_failure'
        return global_index, metadata_df, failure_message
    
    overlap_bbox = landsat_stack.rio.bounds()

    print("stacking sentinel-2 data...")
    s2_stack = stack_data(s2_items, "sentinel_2", config, epsg, overlap_bbox, bbox_is_latlon=False)
    if s2_stack is None:
        print(f"failed to stack sentinel-2 bands for aoi {aoi_bounds}")
        failure_message = 's2_stack_failure'
        return global_index, metadata_df, failure_message

    print("stacking dem data...")
    dem_stack = stack_dem_data(dem_items, config,  epsg, overlap_bbox)
    if dem_stack is None:
        print(f"failed to stack dem data for aoi {aoi_bounds} and date range {date_range}")
        failure_message = 'dem_stack_failure'
        return global_index, metadata_df, failure_message
    print("stacking land cover data...")
    lc_stack = stack_lc_data(lc_items, config, epsg, overlap_bbox)
    if lc_stack is None:
        print(f"failed to stack land cover data for aoi {aoi_bounds} and date range {date_range}")
        failure_message = 'lc_stack_failure'
        return global_index, metadata_df, failure_message

    print("stacking sentinel-1 data...")
    s1_stack = stack_data(s1_items, "sentinel_1", config, epsg, overlap_bbox, bbox_is_latlon=False)
    if s1_stack is None:
        print(f"failed to stack sentinel-1 bands for aoi {aoi_bounds}")
        failure_message = 's1_stack_failure'
        return global_index, metadata_df, failure_message

    print("processing chips...")
    global_index, metadata_df, failure_message = process_chips(s2_stack,
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
    failure_message = 'success'
    return global_index, metadata_df, failure_message
        
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

def fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message):
    
    metadata_df = pd.concat([pd.DataFrame([[global_index,
                                            aoi_index,
                                            None,
                                            None,
                                            None,
                                            lc,
                                            lc_stack.x[(x) * lc_sample_size + int(lc_sample_size / 2)].data,
                                            lc_stack.y[(y) * lc_sample_size + int(lc_sample_size / 2)].data,
                                            epsg,
                                            failure_message]
                                          ],
                                          columns=metadata_df.columns
                                         ),
                             metadata_df],
                            ignore_index=True
                           )
    global_index += 1
    return global_index, metadata_df
    
def process_chips(s2_stack, s1_stack, landsat_stack, lc_stack, dem_stack, epsg, config, global_index, aoi_index, metadata_df, root_path):

    print("Loading lc_stack")

    try:
        lc_stack = lc_stack.compute()
    except Exception as failure_message:
        print("skipping the AOI for no LC data")
        return global_index, metadata_df, failure_message

    print("Loading s2_stack")
    
    try:
        s2_stack = s2_stack.compute()
    except Exception as failure_message:
        print("skipping the AOI for no S2 data")
        return global_index, metadata_df, failure_message

    print("Loading s1_stack")
    
    try:
        s1_stack = s1_stack.compute()
    except Exception as failure_message:
        print("skipping the AOI for no S1 data")
        return global_index, metadata_df, failure_message

    print("Loading dem_stack")
    
    try:
        dem_stack = dem_stack.compute()
    except Exception as failure_message:
        print("skipping the AOI for no dem data")
        return global_index, metadata_df, failure_message

    try:
        landsat_stack = landsat_stack.compute()
    except Exception as failure_message:
        print("skipping the AOI for no landsat data")
        return global_index, metadata_df, failure_message

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
    lc_indices = {1: 0, 2: 0, 5: 0, 7: 0, 8: 0, 11: 0}
    for index in range(0, len(ys)):
        x = xs[index]
        y = ys[index]
        
 
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
            failure_message = "lc_values_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, None, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue

        if (~np.isin(lc_array, [1, 2, 4, 5, 7, 8, 11])).any():
            failure_message = "lc_values_wrong"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, None, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue

        # Skipping Flooded Vegetation
        if (np.isin(lc_array, [4])).any():
            failure_message = "lc_values_flooded_vegetation"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, None, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue

        lc = int(np.unique(lc_array)[0])
        
        if lc_indices[lc] > 400:
            failure_message = f"lc_{lc}_limit"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue

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
        if s2_array is None:
            print("Missing values in S2 array")
            failure_message = "s2_values_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue 

        if len(s2_array.time.values) < 4:
            print("Missing scenes in S2 array")
            failure_message = "s2_scenes_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
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
            failure_message = "s1_values_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue 

        if len(s1_array.time.values) < 4:
            print("Missing scenes in S1 array")
            failure_message = "s1_scenes_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
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
            failure_message = "landsat_values_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue 

        if len(s2_array.time.values) < 4:
            print("Missing scenes in landsat array")
            failure_message = "landsat_scenes_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
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
            failure_message = "dem_values_missing"
            global_index, metadata_df = fail_chip(global_index, aoi_index, metadata_df, lc, lc_stack, lc_sample_size, x, y, epsg, failure_message)
            continue 

        print(f"Generating Chips for chip {global_index}...")
        gen_status, s2_dts, s1_dts, landsat_dts = gen_chips(s2_array, s1_array, landsat_array, lc_array, dem_array, global_index, root_path)
        if gen_status:
            metadata_df = pd.concat([pd.DataFrame([[global_index,
                                                    aoi_index,
                                                    s2_dts,
                                                    s1_dts,
                                                    landsat_dts,
                                                    lc,
                                                    lc_stack.x[(x) * lc_sample_size + int(lc_sample_size / 2)].data,
                                                    lc_stack.y[(y) * lc_sample_size + int(lc_sample_size / 2)].data,
                                                    epsg,
                                                    'success']
                                                  ],
                                                  columns=metadata_df.columns
                                                 ),
                                     metadata_df],
                                    ignore_index=True
                                   )
            global_index += 1
    
    return global_index, metadata_df, None

def main():
    generate_dataset('/home/benchuser/code/config.yml')

if __name__ == '__main__':
    main()
    