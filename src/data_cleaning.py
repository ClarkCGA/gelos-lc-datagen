import numpy as np
import pandas as pd
import ast 
import geopandas as gpd
from shapely.geometry import Point
from .utils import get_continent
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm

platform_name_dict = {
    's2' : 'sentinel_2',
    's1' : 'sentinel_1',
    'landsat' : 'landsat'
}

def add_point(row):
    point = Point(row["x_center"], row["y_center"])
    gdf = gpd.GeoDataFrame([{'geometry': point}], crs=f"EPSG:{row["epsg"]}")
    gdf_reprojected = gdf.to_crs(epsg=4326)
    
    return gdf_reprojected.geometry.iloc[0]
    
def drop_rows(metadata_df, lc_class, count_to_drop):
    import random
    index_to_drop = random.sample(sorted(metadata_df[metadata_df.lc==lc_class].index.values), count_to_drop)
    metadata_df = metadata_df.drop(index_to_drop)

    return metadata_df

def clean_data(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
            
    version = config['dataset']['version']
    working_dir = Path(config['working_dir'])
    output_dir = Path(config['output_dir'])
    metadata = config['metadata']['file']
    
    metadata_df = pd.read_csv(working_dir / version / metadata)

    # ensure only desired lc classes are present
    metadata_df = metadata_df[metadata_df['lc'].isin([1, 2, 5, 7, 8, 11])]
    
    # get sampling factor, max count, and min count
    sampling_factor = config['land_cover']['sampling_factor']
    if sampling_factor:
        max_count = metadata_df.groupby("lc").count().max().iloc[0]
        min_count = metadata_df.groupby("lc").count().min().iloc[0]
        
        # use sampling factor to calculate correction factor, for proportional class drop quantities
        max_distance = max_count - min_count
        max_end_value = min_count * sampling_factor
        max_distance_to_max_end_value = max_count - max_end_value
        correction_factor = max_distance_to_max_end_value / max_distance
        
        # use correction factor to determine proportion of samples above min to drop for each class
        # the number of samples dropped will be proportional to the number of samples above minimum
        # this scales the number of samples between min and min * sampling factor
        if max_distance_to_max_end_value <= 0:
                
            for index, row in metadata_df.groupby("lc").count().iterrows():
                lc_class = index
                class_count = row['chip_id']
                class_distance = class_count - min_count
                drop_quantity = int(correction_factor * class_distance)
                metadata_df = drop_rows(metadata_df, lc_class, drop_quantity)
        
    metadata_df["index"] = np.arange(0, len(metadata_df))
    metadata_df = metadata_df.rename(columns={"chip_id" : "original_chip_id", "index" : "chip_id"})
    
    metadata_df["geometry"] = metadata_df[["x_center", "y_center", "epsg"]].apply(add_point, axis=1)
    metadata_gdf = gpd.GeoDataFrame(metadata_df, geometry="geometry", crs="EPSG:4326")
    (output_dir / version).mkdir(exist_ok=True)
    metadata_gdf.to_csv(output_dir / f'{version}/cleaned_df.csv', index=False)
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_gdf), desc="copying files to output dir..."):
        for col in ["s2_dates", "s1_dates", "landsat_dates"]:
            for i, date in enumerate(ast.literal_eval(row[col])):
                platform = col.split("_")[0]
                src_platform = platform_name_dict[platform]
                src_file = working_dir / version / f"{src_platform}_{row["original_chip_id"]:06}_{i}_{date}.tif"
                dst_file = output_dir / version / f"{platform}_{row["chip_id"]:06}_{date}.tif"
                shutil.copy2(src_file, dst_file)
                if platform in ['s2', 'landsat']:
                    src_file = working_dir / version / f"{src_platform}_{row["original_chip_id"]:06}_{i}_{date}.png"
                    dst_file = output_dir / version / f"{platform}_{row["chip_id"]:06}_{date}.png"
                    
        src_file = working_dir / version / f"dem_{row["original_chip_id"]:06}.tif"
        dst_file = output_dir / version / f"dem_{row["chip_id"]:06}.tif"
        shutil.copy2(src_file, dst_file)
    
    folder_to_zip = working_dir / version
    output_zip_file = output_dir / version / version
    shutil.make_archive(output_zip_file, 'zip', folder_to_zip)

def main():
    clean_data('config.yml')

if __name__ == '__main__':
    main()



