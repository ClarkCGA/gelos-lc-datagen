import numpy as np
import pandas as pd
import ast 
import geopandas as gpd
from shapely.geometry import Point
import shutil
from tqdm import tqdm
from pathlib import Path
from shapely import wkt
import s3fs
from src.gelos_config import GELOSConfig

s3 = s3fs.S3FileSystem(anon=True)

def drop_rows(metadata_df, land_cover_class, count_to_drop):
    import random
    index_to_drop = random.sample(sorted(metadata_df[metadata_df.land_cover==land_cover_class].index.values), count_to_drop)
    metadata_df = metadata_df.drop(index_to_drop)

    return metadata_df

def filter_by_n_dates(row, modality, required_dates=4):
    # helper function to check number of dates for a modality
    return required_dates == len(row[f'{modality}_dates'].split(','))

def gen_thumbnail_urls(row, image, s3_prefix="https://gelos-fm.s3.amazonaws.com/data"):
    """
    Generate S3 urls for thumbnails
    :param row: dictionary with id and dates
    :param s3_prefix: S3 url prefix 
    :param image: str, e.g., "landsat"
    :return urls: a list of urls
    """
    dates = row[f"{image}_dates"]
    dates_list = dates.split(',')
    id = row['id']
    url_list = [
        f"{s3_prefix}/{image}_{id:06}_{date}.png"
        for date in dates_list
    ]
    return ','.join(url_list)
# Color dictionaries
color_dict = {
    '1': '#419bdf',   # Water
    '2': '#397d49',   # Trees
    '5': '#e49635',   # Crops
    '7': '#c4281b',   # Built area
    '8': '#a59b8f',   # Bare ground
    '11': '#e3e2c3',  # Rangeland
}
land_cover = {
    '1': 'Water',
    '2': 'Trees',
    '5': 'Crops',
    '7': 'Built area',
    '8': 'Bare ground',
    '11': 'Rangeland'
}
class DataCleaner:
    def __init__(self, config: GELOSConfig):
        self.config = config
        self.version = self.config.dataset.version
        self.working_dir = Path(self.config.directory.working)
        self.output_dir = Path(self.config.directory.output)
        
    def clean(self):
        metadata_df = pd.read_csv(self.working_dir / self.version / "chip_metadata.csv")
        metadata_df['chip_footprint'] = gpd.GeoSeries(metadata_df['chip_footprint'].dropna().map(wkt.loads), crs=4326)
        metadata_gdf = gpd.GeoDataFrame(metadata_df, geometry = 'chip_footprint', crs=4326)
        metadata_gdf = metadata_gdf[metadata_gdf['status'] == 'success']

        # ensure only desired land_cover classes are present
        metadata_gdf = metadata_gdf[metadata_gdf['land_cover'].isin([1, 2, 5, 7, 8, 11])]
        
        # filter rows where there are insufficient samples
        for modality in ['sentinel_1', 'sentinel_2', 'landsat']:
            metadata_gdf = metadata_gdf[
                metadata_gdf.apply(lambda row: filter_by_n_dates(row, modality, required_dates=4), axis=1)
            ]
        
        # get sampling factor, max count, and min count
        sampling_factor = self.config.land_cover.sampling_factor
        if sampling_factor:
            max_count = metadata_gdf.groupby("land_cover").count().max().iloc[0]
            min_count = metadata_gdf.groupby("land_cover").count().min().iloc[0]
            
            # use sampling factor to calculate correction factor, for proportional class drop quantities
            max_distance = max_count - min_count
            max_end_value = min_count * sampling_factor
            max_distance_to_max_end_value = max_count - max_end_value
            correction_factor = max_distance_to_max_end_value / max_distance
            
            # use correction factor to determine proportion of samples above min to drop for each class
            # the number of samples dropped will be proportional to the number of samples above minimum
            # this scales the number of samples between min and min * sampling factor
            if max_distance_to_max_end_value > 0:
                    
                for index, row in metadata_gdf.groupby("land_cover").count().iterrows():
                    land_cover_class = index
                    class_count = row['chip_index']
                    class_distance = class_count - min_count
                    drop_quantity = int(correction_factor * class_distance)
                    metadata_gdf = drop_rows(metadata_gdf, land_cover_class, drop_quantity)
            
        # create metadata columns
        metadata_gdf['id'] = np.arange(0, len(metadata_gdf))
        metadata_gdf['lat'] = metadata_gdf.geometry.centroid.x
        metadata_gdf['lon'] = metadata_gdf.geometry.centroid.y
        metadata_gdf = metadata_gdf.rename(columns={"chip_index": "original_id"})
        metadata_gdf.index = metadata_gdf['id']
        metadata_gdf['land_cover'] = metadata_gdf['land_cover'].astype(int).astype(str)
        metadata_gdf['category'] = metadata_gdf['land_cover'].map(land_cover)
        metadata_gdf['color'] = metadata_gdf['land_cover'].map(color_dict)

        for image in ["landsat", "sentinel_1", "sentinel_2"]:
            metadata_gdf[f"{image}_thumbs"] = metadata_gdf.apply(
                gen_thumbnail_urls, axis=1, image=image
            )

        (self.output_dir / self.version).mkdir(exist_ok=True)
        
        # save to geojson
        metadata_gdf.to_file(self.output_dir / f'{self.version}/gelos_chip_tracker.geojson', driver='GeoJSON', index=False)

        # move files to destination folder
        for index, row in tqdm(metadata_gdf.iterrows(), total=len(metadata_gdf), desc="copying files to output dir..."):
            for col in ["sentinel_2_dates", "sentinel_1_dates", "landsat_dates"]:
                for i, date in enumerate(row[col].split(',')):
                    platform = col[:-6]
                    src_file = self.working_dir / self.version / f"{platform}_{row["original_id"]:06}_{i}_{date}.tif"
                    dst_file = self.output_dir / self.version / f"{platform}_{row["id"]:06}_{date}.tif"
                    shutil.copy2(src_file, dst_file)
                    src_file = self.working_dir / self.version / f"{platform}_{row["original_id"]:06}_{i}_{date}.png"
                    dst_file = self.output_dir / self.version / f"{platform}_{row["id"]:06}_{date}.png"
                    shutil.copy2(src_file, dst_file)
            src_file = self.working_dir / self.version / f"dem_{row["original_id"]:06}.tif"
            dst_file = self.output_dir / self.version / f"dem_{row["id"]:06}.tif"
            shutil.copy2(src_file, dst_file)
        
        # zip folder
        folder_to_zip = self.working_dir / self.version
        output_zip_file = self.output_dir / self.version / self.version
        shutil.make_archive(output_zip_file, 'zip', folder_to_zip)

def main():
    config = GELOSConfig.from_yaml('/home/benchuser/code/config.yml')
    cleaner = DataCleaner(config)
    cleaner.clean()

if __name__ == '__main__':
    main()
