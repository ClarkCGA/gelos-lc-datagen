import numpy as np
import pandas as pd
import ast 
import geopandas as gpd
from shapely.geometry import Point
import shutil
from tqdm import tqdm
from pathlib import Path
from shapely import wkt

from src.gelos_config import GELOSConfig
 
def drop_rows(metadata_df, land_cover_class, count_to_drop):
    import random
    index_to_drop = random.sample(sorted(metadata_df[metadata_df.land_cover==land_cover_class].index.values), count_to_drop)
    metadata_df = metadata_df.drop(index_to_drop)

    return metadata_df

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
                    class_count = row['chip_id']
                    class_distance = class_count - min_count
                    drop_quantity = int(correction_factor * class_distance)
                    metadata_gdf = drop_rows(metadata_gdf, land_cover_class, drop_quantity)
            
        metadata_gdf["chip_id"] = np.arange(0, len(metadata_gdf))
        metadata_gdf['land_cover'] = metadata_gdf['land_cover'].map(lambda x: int(x))
        metadata_gdf['x_center'] = metadata_gdf.geometry.centroid.x
        metadata_gdf['y_center'] = metadata_gdf.geometry.centroid.y
        metadata_gdf = metadata_gdf.rename(columns={
            "chip_index": "original_chip_id",
            "sentinel_2_dates": "S2L2A_dates",
            "sentinel_1_dates": "S1RTC_dates",
            "landsat_dates": "LC2L2_dates"
        })
        metadata_gdf.index = metadata_gdf['chip_id']
        (self.output_dir / self.version).mkdir(exist_ok=True)
        metadata_gdf.to_file(self.output_dir / f'{self.version}/cleaned_df.geojson', driver='GeoJSON', index=False)

        for index, row in tqdm(metadata_gdf.iterrows(), total=len(metadata_gdf), desc="copying files to output dir..."):
            for col in ["S2L2A_dates", "S1RTC_dates", "LC2L2_dates"]:
                for i, date in enumerate(row[col].split(',')):
                    platform = col[:-6]
                    src_file = self.working_dir / self.version / f"{platform}_{row["original_chip_id"]:06}_{i}_{date}.tif"
                    dst_file = self.output_dir / self.version / f"{platform}_{row["chip_id"]:06}_{date}.tif"
                    shutil.copy2(src_file, dst_file)
                    src_file = self.working_dir / self.version / f"{platform}_{row["original_chip_id"]:06}_{i}_{date}.png"
                    dst_file = self.output_dir / self.version / f"{platform}_{row["chip_id"]:06}_{date}.png"
                    shutil.copy2(src_file, dst_file)
            src_file = self.working_dir / self.version / f"dem_{row["original_chip_id"]:06}.tif"
            dst_file = self.output_dir / self.version / f"dem_{row["chip_id"]:06}.tif"
            shutil.copy2(src_file, dst_file)
        
        folder_to_zip = self.working_dir / self.version
        output_zip_file = self.output_dir / self.version / self.version
        shutil.make_archive(output_zip_file, 'zip', folder_to_zip)

def main():
    config = GELOSConfig.from_yaml('/home/benchuser/code/config.yml')
    cleaner = DataCleaner(config)
    cleaner.clean()

if __name__ == '__main__':
    main()
