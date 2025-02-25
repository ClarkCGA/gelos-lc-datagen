import os
import glob
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgpd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

def select_unique_intersections(gdf):
    """Finds intersecting indices to remove"""
    gdf = gdf[gdf.index != gdf.index_right]
    return gdf.index

def filter_globfire(data_paths, years):
    filtered_gdf = dgpd.read_file(data_paths[years[0]], chunksize=2048)
    for i in range(1, len(years)):
        current_year = years[i]
        print(f"Processing: {years[i-1]} vs {current_year}")
        
        # Load the next year
        next_year_gdf = dgpd.read_file(data_paths[current_year], chunksize=2048)
        
        print(f"Checking for intersecting geometeries: {years[i-1]} vs {current_year}")
        intersections = dgpd.sjoin(filtered_gdf, next_year_gdf, how="inner", predicate="intersects")
        
        print(f"computing indices of intersecting geometeries: {years[i-1]} vs {current_year}")
        intersecting_ids = intersections.map_partitions(select_unique_intersections).compute()
        
        print(f"removing intersection of {current_year} from main datasets")
        filtered_gdf = filtered_gdf.map_partitions(lambda df: df[~df.index.isin(intersecting_ids)])
        
        print(f"Remaining records after filtering {current_year}: {filtered_gdf.compute().shape[0]}")
    
    final_filtered_gdf = filtered_gdf.compute()
    final_filtered_gdf.to_file("final_non_intersecting_wildfires_2015_2023.shp")


if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    yearly_fire = {
        "2015": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2015.shp", 
        "2016": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2016.shp",
        "2017": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2017.shp",
        "2018": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2018.shp",
        "2019": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2019.shp",
        "2020": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2020.shp", 
        "2021": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2021.shp", 
        "2022": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2022.shp", 
        "2023": "/workspace/_output/GLOBFIRE_burned_area_full_dataset_2002_2023/original_globfire_filtered_2023.shp"
    }

    years = sorted(yearly_fire.keys())

    filter_globfire(yearly_fire, years)

    print("Final filtered dataset saved!")