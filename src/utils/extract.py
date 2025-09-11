import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from shapely.geometry import mapping
from shapely.wkt import loads
from shapely.geometry import shape

import warnings
warnings.filterwarnings("ignore")

# utility functions for extracting wildfire events from S2 WCD and MTBS datasets
def add_start_date(event_id):
    start_date = pd.to_datetime(event_id[-8:], format="%Y%m%d")
    return start_date

def add_pre_date(pre_id):
    if len(pre_id)<=15:
        pre_date = pd.to_datetime(pre_id[-8:], format="%Y%m%d")
    else:
        pre_date = pd.to_datetime(pre_id.split("_")[0][-8:], format="%Y%m%d")

    return pre_date

def add_post_date(post_id):
    if len(post_id)<=15:
        post_date = pd.to_datetime(post_id[-8:], format="%Y%m%d")
    else:
        post_date = pd.to_datetime(post_id.split("_")[0][-8:], format="%Y%m%d")

    return post_date

def get_fire_path(fire_df):
    df_filtered = fire_df[fire_df["event_type"].isin(["pre", "post"])]
    summary = (
        df_filtered
        .groupby("location")
        .agg(
            geometry=("geometry", "first"),
            pre_date=("date", lambda x: sorted(x[df_filtered.loc[x.index, "event_type"] == "pre"])[0] 
                      if any(df_filtered.loc[x.index, "event_type"] == "pre") else None),
            post_date=("date", lambda x: sorted(x[df_filtered.loc[x.index, "event_type"] == "post"])[0] 
                       if any(df_filtered.loc[x.index, "event_type"] == "post") else None),
        )
        .reset_index()
    )

    if isinstance(summary["geometry"].iloc[0], str):
        summary["geometry"] = summary["geometry"].apply(loads)
    summary_gdf = gpd.GeoDataFrame(summary, geometry=summary["geometry"], crs="EPSG:4326")
    summary_gdf["pre_date"] = pd.to_datetime(summary_gdf["pre_date"])
    summary_gdf["post_date"] = pd.to_datetime(summary_gdf["post_date"])

    mask_paths = (
        fire_df[fire_df["event_type"] == "mask"]
        .groupby("location")["path"]
        .first()  #one fire mask per location
        .reset_index()
        .rename(columns={"path": "mask_path"})
    )
    summary_gdf = summary_gdf.merge(mask_paths, on="location", how="left")
    return summary_gdf


def filter_fires_by_date(fire_gdf, start_date, end_date):
    """ function to filter fire events by date range specific to the MTBS attribute schema"""
    start_date = pd.to_datetime(start_date, format="%Y%m%d")
    end_date = pd.to_datetime(end_date, format="%Y%m%d")
    fire_gdf["start_date"] = fire_gdf["Event_ID"].apply(add_start_date)
    fire_gdf = fire_gdf[fire_gdf["Incid_Type"].isin(["Wildfire"])] # filter out "Prescribed Fire"
    fire_gdf = fire_gdf[fire_gdf["Comment"].isnull()]
    fire_gdf = fire_gdf[~fire_gdf["Pre_ID"].isnull()]
    fire_gdf = fire_gdf[~fire_gdf["Post_ID"].isnull()]
    fire_gdf["location"] = fire_gdf["Incid_Name"]
    fire_gdf["source"] = fire_gdf["Map_Prog"]
    fire_gdf = fire_gdf[fire_gdf["start_date"]>start_date]
    fire_gdf = fire_gdf[fire_gdf["start_date"]<end_date]
    fire_gdf["pre_date"] = fire_gdf["Pre_ID"].apply(add_pre_date)
    fire_gdf["post_date"] = fire_gdf["Post_ID"].apply(add_post_date)
    return fire_gdf


