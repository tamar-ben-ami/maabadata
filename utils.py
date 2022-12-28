import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import sqlite3
from shapely.wkt import loads

LABEL_FIELD = "STAT_CAUSE_CODE"


def create_dataframe():
    conn = sqlite3.connect('FPA_FOD_20170508.sqlite')
    query = """
    select a.*, "POLYGON ((" || b.xmin || " " || b.ymin || ", " || b.xmax || " " || b.ymin || ", " || b.xmax || " " || b.ymax || ", " || b.xmin || " " || b.ymax || ", " || b.xmin || " " || b.ymin || "))" BOX_GEOMETRY
    from Fires a
    join idx_Fires_Shape b
    on a.OBJECTID = b.pkid
    """
    df = pd.read_sql_query(query, conn)
    return df


def date_features(df):
    df["cont_date_dt"] = pd.to_datetime(df.CONT_DATE, origin="julian",
                                        unit='D')
    df["disc_date_dt"] = pd.to_datetime(df.DISCOVERY_DATE, origin="julian",
                                        unit='D')
    df["cont_year"] = df["cont_date_dt"].apply(
        lambda x: 0 if np.isnan(x.year) else int(x.year))
    df["cont_mon"] = df["cont_date_dt"].apply(
        lambda x: 0 if np.isnan(x.month) else int(x.month))
    df["cont_dow"] = df["cont_date_dt"].apply(
        lambda x: 0 if np.isnan(x.weekday()) else int(x.weekday()))
    df["cont_is_weekend"] = df["cont_dow"].isin([6, 5, 4]).astype(int)
    df["disc_year"] = df["disc_date_dt"].apply(
        lambda x: 0 if np.isnan(x.year) else int(x.year))
    df["disc_mon"] = df["disc_date_dt"].apply(
        lambda x: 0 if np.isnan(x.month) else int(x.month))
    df["disc_dow"] = df["disc_date_dt"].apply(
        lambda x: 0 if np.isnan(x.weekday()) else int(x.weekday()))
    df["disc_is_weekend"] = df["disc_dow"].isin([6, 5, 4]).astype(int)

    df["time_to_cont"] = df["cont_date_dt"] - df["disc_date_dt"]
    df["time_to_cont"] = df["time_to_cont"].apply(
        lambda x: 0 if pd.isnull(x) else x.days)

    features = ["cont_year", "cont_mon", "cont_dow", "cont_is_weekend",
                "disc_year", "disc_mon", "disc_dow", "disc_is_weekend",
                "time_to_cont"]

    return features


def df_to_gdf(df):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs=4269)
    gdf = gdf.to_crs(4326)
    gdf['LATITUDE'] = gdf.geometry.y
    gdf['LONGITUDE'] = gdf.geometry.x
    gdf['POINT_GEOMETRY'] = gdf.geometry
    gdf['BOX_GEOMETRY'] = gpd.GeoSeries(gdf['BOX_GEOMETRY'].apply(loads), crs=4269).to_crs(4326)
    return gdf


def feature_extraction_noa(df):
    # add dummies
    dummy_cols = ["NWCG_REPORTING_AGENCY", "SOURCE_SYSTEM_TYPE"]
    if "Shape" in df.columns:
        df.drop(columns=["Shape"], inplace=True)
    df = pd.get_dummies(df,columns=dummy_cols)
    # encode of FIRE_SIZE_CLASS feature
    df["FIRE_SIZE_CLASS"] = df["FIRE_SIZE_CLASS"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F" : 6, "G":7})
    return df


def get_elevation(gdf):
    url = "https://api.opentopodata.org/v1/ned10m"
    elavation_dfs = []
    all_coords_str = gdf.geometry.y.astype(str) + "," + gdf.geometry.x.astype(str)
    for i in range(0, len(all_coords_str), 100):
        coords = '|'.join(all_coords_str[i:i + 100])
        data = {
            "locations": coords,
            "interpolation": "cubic"
        }
        response = requests.post(url, json=data)
        elavation_dfs.append(pd.json_normalize(response.json(), "results"))
    return pd.concat(elavation_dfs)['elevation']
