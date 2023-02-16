import pandas as pd
import geopandas as gpd
import numpy as np
# import rasterio
import requests
import sqlite3
from shapely.wkt import loads
from config import *
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


LABEL_FIELD = "STAT_CAUSE_CODE"
OID_FIELD = "OBJECTID"


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


def aggregative_features(df):
    # frequency of a month
    months = df["disc_mon"].value_counts().reset_index().rename(columns={"index": "disc_mon", "disc_mon": "amount"})
    months["frequency"] = months["amount"] / df.shape[0]
    df = pd.merge(df, months, left_on=["disc_mon"], right_on=["disc_mon"])
    features = ["month_frequency"]
    return df

def geo_vector_features(df, external_vector_gdfs, is_poly=False):
    gdf = df_to_gdf(df)

    features = []
    for external_gdf, name_of_data, geo_oid_field in external_vector_gdfs:
        gdf[f"distances_{name_of_data}"] = \
        gpd.sjoin_nearest(gdf, external_gdf, how='left', lsuffix='left',
                          rsuffix='right',
                          distance_col=f"distances_{name_of_data}")[
            f"distances_{name_of_data}"]
        features.append(f"distances_{name_of_data}")
        df[f"distances_{name_of_data}"] = df[[OID_FIELD]].merge(gdf[[OID_FIELD, f"distances_{name_of_data}"]], left_on=OID_FIELD, right_on=OID_FIELD)[f"distances_{name_of_data}"]
        # Gdf with polygons!
        # if is_poly:
        #     gs = gpd.sjoin(gdf, external_gdf, how='left', predicate="intersects",
        #               lsuffix='left', rsuffix='right').groupby(
        #         OID_FIELD)[geo_oid_field].size().rename(f"count_intersections_{name_of_data}")
        #     gdf = gdf.merge(gs, how="left", left_on=OID_FIELD, right_index=True)
        #     features.append(f"count_intersections_{name_of_data}")

    return features


def state_county_features_train(train_df):
    features = ["state_county_gb", "state_gb"]
    train_df["STATE_COUNTY"] = train_df["STATE"] + "_" + train_df["COUNTY"]
    gb_state_county = train_df.groupby("STATE_COUNTY").size().rename(
        "state_county_gb")
    train_df["state_county_gb"] = train_df.merge(gb_state_county, how="left", right_index=True,
                  left_on="STATE_COUNTY")["state_county_gb"]
    gb_state = train_df.groupby("STATE").size().rename("state_gb")
    train_df["state_gb"] = train_df.merge(gb_state, how="left", right_index=True,
                  left_on="STATE")["state_gb"]
    return features


def state_county_features_test(test_df, train_df):
    test_df["STATE_COUNTY"] = test_df["STATE"] + "_" + test_df["COUNTY"]
    test_df["state_county_gb"] = test_df.merge(train_df[["state_county_gb"]] , how="left", right_index=True,
                  left_on="STATE_COUNTY")["state_county_gb"]
    test_df["state_gb"] = test_df.merge(train_df[["state_gb"]], how="left", right_index=True,
                  left_on="STATE")["state_gb"]
    return ["state_county_gb", "state_gb"]


def df_to_gdf(df):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs=4269)
    gdf['LATITUDE_NAD83'] = gdf['LATITUDE']
    gdf['LONGITUDE_NAD83'] = gdf['LONGITUDE']
    gdf['POINT_GEOMETRY_NAD83'] = gdf.geometry
    gdf = gdf.to_crs(4326)
    gdf['LATITUDE'] = gdf.geometry.y
    gdf['LONGITUDE'] = gdf.geometry.x
    gdf['POINT_GEOMETRY'] = gdf.geometry
    # gdf['BOX_GEOMETRY'] = gpd.GeoSeries(gdf['BOX_GEOMETRY'].apply(loads), crs=4269).to_crs(4326)
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


def weather_normal_features(gdf, data_path):
    raster_files = {}
    for feat, dir_name in WEATHER_FEATURES_MAP.items():
        raster_files[feat] = {}
        for i in range(1, 13):
            month = str(i) if i >= 10 else f'0{i}'
            bil_file = os.path.join(data_path, dir_name, WEATHER_FILES_MAP[dir_name].format(month))
            raster_files[feat][month] = rasterio.open(bil_file)

    def create_weather_features(row):
        month_str = row['disc_date_dt'].strftime("%m")
        results = []
        for feature in WEATHER_FEATURES_MAP:
            data_value = list(raster_files[feature][month_str].sample([(row['LONGITUDE_NAD83'], row['LATITUDE_NAD83'])]))
            results.append(data_value[0][0])
        return results

    gdf[list(WEATHER_FEATURES_MAP.keys())] = gdf.apply(create_weather_features, axis=1, result_type='expand')

def extract_features(df, train_df=None):
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY"]
    date_features_lst = date_features(df)
    if train_df:
        state_county_features = state_county_features_test(df, train_df)
    else:
        state_county_features = state_county_features_train(df)
    return basic_features_lst + date_features_lst + state_county_features


def run_model(df):
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY"]
    date_features_lst = date_features(df)

    X, y = df[date_features_lst + basic_features_lst], df[LABEL_FIELD]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # aggregative_features(X_train, X_test)
    X_train.drop(columns=["STATE", "COUNTY", "STATE_COUNTY"], inplace=True)
    X_test.drop(columns=["STATE", "COUNTY", "STATE_COUNTY"], inplace=True)

    clf_multi = RandomForestClassifier()
    clf_multi.fit(X_train, y_train)

    print("train score ", clf_multi.score(X_train, y_train))
    print("test score ", clf_multi.score(X_test, y_test))

    y_test, preds = y_test, clf_multi.predict(X_test)
    print(classification_report(y_test, preds))


def print_feature_importance(rf_model):
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = list(rf_model.feature_names_in_)
    for f in range(len(feat_labels)):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[sorted_indices[f]],
                                importances[sorted_indices[f]]))