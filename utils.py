import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import requests
import sqlite3
from config import *
import os
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

LABEL_FIELD = "STAT_CAUSE_CODE"
OID_FIELD = "OBJECTID"
null_columns = ['ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID',
                'MTBS_FIRE_NAME', 'COMPLEX_NAME']
leakadge_columns = ['FIRE_NAME', 'NWCG_REPORTING_AGENCY',
                    'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_ID',
                    'SOURCE_REPORTING_UNIT_NAME',
                    'ICS209NAME']
id_columns = ['OBJECTID', 'FOD_ID', 'FPA_ID']


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


def aggregative_features_train(train_gdf):
    # frequency per month
    months_stats = train_gdf["disc_mon"].value_counts().reset_index().rename(
        columns={"index": "disc_mon", "disc_mon": "month_freq"})
    months_stats["month_freq"] = months_stats["month_freq"] / train_gdf.shape[
        0]
    train_gdf['month_freq'] = \
        pd.merge(train_gdf, months_stats, left_on=["disc_mon"],
                 right_on=["disc_mon"])['month_freq']
    # frquency per day of week
    weekday_stats = train_gdf["disc_dow"].value_counts().reset_index().rename(
        columns={"index": "disc_dow", "disc_dow": "weekday_freq"})
    weekday_stats["weekday_freq"] = weekday_stats["weekday_freq"] / \
                                    train_gdf.shape[0]
    train_gdf['weekday_freq'] = \
        pd.merge(train_gdf, weekday_stats, left_on=["disc_dow"],
                 right_on=["disc_dow"])['weekday_freq']
    return ['month_freq', 'weekday_freq']


def aggregative_features_test(test_gdf, train_gdf):
    test_gdf['month_freq'] = \
        pd.merge(test_gdf, train_gdf[['disc_mon', 'month_freq']].drop_duplicates(), left_on=["disc_mon"],
                 right_on=["disc_mon"])['month_freq']

    test_gdf['weekday_freq'] = \
        pd.merge(test_gdf, train_gdf[['disc_dow', 'weekday_freq']].drop_duplicates(),
                 left_on=["disc_dow"],
                 right_on=["disc_dow"])['weekday_freq']

    return ['month_freq', 'weekday_freq']


def geo_vector_features(gdf):
    features = []
    for name_of_data in LAYERS:
        features.append(f"distances_{name_of_data}")
        print(name_of_data)
        path = os.path.join(DATA_PATH, LAYERS[name_of_data])
        external_gdf = gpd.read_file(path, crs="EPSG:4326")[['geometry']].drop_duplicates()
        distances = gpd.sjoin_nearest(gdf, external_gdf, how='left', lsuffix='left',
                                      rsuffix='right',
                                      distance_col=f"distances_{name_of_data}")
        distances.drop_duplicates(subset=[OID_FIELD], inplace=True)
        gdf = gdf.merge(distances[[OID_FIELD, f"distances_{name_of_data}"]], on=OID_FIELD, how='left')
    return gdf, features


def state_county_features_train(train_df):
    features = ["state_county_gb", "state_gb"]
    train_df["STATE_COUNTY"] = train_df["STATE"] + "_" + train_df["COUNTY"]
    gb_state_county = train_df[['STATE_COUNTY', OID_FIELD]].groupby(
        "STATE_COUNTY").size().rename(
        "state_county_gb")
    train_df["state_county_gb"] = \
        train_df.merge(gb_state_county, how="left", right_index=True,
                       left_on="STATE_COUNTY")["state_county_gb"]
    gb_state = train_df[['STATE', OID_FIELD]].groupby("STATE").size().rename(
        "state_gb")
    train_df["state_gb"] = \
        train_df.merge(gb_state, how="left", right_index=True,
                       left_on="STATE")["state_gb"]
    return features


def state_county_features_test(test_gdf, train_gdf):
    test_gdf["STATE_COUNTY"] = test_gdf["STATE"] + "_" + test_gdf["COUNTY"]
    test_gdf["state_county_gb"] = \
        test_gdf.merge(
            train_gdf[["STATE_COUNTY", "state_county_gb"]].drop_duplicates(),
            how="left",
            on="STATE_COUNTY")["state_county_gb"]
    test_gdf["state_gb"] = \
        test_gdf.merge(train_gdf[["state_gb", 'STATE']].drop_duplicates(),
                       how="left", on="STATE")[
            "state_gb"]
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


def fire_size_class_to_num(df):
    df["FIRE_SIZE_CLASS"] = df["FIRE_SIZE_CLASS"].map(
        {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})
    return df


def fire_duration_feature(train_gdf):
    # splitting train for empty and non empty rows
    full_times = train_gdf.dropna(
        subset=["disc_date_dt", "DISCOVERY_TIME", "cont_date_dt", "CONT_TIME"])

    # creating date columns with hours
    full_times["cont_timestamp"] = full_times["cont_date_dt"].astype(str) + " " + full_times[
        "CONT_TIME"].astype(str).apply(lambda row: row[:2] + ":" + row[2:])
    full_times["disc_timestamp"] = full_times["disc_date_dt"].astype(str) + " " + \
                                   full_times["DISCOVERY_TIME"].astype(str).apply(
                                       lambda row: row[:2] + ":" + row[2:])

    full_times["cont_timestamp"] = pd.to_datetime(full_times["cont_timestamp"])
    full_times["disc_timestamp"] = pd.to_datetime(full_times["disc_timestamp"])

    # duration in minutes
    full_times["fire_duration_min"] = full_times.apply(lambda row: pd.Timedelta(
        row["cont_timestamp"] - row["disc_timestamp"]).seconds / 60.0,
                                                       axis=1)
    # calculating average per group [month, weekday ,state, fire size class]
    group_cols = ["disc_mon", "disc_dow", "STATE", "FIRE_SIZE_CLASS"]
    avg_df = full_times.groupby(group_cols).mean(["fire_duration_min"]).reset_index()

    # merging averages and full times with train gdf
    train_gdf["avg_fire_duration_min"] = train_gdf.merge(avg_df, on=group_cols, how='left')["fire_duration_min"]
    train_gdf["fire_duration_min"] = train_gdf.join(full_times, lsuffix='_caller', rsuffix='_other')[
        "fire_duration_min"]
    train_gdf["fire_duration_min"] = train_gdf["fire_duration_min"].combine_first(train_gdf["avg_fire_duration_min"])

    return ["fire_duration_min"]


def fire_duration_feature_test(test_gdf, train_gdf):
    # splitting test for empty and non empty rows
    full_times = test_gdf.dropna(
        subset=["disc_date_dt", "DISCOVERY_TIME", "cont_date_dt", "CONT_TIME"])

    # creating date columns with hours
    full_times["cont_timestamp"] = full_times["cont_date_dt"].astype(
        str) + " " + full_times[
                                       "CONT_TIME"].astype(str).apply(
        lambda row: row[:2] + ":" + row[2:])
    full_times["disc_timestamp"] = full_times["disc_date_dt"].astype(
        str) + " " + \
                                   full_times["DISCOVERY_TIME"].astype(
                                       str).apply(
                                       lambda row: row[:2] + ":" + row[2:])

    full_times["cont_timestamp"] = pd.to_datetime(full_times["cont_timestamp"])
    full_times["disc_timestamp"] = pd.to_datetime(full_times["disc_timestamp"])

    # duration in minutes
    full_times["fire_duration_min"] = full_times.apply(
        lambda row: pd.Timedelta(
            row["cont_timestamp"] - row["disc_timestamp"]).seconds / 60.0,
        axis=1)

    # calculating average per group [month, weekday ,state, fire size class]
    group_cols = ["disc_mon", "disc_dow", "STATE", "FIRE_SIZE_CLASS"]

    # calculating average on train df because it contains more info
    avg_df = train_gdf.groupby(group_cols).mean(["fire_duration_min"]).reset_index()

    # merging averages and full times with train gdf
    test_gdf["avg_fire_duration_min"] = \
        test_gdf.merge(avg_df, on=group_cols, how='left')["fire_duration_min"]
    test_gdf["fire_duration_min"] = \
        test_gdf.join(full_times, lsuffix='_caller', rsuffix='_other')[
            "fire_duration_min"]
    test_gdf["fire_duration_min"] = test_gdf[
        "fire_duration_min"].combine_first(test_gdf["avg_fire_duration_min"])

    return ["fire_duration_min"]


def get_elevation(gdf):
    url = "https://api.opentopodata.org/v1/ned10m"
    elavation_dfs = []
    all_coords_str = gdf.geometry.y.astype(str) + "," + gdf.geometry.x.astype(
        str)
    for i in range(0, len(all_coords_str), 100):
        coords = '|'.join(all_coords_str[i:i + 100])
        data = {
            "locations": coords,
            "interpolation": "cubic"
        }
        response = requests.post(url, json=data)
        elavation_dfs.append(pd.json_normalize(response.json(), "results"))
    return pd.concat(elavation_dfs)['elevation']


def geo_raster_features(gdf):
    raster_files = {}
    for feat, dir_name in WEATHER_FEATURES_MAP.items():
        raster_files[feat] = {}
        for i in range(1, 13):
            month = str(i) if i >= 10 else f'0{i}'
            bil_file = os.path.join(WEATHER_DATA_PATH, dir_name, WEATHER_FILES_MAP[dir_name].format(month))
            raster_files[feat][month] = rasterio.open(bil_file)

    def create_weather_features(row):
        month_str = row['disc_date_dt'].strftime("%m")
        results = []
        for feature in WEATHER_FEATURES_MAP:
            data_value = list(
                raster_files[feature][month_str].sample([(row['LONGITUDE_NAD83'], row['LATITUDE_NAD83'])]))
            results.append(data_value[0][0])
        return results

    gdf[list(WEATHER_FEATURES_MAP.keys())] = gdf.apply(create_weather_features, axis=1, result_type='expand')
    return gdf, list(WEATHER_FEATURES_MAP.keys())


def plot_feature_importance(model):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = list(model.feature_names_in_)

    fi = pd.DataFrame({"labels": feat_labels, "importance": importances})
    fi = fi.sort_values('importance')
    fig = px.bar(fi, x='labels', y='importance', color='importance',
                 height=400)
    fig.show()


def extract_features_train(train_gdf):
    """function performs all basic data manipulations that are needed on both train and test dataframe"""
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY",
                          "FIRE_SIZE"]  # , "FIRE_SIZE_CLASS"]
    date_features_lst = date_features(train_gdf)
    state_county_features = state_county_features_train(train_gdf)
    fire_duration = fire_duration_feature(train_gdf)
    train_gdf, vector_features = geo_vector_features(train_gdf)
    train_gdf, raster_features = geo_raster_features(train_gdf)
    agg_features = aggregative_features_train(train_gdf)
    basic_features_lst.remove("STATE")
    basic_features_lst.remove("COUNTY")

    features = basic_features_lst + date_features_lst + state_county_features + \
               agg_features + raster_features + vector_features + fire_duration
    return train_gdf, features


def extract_features_test(test_gdf, train_gdf):
    """function performs all basic data manipulations that are needed on both train and test dataframe"""
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY",
                          "FIRE_SIZE"]  # , "FIRE_SIZE_CLASS"]
    date_features_lst = date_features(test_gdf)
    state_county_features = state_county_features_test(test_gdf, train_gdf)
    fire_duration = fire_duration_feature_test(test_gdf, train_gdf)
    test_gdf, vector_features = geo_vector_features(test_gdf)
    test_gdf, raster_features = geo_raster_features(test_gdf)
    agg_features = aggregative_features_test(test_gdf, train_gdf)
    basic_features_lst.remove("STATE")
    basic_features_lst.remove("COUNTY")

    features = basic_features_lst + date_features_lst + state_county_features + \
               agg_features + raster_features + vector_features + fire_duration
    return test_gdf, features


def fit_model(X, y):
    X = X.fillna(0)
    model = RandomForestClassifier(random_state=10, min_samples_leaf=2,
                                   min_samples_split=30, n_estimators=203)
    # model = RandomForestClassifier(random_state=10)
    model.fit(X, y)
    return model


def predict_results(X_test, model):
    X_test = X_test.fillna(0)
    return model.predict(X_test)
