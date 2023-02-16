import pandas as pd
import geopandas as gpd
import numpy as np
# import rasterio
import requests
import sqlite3
from shapely.wkt import loads
from config import *
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, \
    balanced_accuracy_score
from scipy.stats import randint

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
        df[f"distances_{name_of_data}"] = \
            df[[OID_FIELD]].merge(
                gdf[[OID_FIELD, f"distances_{name_of_data}"]],
                left_on=OID_FIELD, right_on=OID_FIELD)[
                f"distances_{name_of_data}"]
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


def weather_normal_features(gdf, data_path):
    raster_files = {}
    for feat, dir_name in WEATHER_FEATURES_MAP.items():
        raster_files[feat] = {}
        for i in range(1, 13):
            month = str(i) if i >= 10 else f'0{i}'
            bil_file = os.path.join(data_path, dir_name,
                                    WEATHER_FILES_MAP[dir_name].format(month))
            raster_files[feat][month] = rasterio.open(bil_file)


def create_weather_features(row):
    month_str = row['DISCOVERY_DATE'].strftime("%m")
    results = []
    for feature in WEATHER_FEATURES_MAP:
        data_value = list(raster_files[feature][month_str].sample(
            [(row['LONGITUDE_NAD83'], row['LATITUDE_NAD83'])]))
        results.append(data_value[0][0])
    return results


def print_feature_importance(rf_model):
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = list(rf_model.feature_names_in_)
    for f in range(len(feat_labels)):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[sorted_indices[f]],
                                importances[sorted_indices[f]]))


def extract_features_train(train_gdf):
    """function performs all basic data manipulations that are needed on both train and test dataframe"""
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY",
                          "FIRE_SIZE"]  # , "FIRE_SIZE_CLASS"]
    date_features_lst = date_features(train_gdf)
    state_county_features = state_county_features_train(train_gdf)
    return basic_features_lst + date_features_lst + state_county_features


def extract_features_test(test_gdf, train_gdf):
    """function performs all basic data manipulations that are needed on both train and test dataframe"""
    basic_features_lst = ["LONGITUDE", "LATITUDE", "STATE", "COUNTY",
                          "FIRE_SIZE"]  # , "FIRE_SIZE_CLASS"]
    date_features_lst = date_features(test_gdf)
    state_county_features = state_county_features_test(test_gdf, train_gdf)
    return basic_features_lst + date_features_lst + state_county_features


def fit_model(X, y):
    model = RandomForestClassifier(random_state=10)

    dists = {
        'n_estimators': randint(2, 256),
        'min_samples_leaf': randint(2, 16),
        'min_samples_split': randint(2, 32)
    }

    rs = RandomizedSearchCV(estimator=model, param_distributions=dists, cv=5,
                            verbose=1,
                            scoring=make_scorer(balanced_accuracy_score),
                            n_iter=n_configs)
    rs.fit(X, y)
    return rs.best_estimator_


def predict_results(X_test, model):
    return model.predict


def main():
    # fill me
    train_df = create_dataframe()
    train_df = train_df[100:]
    test_df = train_df[:100]
    train_gdf, test_gdf = df_to_gdf(train_df), df_to_gdf(test_df)
    train_features, test_features = extract_features_train(train_gdf), \
                                    extract_features_test(test_gdf, train_gdf)
    model = fit_model(train_gdf[train_features], train_gdf[LABEL_FIELD])
    test_results = predict_results(test_gdf[test_features], model)

# Noa:
# Cont time
# Exploratory Data Analysis
# Document
#
# Tamar:
# State County Features
# Model
#
# Micha:
# Geo Features + External Features
# Geo Graphs for EDA
