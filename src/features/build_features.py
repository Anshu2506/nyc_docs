import pandas as pd
import numpy as np
import pathlib
import sys
import os
import yaml

from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATA
# ==========================================
def load_train(train_path):

    train = pd.read_csv(train_path)

    return train


# ==========================================
# CONVERT DATETIME
# ==========================================
def convert_datetime(df):

    df['pickup_datetime'] = pd.to_datetime(
        df['pickup_datetime']
    )

    return df


# ==========================================
# BASIC TIME FEATURES
# ==========================================
def time_features(df):

    df['hour'] = df['pickup_datetime'].dt.hour

    df['day_of_week'] = (
        df['pickup_datetime'].dt.dayofweek
    )

    df['month'] = (
        df['pickup_datetime'].dt.month
    )

    df['day'] = (
        df['pickup_datetime'].dt.day
    )

    # weekend flag
    df['is_weekend'] = (
        df['day_of_week'].isin([5, 6])
    ).astype(int)

    # night flag
    df['is_night'] = (
        df['hour'].between(0, 5)
    ).astype(int)

    # rush hour
    df['is_rush_hour'] = (
        (
            df['day_of_week'] <= 4
        )
        &
        (
            df['hour'].between(16, 19)
        )
    ).astype(int)

    return df


# ==========================================
# TARGET TRANSFORM
# ==========================================
def target_transform(df):

    df['log_trip_duration'] = np.log1p(
        df['trip_duration']
    )

    return df


# ==========================================
# HAVERSINE DISTANCE
# ==========================================
def haversine(
    lon1,
    lat1,
    lon2,
    lat2
):

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin((lon2 - lon1) / 2) ** 2
    )

    return (
        6371
        *
        2
        *
        np.arcsin(np.sqrt(a))
    )


def add_haversine(df):

    df['dist_km'] = haversine(
        df['pickup_longitude'],
        df['pickup_latitude'],
        df['dropoff_longitude'],
        df['dropoff_latitude']
    )

    return df


# ==========================================
# MANHATTAN DISTANCE
# ==========================================
def manhattan_distance(df):

    df['manhattan_dist'] = (
        (
            df['pickup_latitude']
            -
            df['dropoff_latitude']
        ).abs()
        +
        (
            df['pickup_longitude']
            -
            df['dropoff_longitude']
        ).abs()
    )

    return df


# ==========================================
# DIRECTION FEATURE
# ==========================================
def direction_feature(df):

    df['direction'] = np.degrees(
        np.arctan2(
            (
                df['dropoff_longitude']
                -
                df['pickup_longitude']
            ),
            (
                df['dropoff_latitude']
                -
                df['pickup_latitude']
            )
        )
    )

    return df


# ==========================================
# INTERACTION FEATURES
# ==========================================
def interaction_features(df):

    df['dist_hour_interaction'] = (
        df['dist_km']
        *
        df['hour']
    )

    df['dist_night_interaction'] = (
        df['dist_km']
        *
        df['is_night']
    )

    return df


# ==========================================
# REMOVE UNUSED COLUMNS
# ==========================================
def remove_unused_columns(df):

    # object/string columns
    object_cols = df.select_dtypes(
        include=['object', 'string']
    ).columns

    # datetime columns
    datetime_cols = df.select_dtypes(
        include=['datetime64[ns]']
    ).columns

    drop_cols = (
        list(object_cols)
        +
        list(datetime_cols)
    )

    df = df.drop(
        columns=drop_cols,
        errors='ignore'
    )

    return df


# ==========================================
# CLEAN DATA
# ==========================================
def clean_data(df):

    # remove inf values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # remove NaN rows
    df = df.dropna()

    return df


# ==========================================
# SPLIT DATASET
# ==========================================
def split_dataset(
    df,
    target,
    test_size,
    seed
):

    y = df[target]

    X = df.drop(
        columns=[target]
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================
# SAVE FILES
# ==========================================
def save_file(
    X_train,
    X_test,
    y_train,
    y_test,
    output_path
):

    os.makedirs(
        output_path,
        exist_ok=True
    )

    X_train.to_csv(
        output_path + "/x_train.csv",
        index=False
    )

    X_test.to_csv(
        output_path + "/x_test.csv",
        index=False
    )

    y_train.to_csv(
        output_path + "/y_train.csv",
        index=False
    )

    y_test.to_csv(
        output_path + "/y_test.csv",
        index=False
    )

    print("✅ Feature files saved")


# ==========================================
# MAIN
# ==========================================
def main():

    curr = pathlib.Path(__file__)

    homedir = curr.parent.parent.parent

    # params
    params_path = (
        homedir.as_posix() + "/params.yaml"
    )

    params_file = yaml.safe_load(
        open(params_path)
    )['build_features']

    # input train file
    train_path = sys.argv[1]

    train_file = (
        homedir.as_posix() +"/"+ train_path
    )

    # output path
    output_path = (
        homedir.as_posix() + "/data/external"
    )

    # ======================================
    # LOAD
    # ======================================
    train = load_train(train_file)

    # ======================================
    # FEATURE ENGINEERING
    # ======================================
    train = convert_datetime(train)

    train = time_features(train)

    train = target_transform(train)

    train = add_haversine(train)

    train = manhattan_distance(train)

    train = direction_feature(train)

    train = interaction_features(train)

    # ======================================
    # REMOVE BAD COLUMNS
    # ======================================
    train = remove_unused_columns(train)

    # ======================================
    # CLEAN DATA
    # ======================================
    train = clean_data(train)

    # ======================================
    # SPLIT
    # ======================================
    X_train, X_test, y_train, y_test = (
        split_dataset(
            train,
            params_file['target'],
            params_file['test_size'],
            params_file['seed']
        )
    )

    # ======================================
    # SAVE
    # ======================================
    save_file(
        X_train,
        X_test,
        y_train,
        y_test,
        output_path
    )


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":

    main()