import pandas as pd
import numpy as np
import pathlib
import sys
from sklearn.model_selection import train_test_split
import os
import yaml


# read data
def load_data(data_path):
    df=pd.read_csv(data_path)
    return df

# convert datetime
def convert_datetime(df):
    df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
    return df

# transform data
def transform(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek 
    df['day_name'] = df['pickup_datetime'].dt.day_name()   
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['is_weekend'] = df['pickup_datetime'].dt.dayofweek

    return df

def transform2(df):
    df['is_rush_hour'] = ((df['day_of_week'] <= 4) & (df['hour'].between(16, 19))).astype(int) 
    df['log_trip_duration'] = np.log1p(df['trip_duration'])
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 0 and x <= 5 else 0)   
    return df

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])            # Mesafe: Haversine (Kuş uçuşu KM)
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def add_haversine(df):
    df['dist_km'] = haversine(
        df['pickup_longitude'],
        df['pickup_latitude'],
        df['dropoff_longitude'],
        df['dropoff_latitude']
    )
    return df


def manhattan_dist(df):

    df['manhattan_dist'] = (df['pickup_latitude'] - df['dropoff_latitude']).abs() + \
                                                    (df['pickup_longitude'] - df['dropoff_longitude']).abs() 
    return df

def direction(df):
    df['direction'] = np.degrees(np.arctan2(df['dropoff_longitude'] - df['pickup_longitude'], df['dropoff_latitude'] - df['pickup_latitude']))    
    return df

def dist_transform(df):
        
    df['dist_hour_interaction'] = df['dist_km'] * df['hour']                                   
    df['dist_night_interaction'] = df['dist_km'] * df['is_night']
    return df

def split_dataset(df,target,test_size,seed):
    y=df[target]
    x=df.drop(columns=[target])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=seed)
    return x_train,x_test,y_train,y_test


def save_file(x_train,x_test,y_train,y_test,output_path):
    os.makedirs(output_path,exist_ok=True)

    x_train.to_csv(output_path+"/x_train.csv",index=False,header=True)
    x_test.to_csv(output_path+"/x_test.csv",index=False,header=True)
    y_train.to_csv(output_path+"/y_train.csv",index=False,header=True)
    y_test.to_csv(output_path+"/y_test.csv",index=False,header=True)






def main():
    curr=pathlib.Path(__file__)
    homedir=curr.parent.parent.parent
    params_path=homedir.as_posix()+"/params.yaml"
    params_file=yaml.safe_load(open(params_path))['build_features']
    output_path=homedir.as_posix()+"/data/external"

    input_path=sys.argv[1]
    input_file=homedir.as_posix()+input_path
    df=load_data(input_file)
    df1=convert_datetime(df)
    df2=transform(df1)
    df3=transform2(df2)
    df4=add_haversine(df3)
    df5=manhattan_dist(df4)
    df6=direction(df5)
    df7=dist_transform(df6)
    x_train,x_test,y_train,y_test=split_dataset(df7,params_file['target'],params_file['test_size'],params_file['seed'])
    save_file(x_train,x_test,y_train,y_test,output_path)


if __name__=="__main__":
    main()







