import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pathlib
import sys
import yaml


#load data 
def load_dataset(data_path):
    df=pd.read_csv(data_path)
    return df

#split data
def split_data(df,test_size,seed):
    train,test=train_test_split(df,test_size=test_size,random_state=seed)
    return train,test

#save data
def save_file(train,test,output_path):
    os.makedirs(output_path,exist_ok=True)

    train.to_csv(output_path +"/train.csv",index=False,header=True)
    test.to_csv(output_path +"/test.csv",index=False,header=True)

def main():
    curr=pathlib.Path(__file__)    
    homedir=curr.parent.parent.parent

    #load input data
    input_path=sys.argv[1]
    input_data=homedir.as_posix() + input_path

    #params file
    params_path=homedir.as_posix()+"/params.yaml"
    params_file=yaml.safe_load(open(params_path))['make_dataset']

    #gives output path to save file
    output_file=homedir.as_posix()+'/data/processed'


    df=load_dataset(input_data)
    train,test=split_data(df,params_file['test_size'],params_file['seed'])
    save_file(train,test,output_file)


if __name__=="__main__":
    main()

    