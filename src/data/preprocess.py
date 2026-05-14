import pandas as pd
import os
import pathlib
import sys


def load_train(train_path):
    train=pd.read_csv(train_path)
    return train

def load_test(test_path):
    test=pd.read_csv(test_path)
    return test

def missing_value(train, test):
    # compute from train only
    numeric_cols = train.select_dtypes(include=['number']).columns

    medians = train[numeric_cols].median()

    train[numeric_cols] = train[numeric_cols].fillna(medians)
    test[numeric_cols] = test[numeric_cols].fillna(medians)

    return train, test

def save_file(train,test,output_file):
    os.makedirs(output_file,exist_ok=True)

    train.to_csv(output_file+"/train.csv",index=False,header=True)
    test.to_csv(output_file+"/test.csv",index=False,header=True)

def main():
    curr=pathlib.Path(__file__)
    homedir=curr.parent.parent.parent
    train_path=sys.argv[1]
    train_file=homedir.as_posix()+"/"+train_path
    test_path=sys.argv[2]
    test_file=homedir.as_posix()+"/"+test_path
    output_path=homedir.as_posix()+"/data/interim"

    train=load_train(train_file)
    test=load_test(test_file)
    train,test=missing_value(train,test)
    save_file(train,test,output_path)

if __name__=="__main__":
    main()