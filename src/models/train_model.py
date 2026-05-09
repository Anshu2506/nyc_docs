from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import numpy as np
import pathlib 
import sys
import os
import joblib



# read x_train,x_test,y_test,y_train
def load_data(path1,path2,path3,path4):
    x_train=pd.read_csv(path1)
    x_test=pd.read_csv(path2)
    y_train = pd.read_csv(path3).squeeze()   
    y_test = pd.read_csv(path4).squeeze()

    return x_train,x_test,y_train,y_test


def get_model():
    models={
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "ExtraTree": ExtraTreeRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }

    return models

def train_n_evaluate(models,x_train,x_test,y_train,y_test):
    list={}
    for name,model in models.items():
        model.fit(x_train,y_train)

        y_pred=model.predict(x_test)

        mae=mean_absolute_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))

        list[model]={
            "model": model,
            "mae" : mae,
            "R2" : r2,
            "RMSE" : rmse
        }

    return list

def get_best_model(list):
    best_model = min(list, key=lambda x: list[x]["RMSE"])
    print("Best Model:" ,best_model)
    print("metrics:" ,list[best_model])
    return best_model

def save_model(best_model, model_name="best_model.pkl"):
    os.makedirs("models_matrix", exist_ok=True)
    path = os.path.join("models_matrix", model_name)

    joblib.dump(best_model, path)
    print(f" Model saved at: {path}")

def main():
    curr=pathlib.Path(__file__)
    homedir=curr.parent.parent.parent
    path1=sys.argv[1]
    path2=sys.argv[2]
    path3=sys.argv[3]
    path4=sys.argv[4]

    x_train,x_test,y_train,y_test=load_data(path1,path2,path3,path4)
    models=get_model()
    list=train_n_evaluate(models,x_train,x_test,y_train,y_test)
    best_model=get_best_model(list)
    save_model(best_model)

if __name__=="__main__":
    main()


