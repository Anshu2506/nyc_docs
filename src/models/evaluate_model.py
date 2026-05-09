import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import os
import pathlib
import json
import sys

# load data
def load_data(x_test_path,y_test_path):
    x_test=pd.read_csv(x_test_path)
    y_test=pd.read_csv(y_test_path)

    return x_test,y_test


# load model
def load_model(model_path):

    model = joblib.load(model_path)

    return model

# evaluate model
def evaluate_model(
    model,
    X_test,
    y_test
):

    # prediction
    y_pred = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return metrics



# ==========================================
# SAVE METRICS
# ==========================================

def save_metrics(
    metrics,
    output_path
):

    os.makedirs(
        output_path,
        exist_ok=True
    )

    metrics_path = (
        output_path + "/metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("✅ Metrics saved")


def main():
    curr=pathlib.Path(__file__)
    homedir=curr.parent.parent.parent


    x_test_path=sys.argv[1]
    x_test_file=homedir.as_posix() + x_test_path
    y_test_path=sys.argv[2]
    y_test_file=homedir.as_posix() + y_test_path
    x_test,y_test=load_data(x_test_file,y_test_file)

    model_path=sys.argv[3]
    model_file=homedir.as_posix()+ model_path
    model=load_model(model_file)

    metrics=evaluate_model(model,x_test,y_test)

    output_path = (
        homedir.as_posix()+ "/reports"
    )

    save_metrics(metrics,output_path)

if __name__=="__main__":
    main()

