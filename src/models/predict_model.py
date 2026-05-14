import pandas as pd
import joblib
import os
import pathlib
import sys

# load data   
def load_data(data_path):
    data=pd.read_csv(data_path)
    return data

#load model
def load_model(model_path):

    model = joblib.load(model_path)

    return model

#predict
def predict(model, data):

    predictions = model.predict(data)

    return predictions

def save_prediction(output_path,predictions):
    os.makedirs(output_path,exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "prediction": predictions
        }
    )

    pred_df.to_csv(
        output_path + "/predictions.csv",
        index=False
    )

    print("✅ Predictions saved")

def main():
    curr=pathlib.Path(__file__)
    homedir=curr.parent.parent.parent

    data_path=sys.argv[1]
    data_file=homedir.as_posix()+"/"+data_path
    data=load_data(data_file)

    model_path=sys.argv[2]
    model_file=homedir.as_posix()+"/"+model_path
    model=load_model(model_file)

    predictions=predict(model,data)

    output_path=homedir.as_posix()+"/predictions"

    save_prediction(output_path,predictions)

if __name__=="__main__":
    main()