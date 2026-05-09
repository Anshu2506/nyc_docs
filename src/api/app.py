from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import pathlib
import joblib


# ==========================================
# CREATE APP
# ==========================================
app = FastAPI()


# ==========================================
# LOAD MODEL
# ==========================================
curr = pathlib.Path(__file__)

homedir = curr.parent.parent.parent

model_path = homedir.as_posix() + "/models_matrix/best_model.pkl"

model = joblib.load(model_path)


# ==========================================
# INPUT SCHEMA
# ==========================================
class TaxiInput(BaseModel):

    vendor_id: int

    passenger_count: int

    pickup_longitude: float

    pickup_latitude: float

    dropoff_longitude: float

    dropoff_latitude: float

    hour: int

    day_of_week: int

    month: int

    day: int

    is_weekend: int

    is_rush_hour: int

    is_night: int

    dist_km: float

    manhattan_dist: float

    direction: float

    dist_hour_interaction: float

    dist_night_interaction: float


# ==========================================
# HOME ROUTE
# ==========================================
@app.get("/")
def home():

    return {
        "message": "NYC Taxi Duration API"
    }


# ==========================================
# PREDICT ROUTE
# ==========================================
@app.post("/predict")
def predict(data: TaxiInput):

    # convert to dataframe
    df = pd.DataFrame([data.dict()])

    # prediction
    pred = model.predict(df)

    # inverse log transform
    prediction = np.expm1(pred[0])

    return {
        "prediction": float(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

