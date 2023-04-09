from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI


import xgboost as xgb
import numpy as np

app = FastAPI()


class item_predict(BaseModel):
    windspeed: float


class multi_item_predict(BaseModel):
    windspeed: float
    dayofyear: int
    hour: int
    dayofweek: int
    quarter: int
    month: int
    year: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_by_windspeed/")
async def predict_value(item: item_predict):
    xbg_reg = xgb.XGBRegressor()
    xbg_reg.load_model("models/model.json")
    print('model loaded')
    pred = xbg_reg.predict(np.array([item.windspeed]))
    print('prediction done')
    return {"windspeed": item.windspeed, "prediction": pred.tolist()}


@app.post("/predict_multivariable/")
async def predict_value(item: multi_item_predict):
    xbg_reg = xgb.XGBRegressor()
    xbg_reg.load_model("models/model.json")
    print('model loaded')
    pred = xbg_reg.predict(np.array([item.windspeed, item.dayofweek, item.hour, item.dayofweek, item.quarter, item.month, item.year]))
    print('prediction done')
    return {"windspeed": item.windspeed,
            "dayofyear": item.dayofyear,
            "hour": item.hour,
            "dayofweek": item.dayofweek,
            "quarter": item.quarter,
            "month": item.month,
            "year": item.year,
            "prediction": pred.tolist()[0]}
