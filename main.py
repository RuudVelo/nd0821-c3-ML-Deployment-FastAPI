# Put the code for your API here.
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import uvicorn

# project libraries
from model.data import process_data
from model.model import inference


# Instantiate the app.
app = FastAPI()

# Welcome users
@app.get("/")
async def say_welcome():
    return {"Welcome: This model API to predict salary of adults"}


# Using of census_clean.csv namings
class InputData(BaseModel):
    age: Union[float, int] = Field(..., example=44)
    workclass: str = Field(..., example="Private")
    fnlgt: Union[float, int] = Field(..., example=83311)
    education: str = Field(..., example="11th")
    education_num: Union[float, int] = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Divorced")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: Union[float, int] = Field(..., alias="capital-gain", example=2174)
    capital_loss: Union[float, int] = Field(..., alias="capital-loss", example=0)
    hours_per_week: Union[float, int] = Field(..., alias="hours-per-week", example=50)
    native_country: str = Field(..., alias="native-country", example="Cuba")


# Send data via POST to API
@app.post("/inference/")
async def predict(inference: InputData):

    dict_input = inference.dict(by_alias=True)
    # Create sample df for inference
    sample_df = pd.DataFrame(dict_input, index=[0])

    # Generate the categorical features for transformation
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Load artifacts
    model = joblib.load("model_artifacts/model.pkl")
    encoder = joblib.load("model_artifacts/encoder.pkl")
    lb = joblib.load("model_artifacts/lb.pkl")

    X, _, _, _ = process_data(
       sample_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # predict input
    pred = int(model.predict(X))

    # convert prediction and add to input inference dataset
    prediction = ">50k" if pred == 1 else "<=50k"
    dict_input["prediction"] = prediction

    return dict_input


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
