# Script to train machine learning model.

# general libraries
import pandas as pd
import joblib
import logging

# project libraries
from data import process_data
from model import inference

# Load artifacts
model = joblib.load("../model_artifacts/model.pkl")
encoder = joblib.load("../model_artifacts/encoder.pkl")
lb = joblib.load("../model_artifacts/lb.pkl")

datapath = "../data/census_clean.csv"
data = pd.read_csv(datapath, index_col=False)

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

X, y, _, _ = process_data(
       data,
        categorical_features=cat_features,label='salary',training=False, encoder=encoder, lb=lb
    )

print(pd.DataFrame(X)[6].head(10))
print(data.head(10))
y_pred = inference(model, X)

data = data.reset_index(drop=True)
data['prediction'] = y_pred

print(data[['salary','prediction']].head(20))

