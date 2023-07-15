# Script to train machine learning model.

# general libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os
import logging

# project libraries
from data import process_data
from model import train_model, compute_model_metrics, inference, slice_evaluation

# Logger file initialization
logging.basicConfig(filename='training_logs.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data
logging.info("Load data")
datapath = "../data/census_clean.csv"
data = pd.read_csv(datapath, index_col=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info("Split data for training")
train, test = train_test_split(data, test_size=0.20, random_state=101)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logging.info("Preprocess data")
X_test,y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,label='salary',training=False, encoder=encoder, lb=lb
)

# Train and save a model.
logging.info("Training model")
model = train_model(X_train,y_train)

logging.info("Saving model artifacts")
joblib.dump(model, '../model_artifacts/model.pkl')
joblib.dump(encoder, '../model_artifacts/encoder.pkl')
joblib.dump(lb, '../model_artifacts/lb.pkl')

# Evaluate model on testset
logging.info("Evaluating model on testset")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"precision: {precision: .2f}. recall: {recall: .2f}. fbeta: {fbeta: .2f}")

# Evaluate on slices
logging.info("Evaluating model on slices")
slice_evaluation(cat_features,test.reset_index(drop=True),y_test,y_pred)