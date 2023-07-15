import pytest
import os 
import logging 
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from model.model import inference, compute_model_metrics
from model.data import process_data

"""
Define fixtures
"""

@pytest.fixture(scope="module")
def dataset():
    """
    Load CSV file and return it 
    """
    path = "./data/census_clean.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        logging.error("File for modelling is not found")
        raise err

    # Check the df shape
    if df.empty:
        error_message = "File for modelling is empty"
        logging.error(error_message)
        raise AssertionError(error_message)

    return df

@pytest.fixture(scope="module")
def model_file():
    """
    Load model file and return it 
    """
    model_path = "./model_artifacts/model.pkl"
    if os.path.isfile(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as err:
            logging.error("model: error loading model")
            raise err
    else:
        raise FileNotFoundError(f"model: file '{model_path}' not found")

    return model


@pytest.fixture(scope="module")
def encoder_file():
    """
    Load encoder file and return it 
    """
    encoder_path = "./model_artifacts/encoder.pkl"
    if os.path.isfile(encoder_path):
        try:
            encoder = joblib.load(encoder_path)
        except Exception as err:
            logging.error("encoder: error loading encoder")
            raise err
    else:
        raise FileNotFoundError(f"encoder: file '{encoder_path}' not found")

    return encoder

@pytest.fixture(scope="module")
def lb_file():
    """
    Load labelbinarizer file and return it 
    """
    lb_path = "./model_artifacts/lb.pkl"
    if os.path.isfile(lb_path):
        try:
            lb = joblib.load(lb_path)
        except Exception as err:
            logging.error("lb: error loading labelbinarizer")
            raise err
    else:
        raise FileNotFoundError(f"lb: file '{lb_path}' not found")

    return lb

@pytest.fixture(scope="module")
def features():
    """
   Generate categorical features
    """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

@pytest.fixture(scope="module")
def training_dataset(dataset, features):
    """
    Dataset X_train for test
    """
    x_train, x_test = train_test_split(dataset, 
                                test_size=0.20, 
                                random_state=101, 
                                )
    X_train, y_train, encoder, lb = process_data(
                                            x_train,
                                            categorical_features=features,
                                            label='salary',
                                            training=True,
                                            )
    return X_train, y_train

"""
Test units
"""

def test_features(dataset, features):
    """
    Test if features are present in the data
    """
    missing_features = sorted(set(features) - set(dataset.columns))
    if missing_features:
        error_message = "features: missing features"
        logging.error(error_message)
        raise AssertionError(error_message)
    

def test_load_model():
    """
    Test for loading model
    """
    model_path = "../model_artifacts/model.pkl"
    if os.path.isfile(model_path):
        try:
            _ = joblib.load(model_path)
        except Exception as err:
            logging.error("model: no model")
            raise err

def test_load_encoder():
    """
    Test for loading encoder
    """
    encoder_path = "../model_artifacts/encoder.pkl"
    if os.path.isfile(encoder_path):
        try:
            _ = joblib.load(encoder_path)
        except Exception as err:
            logging.error("encoder: no encoder")
            raise err

def test_loading_lb():
    """
    Test for loading LabelBinarizer
    """
    lb_path = "../model_artifacts/lb.pkl"
    if os.path.isfile(lb_path):
        try:
            _ = joblib.load(lb_path)
        except Exception as err:
            logging.error("lb: no labelbinarizer")
            raise err

def test_model_type(model_file):
    """ 
    Check model type 
    """
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    expected_model = GridSearchCV(LogisticRegression(random_state=101), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=9)

    assert isinstance(model_file, type(expected_model))

def test_inference(training_dataset, model_file):
    """
    Check inference
    """
    X_train, y_train = training_dataset

    try:
        preds = inference(model_file, X_train)
        assert len(preds) == len(X_train)
    except Exception as err:
            logging.error(
            "No inference can be done with train data and model")
            raise err
    
def test_compute_model_metrics(model_file, training_dataset):
    """
    Check the functions of performance metric calculations
    """
    X_train, y_train = training_dataset

    preds = inference(model_file, X_train)

    try:
        precision, recall, fbeta = compute_model_metrics(y_train, preds)
    except Exception as err:
        logging.error(
        "No performance metrics can be calculated on the training dataset")
        raise err