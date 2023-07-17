""" Testing of root and end point """
from fastapi.testclient import TestClient
import json


# Import the app from main
from main import app

# Instantiate testing client
client = TestClient(app)


def test_root():
    """ Test the root page for a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Welcome to this model API to predict salary of adults"}


def test_predict_under():
    """ Test post to predict under 50K """

    r = client.post("/inference", json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-famil",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per_week": 40,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json()['prediction'] == "<=50K"


def test_predict_over():
    """ Test post to predict over 50K """
    r = client.post("/inference", json={
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "AProf-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per_week": 40,
        "native-country": "India"
    })

    assert r.status_code == 200
    assert r.json()['prediction'] == ">50K"

    if '__name__' == '__main__':
        test_root()
        test_predict_under()
        test_predict_over()