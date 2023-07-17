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
    assert r.json() == [
        "Welcome: This model API to predict salary of adults"]

def test_predict_under():
    """ Test post to predict under 50K """
    r = client.post("/inference/", json={
        "age": 38,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"})

    assert r.status_code == 200
    assert r.json()['prediction'] == "<=50k"

def test_predict_over():
    """ Test post to predict over 50K """
    r = client.post("/inference/", json={
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json()['prediction'] == ">50k"

    if '__name__' == '__main__':
        test_root()
        test_predict_under()
        test_predict_over()