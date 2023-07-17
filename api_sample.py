import requests
import json

# Heroku app url
url = "https://ml-udacity-salary-app-743b155e693f.herokuapp.com/inference/"

# sample for prediction
example =  {"age": 42,
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
            "native-country": "United-States"}

# POST to API and generate the response
response = requests.post(url, json=example )

# Show the response along with example metadata
print("Response status code", response.status_code)
print("response total:")
print(response.json())
