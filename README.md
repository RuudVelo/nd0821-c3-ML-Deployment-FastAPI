# Deploying a Machine Learning Model on Heroku with FastAPI
This repository contains the third project of the Udacity ML DevOps Engineer course. 

* The link to Github is: https://github.com/RuudVelo/nd0821-c3-ML-Deployment-FastAPI

* The Heroku app is not live (original url was "https://ml-udacity-salary-app.herokuapp.com/inference/"), because of costs. For screenshots of the app live see: https://github.com/RuudVelo/nd0821-c3-ML-Deployment-FastAPI/tree/master/screenshots

# Project setup

* A ML model was trained using the census data from the Census Income Dataset from UCI. Goal was to predict the salary of adults as either <=50k per year or >50k. For further information see the model_card.md
* Inference of the model was facilitated with FastAPI and served by Heroku
* Basic unit tests and API tests were set up (can be found in the tests folder)
* CI/CD was setup using GitHub actions and connected with Heroku

# Model training

* The model can be trained with
``` 
python model/train_model.py
```
The model artifacts are stored in the model_artifacts folder

Tests can be found in the tests folder in the tests.py file

# API 

* The app is setup using FastAPI. The code can be found in the root directory in the main.py file
* For Heroku the Procfile, requirements.txt, python scripts and the saved model_artifacts are important
* The api_sample.py file in the root folder can be used as an example inference dataset. It will post the sample dataset to the Heroku app and returns the status code, prediction of the model and some metadata of the defined example 

# Instructions from Udacity for the project

Working in a command line environment is recommended for ease of use with git. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.
