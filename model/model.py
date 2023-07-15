# general libraries
import os
import logging
import multiprocessing

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(filename='training_logs.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Create the Logistic Regression model
    logistic_regression = LogisticRegression(random_state=101)

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # multiprocessing to speed up
    njobs = multiprocessing.cpu_count() - 1

    # Create the GridSearchCV object
    cls_gs = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy', n_jobs=njobs)

    # Fit the model on the training data
    cls_gs.fit(X_train, y_train)

    logging.info("******* Best parameters gridsearch *******")
    logging.info("best parameters: {}".format(cls_gs.best_params_))
    logging.info("best score: {:.2f}".format(cls_gs.best_score_))

    return cls_gs

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Logistic regression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds


# Make sliced inferences on categorical features
def slice_evaluation(cat_features, data, y, predictions):
    
    """ Run slice evaluations.

    Inputs
    ------
    cat_features : List with categorical features
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    sliced_metrics = []

    for cat in cat_features:
        for cls in data[cat].unique():
            df_temp = data[data[cat] == cls]
            index_cat = df_temp.index
            y_slice = y[index_cat]
            y_slice_pred = predictions[index_cat]

            precision, recall, fbeta = compute_model_metrics(y_slice, y_slice_pred)
            row = f"{cat} - {cls} : precision: {precision: .2f}. recall: {recall: .2f}. fbeta: {fbeta: .2f}"
            sliced_metrics.append(row)

    with open('../model/slice_output.txt', 'w') as file:
        for row in sliced_metrics:
            file.write(row + '\n')
