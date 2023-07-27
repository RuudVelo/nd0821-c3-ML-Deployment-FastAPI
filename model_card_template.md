# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project is the third of the Udacity Machine Learning Devops Engineer Nanodegree program. Goal was to build and deploy an app on Heroku using various ML engineering tools and principles. The model (LogisticRegression) that is used, serves as a baseline to iterate further.  

## Intended Use
It can be used for illustration purposes, not for commercial use. 
## Training Data
Data coming from Census Income Dataset from UCI and can be found here: https://archive.ics.uci.edu/dataset/20/census+income
## Evaluation Data
The dataset was split in training, validation data with a random_state of 101 to reproduce results. The validation set compromised 20% of the total dataset. The training part was used in a Grid Search Cross-Validation procedure
## Metrics
The model was trained using hyperparameter tuning for accuracy (0.85) as target metric. Further metrics evaluated are:

precision:  0.74. recall:  0.60. fbeta:  0.66

These results represent the test set which was not used for training nor evaluation

## Ethical Considerations
Data is open sourced on UCI machine learning repository for educational purposes, so don't use this for commercial settings or explanations. 
## Caveats and Recommendations
Model quality can be improved a lot. That was however not the purpose of this project. Feature optimization and other model architectures will definitely lead to better results. When looking at results for slices (segments like sex and race), it can be seen some slices perform better than other. The segment results can be found in the model/slice_output.txt file.
