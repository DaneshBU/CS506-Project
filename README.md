# CS506-Project
# Credit Risk Detection

## Problem statement
Lending money is a core part of banks and financial institutions, but it comes with inherent risk. Every time a loan is issued, there is a possibility that the borrower may fail to repay it. These events, known as loan defaults, can lead to substantial financial losses if not managed properly. Rather than manually making credit decisions on metrics such as income, employment, and credit history, we want to make a model that can automate this process. This is especially needed as the volume of loan applications increases and data gets more detailed.

---

## Description
The project will process a credit risk data set which will have details such as credit risk history, income, home ownership, employment, etc as well as an outcome if they defaulted or not. We will use this data set to train a model to determine the probability of a client defaulting on a loan. We will need to keep some of the data for testing purposes where we measure the accuracy of the model.

**Note:** We do recognize that people come from different backgrounds in their lives and putting people all into one bucket of credit risk detection may not be morally right. If someone defaulted on a loan early in their life it doesn’t mean they will default many years later (their situation could have changed) we will try to take this into account when creating our model.

---

## Project Timeline

### Week 1
- Setup Project and Clearly understand the problem and how we want to solve it  
- We should finalize what we consider as success metrics  
- Do some research on credit risk modeling  
- Setup any project structure and tools needed (Github, libraries, APIs)  
- Choose a tech stack  

### Week 2
- Explore our Data set (understand it’s features)  
- Spot relationships  
- Identify anomalies  

### Week 3
- Split data set into training, validation, and test data  
- Remove any anomalies or inconsistencies  
- Handle any missing values in the data set  
- Detect all features of data e.g. home ownership, credit history, employment, etc  

### Week 4
- Build a first working model  
- Evaluate performance  
- Document results  

### Week 5
- Try to improve on the current model (maybe make a new model)  
- Compare with previous model(s)  
- Tune hyperparameters  

### Week 6
- Continue to improve model  
- Compare all models and choose best performing one  
- Potentially start working on a frontend and backend integration  

### Week 7
- Evaluate final model on test set  
- Spot any false positives or false negatives  
- Spot any ethical and moral concerns with our models particularly on fairness  
- Continue with frontend and backend integration  

### Week 8
- Wrap up project (Full frontend and backend integration should be complete)  
- Complete Final report  
- Prepare for presentation  
- Clean up any code (make sure documentation is good)  

---

## Goal
The primary goal of this project is to build and evaluate a supervised machine learning model that predicts the probability that a loan applicant will default on a loan based on their historical credit, employment and financial data.

Essentially, the project aims to:
- Train a classification model using labeled credit risk data.  
- Use applicants' financial, employment and credit history features to predict loan default outcomes.  
- Achieve reliable predictive performance measures using ROC, precision, recall and F1 score on a test set  
- Identify and analyze the most influential features contributing to default risk.  
- Spot any ethical or moral concerns with our model and address them (try to find ways to avoid this)  

**Note:** Although the model produces a probability, a threshold can be applied by the user to convert the output into a binary decision for evaluation purposes. This is useful as different Banks and Financial Instituions will have different thresholds on probabilities of defaulting e.g. one Bank might have a probability greater than 0.6 as a default risk whereas others may have it at 0.8 or higher. 

---

## Primary Dataset
The primary dataset used for this project is the Kaggle dataset - Credit Risk Dataset. This dataset contains historical loan records for individual applicants and is well suited for our problem statement as well as for supervised learning tasks.

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

---

## Data Description
The data gives us access to borrowers' financial background, employment status and loan characteristics. The key features are as follows:

- **Income**: Applicants reported annual income  
- **Employment length**: Number of years the applicant has been employed  
- **Home ownership status**: Ownership category which includes if the person rents, owns or has put it on mortgage  
- **Loan Information**: Loan amount, interest rate and loan purpose  
- **Credit history indicators**: Measures related to applicants past credit behavior and risk  
- **Loan default outcome**: A binary label indicating whether the borrower has defaulted on the loan  

The target variable is the loan default outcome, which allows the problem to be framed as a binary classification task.

---

## Data Collection Method
The dataset would be downloaded from Kaggle and stored locally within the project repository.  
No data will be collected through surveys, scraping or APIs.

---

## Data Preprocessing
Before training our model the data will undergo several preprocessing steps to ensure the quality and consistency of the dataset:

- Cleaning invalid or inconsistent records  
- Handling missing values using appropriate imputation strategies depending on the feature type  
- Encoding categorical variables using techniques such as one hot encoding or label encoding  
- Normalizing numerical features where required  

These steps prepare the data for effective and consistent training ensuring fair evaluation of the machine learning model.

---

## Modelling Plan
We will be exploring multiple supervised learning models to compare performance and interpretability:

- Logistic Regression  
- Decision Trees  
- Random Forest  
- Gradient Boosting (XGBoost or something similar)  

These models will be evaluated using standard classification metrics and the final model giving us the best performance will be chosen for our final prototype.

------ waqars edits


# Credit Risk detection

## How to Build and Run the Code

First, install all dependencies and build the environment:


`make install`

Run the full pipeline
`make run`

Run tests
`make test`

Main script 
`final_code.py`

All outputs (plots, predictions, feature importance) are saved in:
`outputs/`

## Project Goal
The goal of this project is to predict whether a loan applicant will default using a real-world credit risk dataset.

This is a binary classification problem:

- 1 -> Default
- 0 -> No default

## Dataset
The dataset is loaded from:
`https://raw.githubusercontent.com/DaneshBU/CS506-Project/main/data/credit_risk_dataset.csv`
it includes
- Numerical features (income, loan amount, interest rate, etc.)
- Categorical features (loan grade, home ownership, etc.)
- Target variable: __loan_status__

## Data Processing
Steps performed
1. Remove duplicate rows
2. Seperate features and target (__loan_status__)
3. Automatically detect:
   - Numerical features
   - Categorical features

## Preprocessing

__Numerical features__
- Missing values -> Median imputation
- scaling -> standard scalar

__Categorical features:__
- Missing values -> Most frequent imputation
- Encoding -> One-hot encoding

## Data Splitting
The dataset is split into:
- 60% __Training__
- 20% __Validation__
- 20% __Test__
Stratified splitting is used to preserve class balance.

## Model
we use:
`XGBoost Classifier`
Why XGBoost:
- Handles tabular data very well
- Captures non-linear relationships
- Built-in regularization

## Model Configuration
Key parameters:
```
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
```

## Threshold Tuning
Instead of using default 0.5:
- We test threshold from 0.1 to 0.9
- Select the best threshold on __validation F1 score__
This improves performance for imbalanced data.

## Evaluation Metrics
We evaluate using
- Accuracy
- Prescion
- Recall 
- F1 score
- ROC AUC
- Confusion matrix

## Visualizations
The project generates the following plots:
- Roc curve
- prescion recall curve
- Validation confusion matrix
- Test confusion matrix
- Top 20 Feature Importances

All saved in
` outputs/`


## Results
The model achieves strong performance on the test set using the optimized threshold.
Outputs generated
```
outputs/XGBoost_test_predictions.csv
outputs/xgboost_feature_importance.csv
outputs/xgboost_roc_curve.png
outputs/xgboost_precision_recall_curve.png
outputs/xgboost_validation_confusion_matrix.png
outputs/xgboost_test_confusion_matrix.png
outputs/xgboost_feature_importance.png
```

## Testing
We include a small test suite using __pytest__.

Tests check:
- Dataset loads correctly
- Target column exists
- Target is binary
- Duplicate removal works
- Train/validation/test split works
- Feature types are correctly detected

Run tests:
`make test`

## Github Workflow
A GitHub Actions workflow is included:
`.github/workflows/tests.yml`
This automatically:
1. Installs dependencies
2. Run tests

Triggered on:
- Push
- Pull requests

## How Everything Connects
- __final_code.py__ → runs full pipeline (data → model → results)
- __Makefile__ → builds, runs, tests project
- __tests__ -> ensures tests pass automatically
- outputs/ -> stores all results and visualizations

## Conclusion
This project demonstrates an end-to-end machine learning pipeline:
- Data preprocessing
- Model training (XGBoost)
- Threshold tuning
- Evaluation
- Visualization
- Automated testing

The model successfully predicts loan default risk using structured financial data.

```
- Build/run instructions (first section)
- Testing + GitHub workflow
- Data + modeling explanation
- Visualizations
- Results
```


