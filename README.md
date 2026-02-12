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
The primary goal of this project is to build and evaluate a supervised machine learning model that predicts whether a loan applicant will default or not based on their historical credit, employment and financial data.

Essentially, the project aims to:
- Train a classification model using labeled credit risk data.  
- Use applicants' financial, employment and credit history features to predict loan default outcomes.  
- Achieve reliable predictive performance measures using ROC, precision, recall and F1 score on a test set  
- Identify and analyze the most influential features contributing to default risk.  
- Spot any ethical or moral concerns with our model and address them (try to find ways to avoid this)  

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
