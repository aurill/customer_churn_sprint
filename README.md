### **Customer_Churn_Sprint**
Predict customer churn for Sprint, a major US telecommunications company. This Machine learning project helps retain customers & and optimize services.

## **Week 1 Project Description**:

This repository contains a comprehensive machine learning project focused on predicting customer churn for Sprint, one of the largest telecom companies in the USA. Customer churn, or the rate customers leave a service provider, is a critical concern for businesses like Sprint. Predicting churn allows telecom companies to retain valuable customers and optimize their services proactively.

In this project, we leverage historical customer data, including information about customer demographics, contract details, usage patterns, and churn status to predict Customer Churn at Sprint Technologies. 

## **Overview**

Customer churn is the percentage of customers who stopped using a company's product or service during a certain time frame. This situation is a real problem across many global industries, and the average churn rate can be surprisingly high. For some global markets, churn rates can be as high as 30%. According to an article published by [Paddle](https://www.paddle.com/resources/industry-retention-rates), Churn rates are among the highest in the United States and it is due to the challenge of switching. Loyalty programs, long-term contracts, and similar services among competitors discourage customers from leaving. Solutions to mitigating the effects of customer churn involve companies seeking to improve the quality of their customer service, issue more personalized offers, engage in competitive pricing, provide quality service, or leverage the power of machine learning to predict customer churn rates - which is what this project entails. This project employs various machine-learning techniques and models to build a robust customer churn prediction system. The ultimate goal is to develop a model that can effectively identify customers at risk of churning for the upcoming months, enabling Sprint to take proactive measures to retain them. 

We will discuss the machine-learning techniques and models that were used in the project below:

- **Ridge Classifier with Grid Search**: Ridge classifiers are suitable for binary classification problems like churn prediction. They can handle features that may be correlated, which is common in customer data. Ridge Classifier can account for potential correlations among customer attributes, helping to make accurate churn predictions.

+ **LightGBM Classifier with Bayesian Optimization**: LightGBM is a gradient-boosting framework that excels in speed and efficiency, making it an excellent choice for handling large datasets, such as those encountered in telecom customer churn prediction. It's capable of handling categorical features efficiently and it can capture complex relationships between features, making it well-suited for identifying subtle churn patterns. Its high accuracy and speed make LightGBM a valuable addition to the set of classifiers.

## **Defining a Churn Event**.

A Churn event could manifest in a variety of factors. This could stem from customer complaints where a High number of customer complaints may be indicative of underlying issues with service quality, billing, or customer support, leading to dissatisfaction and churn. Another likelihood could arise if whether or not a customer is an active member - as active members, who actively engage with Sprint's services and promotions, are more likely to stay. Inactive members may not see the value in continuing with Sprint. Our aim in this Machine Learning Project is to predict the specific features that correspond to sprint customer churn rates. 

## **The Data Collection Process**

After we have defined a churn event, we will obtain data on these factors as well as other factors. These include the customer's **ID**- a unique identifier for a sprint customer, their **Age**, the **Contract_Type** which varied from Month to Month to Annual, **Monthly_Charges** - for their sprint services charges,  **Tenure** - the number of years they have been with sprint,  **Total_Charges** - which was a product of the Monthly Charges and the Tenure, Phone_Service - which measures if the customer has phone service or not, **Multiple_Lines** - if they have multiple phone lines on the sprint network, **Online_Backup** of their sprint service data, **Active_Member** - a status to indicate whether or not they are still active on the sprint network and lastly, **Customer_Complaints** - which measures whether or not a customer has complained about a sprint product or service in the past. 

## **Data Preprocessing** 

After the Churn data had been compiled, the data preprocessing work was executed. This involved using Google Sheets to remove duplicate rows, dropping rows that had missing Customer IDs, missing customer complaints, active member data age, and other missing data. As for the missing Monthly Charges data, Tenure, and Total Charges data that were missing, imputation was done by using the formula: *Total_Charges = Monthly_Charges * Tenure* so as to preserve the number of data in the dataset. Additionally, the columns Contract Type, Phone Service, Multiple_Lines, Online_Backup, Active_Member, Churn, and Customer_Complaints contained 'Yes' and 'No' entries as well as 'True' and 'False' entries. *Google Excel's Find and Replace* feature was used for the binary encoding process by converting these categorical features into numerical representations (0 and 1). 

_For the Data Loading process in Google Colab:_

```python
# Installing the necessary packages & importing libraries

!pip install pyforest # a package that lazy imports commonly used data science and machine learning libraries
import pyforest # importing the pyforest library

drive.mount('/content/gdrive') # mounting Google Drive to the Colab environment. 
data= pd.read_excel(r'/content/gdrive/My Drive/Customer Churn/Customer Churn Main.xlsx') # reading the dataset and storing it as a Pandas DataFrame named 'data'
```
_For the binary encoding process, the following was performed_ 

1. Opening the dataset in Excel.
2. Using the "Find and Replace" feature to convert categorical values to binary (0 and 1) as follows:

   - Opening the "Find" dialog (usually Ctrl + F).
   - Going to the "Replace" tab.
   - Going into the "Find what" field, and enter the value to be replaced (e.g., "Yes", "No", "True", "False").
   - Entering "1" for True/ Yes values and "0" for False / No in the "Replace with" field.
   - Clicking "Replace All" to replace all instances.
   - Repeating this process for each column as needed.

3. Saving the modified dataset.

## **Exploratory Data Analysis (EDA)**

To analyze the correlation between all the columns in the dataset and the churn column, so as to gain insights into the relationship between the features and the target variable:

```python
data.corr()['Churn'].sort_values()
```
The results of this code provided all the essential data needed for the _Feature Engineering and selection process_. 


## **Feature Engineering & Selection**

The columns **Tenure**, **Active_Member** and **Customer_Complains** all showed positive correlation with **Churn**. **Tenure** had a correlation of _0.272682_, while **Active_Member** and **Customer_Complains** had a correlation of _0.941265_. These values indicate that customers who are active members and those who have made complaints in the past are more likely to churn. Since these features are potentially important to the model, they were added to a list named **features** and they were retained in the modeling and machine learning process. 

```python
features = ['Tenure', 'Customer Complains', 'Active_Member'] # Adding all the columns to a list called features that showed a positive correlation with churn in the data.
```

## **Model Selection**

As discussed before, various machine learning algorithms were used for the Model selection process. 

The dataset was split into train/test parts: 

```python
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Churn'], test_size = 0.25, random_state = 42)
X_train.shape, X_test.shape
```
**Firstly**, Ridge Classifier algorithm was used due to its suitability for binary classification problems like customer churn prediction and also in the handling of the features of the dataset that was found to be correlated.
This algorithm was optimized using Grid Search which improves the model's performance by identifying the combination of hyperparameters that yields the best results based on a chosen evaluation metric (e.g., accuracy, precision, recall, ROC AUC).

```python
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Defining the hyperparameter grid
param_grid = {'alpha': [0.1, 1, 10, 100]}

# Creating the Ridge Classifier
ridgeclassifier = RidgeClassifier()

# Performing grid search
grid_search = GridSearchCV(ridgeclassifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Geting the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Training the Ridge Classifier with the best hyperparameters
ridgeclassifier = RidgeClassifier(alpha=best_alpha)
ridgeclassifier.fit(X_train, y_train)

# Predicting testing values
y_pred_test = ridgeclassifier.predict(X_test)

# Calculating accuracy, recall, precison and roc_auc
accuracy = accuracy_score(y_test, y_pred_test) * 100
recall = recall_score(y_test,y_pred) * 100
precision = precision_score(y_test,y_pred) * 100
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[: , 1]) * 100

print(f'Ridge Classifier Accuracy Score : {accuracy:.3f} %')
print(f'Ridge Classifier Recall Score : {recall:.3f} %')
print(f'Ridge Classifier Precision Score : {precision:.3f} %')
print(f'Ridge Classifier roc auc Score : {roc_auc:.3f} %')
```

**Accuracy Score**: The accuracy score indicates the proportion of correctly predicted instances in the test dataset. In this case, the Ridge Classifier achieved an accuracy of approximately 96.203%, which means it correctly predicted about 96.203% of the churn outcomes in the test data.

**Recall Score**: Recall, also known as sensitivity or true positive rate, measures the proportion of actual positive cases (churned customers) that were correctly predicted as positive by the model. Here, the Ridge Classifier achieved a recall of approximately 99.275%, indicating that it correctly identified about 99.275% of the customers who actually churned.

**Precision Score**: Precision measures the proportion of true positive predictions out of all positive predictions made by the model. In this case, the Ridge Classifier achieved a precision of approximately 94.483%. This means that out of all the customers predicted as churning by the model, about 94.483% of them actually did churn.

**ROC AUC Score**: The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score is a measure of the model's ability to distinguish between positive and negative classes. It considers both the true positive rate (recall) and false positive rate. The Ridge Classifier achieved an ROC AUC score of approximately 94.675%, indicating that the model has a good ability to discriminate between churn and non-churn customers.

_These scores indicate that the Machine learning algorithm performed well in predicting customer churn with high accuracy, recall, precision, and ROC AUC values._


**Secondly**, in order to try capturing complex non-linear relationships within the data and for the purposes of cross-validation, LightGBM algorithm was also implemented. This algorithm was optimized using Bayesian Optimization to search for the best set of hyperparameters for the classifier using a brute-force search method. 

```python
!pip install scikit-optimize

from skopt import BayesSearchCV
from skopt.space import Real, Integer
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Define the search space for hyperparameters
param_space = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'max_depth': Integer(1, 32),
    'num_leaves': Integer(2, 256),
    'min_child_samples': Integer(1, 100),
    'subsample': Real(0.1, 1.0, 'uniform'),
    'colsample_bytree': Real(0.1, 1.0, 'uniform'),
    'n_estimators': Integer(50, 1000)
}

# Create a LightGBM Classifier
lgb_classifier = lgb.LGBMClassifier(verbose=-1)

# Use Bayesian optimization with cross-validation to find the best hyperparameters
opt = BayesSearchCV(
    lgb_classifier,
    param_space,
    n_iter=50,  # Number of optimization iterations
    scoring='roc_auc',  # Choosing an appropriate metric for the problem
    cv=5,  # Number of cross-validation folds
    n_jobs=-1  # Using all available CPU cores
)
opt.fit(X_train, y_train)

# Get the best hyperparameters
best_params = opt.best_params_

# Train the LightGBM Classifier with the best hyperparameters
best_lgb_classifier = lgb.LGBMClassifier(**best_params, verbose=-1)
best_lgb_classifier.fit(X_train, y_train)

# Evaluate the optimized LightGBM Classifier
y_pred_test = best_lgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test) * 100
recall = recall_score(y_test, y_pred_test) * 100
precision = precision_score(y_test, y_pred_test) * 100
roc_auc = roc_auc_score(y_test, best_lgb_classifier.predict_proba(X_test)[:, 1]) * 100

print(f'LightGBM Accuracy Score: {accuracy:.3f} %')
print(f'LightGBM Recall Score: {recall:.3f} %')
print(f'LightGBM Precision Score: {precision:.3f} %')
print(f'LightGBM ROC AUC Score: {roc_auc:.3f} %')
```

**Accuracy Score**: The accuracy score indicates the proportion of correctly predicted instances in the test dataset. In this case, the Ridge Classifier achieved an accuracy of approximately 96.203%, which means it correctly predicted about 96.203% of the churn outcomes in the test data.

**Recall Score**: Recall, also known as sensitivity or true positive rate, measures the proportion of actual positive cases (churned customers) that were correctly predicted as positive by the model. Here, the Ridge Classifier achieved a recall of approximately 99.275%, indicating that it correctly identified about 99.275% of the customers who actually churned.

**Precision Score**: Precision measures the proportion of true positive predictions out of all positive predictions made by the model. In this case, the Ridge Classifier achieved a precision of approximately 94.483%. This means that out of all the customers predicted as churning by the model, about 94.483% of them actually did churn.

**ROC AUC Score**: The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score is a measure of the model's ability to distinguish between positive and negative classes. The Ridge Classifier achieved an ROC AUC score of approximately 94.708%, indicating that the model has a good ability to discriminate between churn and non-churn customers.

_Similarly to the Ridge Classifier algorithm, these scores indicate that this Machine learning algorithm performed well in predicting customer churn with high accuracy, recall, precision, and ROC AUC values._
