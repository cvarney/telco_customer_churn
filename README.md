**Telco Customer Churn Project**

**Author:** Christopher Varney

# Introduction

# Dataset Description

# Exploratory Data Analysis

Numerical Properties

In the Figure below we examine the distributions of our numerical data:
![Numerical Data Distributions](https://github.com/cvarney/telco_customer_churn/blob/main/Numerical.png?raw=true)

In the Table below we see the count, mean, and standard deviation for numerical categories provided a customer has left the company: `tenure`, `MonthlyCharges`, `TotalCharges` and `NumServices`.

| Churn = Yes |    tenure |   MonthlyCharges |   TotalCharges |   NumServices |
|:------|----------:|-----------------:|---------------:|--------------:|
| count | 1869      |        1869      |        1869    |    1869       |
| mean  |   17.9791 |          74.4413 |        1531.8  |       3.61691 |
| std   |   19.5311 |          24.6661 |        1890.82 |       1.60994 |

In the Table below we see the count, mean, and standard deviation for numerical categories provided a customer has stayed with the company: `tenure`, `MonthlyCharges`, `TotalCharges` and `NumServices`.

|  Churn = No     |    tenure |   MonthlyCharges |   TotalCharges |   NumServices |
|:------|----------:|-----------------:|---------------:|--------------:|
| count | 5163      |        5163      |       5163     |    5163       |
| mean  |   37.65   |          61.3074 |       2555.34  |       3.76448 |
| std   |   24.0769 |          31.0946 |       2329.46  |       2.24953 |

Here we see that the average tenure is 17.98 months for a customer that has left and is 37.65 months for a customer that has stayed with the company. We also note that the monthly charges are smaller for customers that remain, but the number of services subscribed to by the customer are similar. The total charges are increased for customers that remained, but that is clearly due ot the increased tenure.


## Demographic Features
In the Figure below, I show a histogram of features and the Churn rate for each feature belonging to Demographics. Gender appears to have no impact on Churn rate, but the Churn rate is higher if the customer does not have a `Partner` or `Dependents`. Additionaly, the Churn rate is higher if the customer is a `SeniorCitizen`.
![Demographic Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Demographic.png?raw=true)

## Account Features
In the Figure below, I show a histogram of features including the Churn rate. For `tenure`, we see that the Churn rate is high if the tenure is short. Additionally, customers who select `PaperlessBilling`, `Month-to-Month` contracts, and `Electronic check` as `PaymentMethod` have higher Churn rates.

![Account Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Account.png?raw=true)

## Service Features
In the Figure below, I show a histogram of the Churn rate for each feature belonging to Service. Customers who subscribe to `PhoneService` and `Fiber optic` `InternetService` are more likely to Churn. Customers who do not subscribe to `OnlineBaackup`, `OnlineSecurity`, or `TechSupport` have higher Churn rates. `StreamingTV` and `StreamingMovies` have similar Churn rates. In terms of number of services, customers who subscribe to 1, 7, or 8 services have a lower Churn rate.
![Service Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Services.png?raw=true)

## Mutual Information Score for Feature Importance
![Mutual Information Score](https://github.com/cvarney/telco_customer_churn/blob/main/MutualInfo.png?raw=true)

# Methodology
To analyze this data set, we will consider models for binary classification: Perceptron, LogisticRegresion, Support Vector Machines (SVM), $k$-Nearest Neighbors (KNN), and Random Forest. 


## Model Comparison
To compare the models, I ran a 5-fold cross-validation on the default settings for each model. Those results are shown in the table below:

|  Model     | Accuracy | F1 Score | Precision | Recall |
|:-----------|---------:|---------:|----------:|-------:|
| Perceptron | 0.7196 |  0.4161 | 0.5586 | 0.5029 |
| LogisticRegression | 0.8030 | 0.5843 | 0.6475 | 0.5328 |
| SVM | 0.7994 | 0.5534 | 0.6573 | 0.4788 |
| KNN | 0.7622 | 0.5275 | 0.5459 | 0.5109 |
| Random Forest | 0.7850 | 0.5274 | 0.6162 | 0.4620 |

From this cross validation, we see that the `LogisticRegression` model has the best classification accuracy on the training set. 

## Hyperparameter Tuning
Hyperparameters were tuned by varying the inverse regularization strength $C$ and the solver with the following parameter options:
```
grid_parameters = {
    'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10.0, 50.0, 100.0, 500.0, 1000.0],
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag'],
    'max_iter': [1000, 10000]
}
```

I utilized 10-fold `GridSearchCV` and determined the best parameters to be the `liblinear` solver and $C=5$. Hyperparameter tuning improved the performance of the model on the training set to 80.9%.

# Results & Discussion
In order to evaluate our model on the test data, I calculated the accuracy, f1 score, precision, and recall:

| Accuracy | F1 Score | Precision | Recall |
|---------:|---------:|----------:|-------:|
| 0.792    |  0.575   | 0.683     | 0.497  |

Here we note that the accuracy of the model on the test set is 79.2%, which is lower than the accuracy on the training set 80.9%. This is 

Additionally, I computed the confusion matrix, which is shown below:

![Confusion Matrix](https://github.com/cvarney/telco_customer_churn/blob/main/CM.png?raw=true)

Overall the model had 1144 true negatives and 248 true positives, for a total of 1392 correct classifications with 366 misclassifications (115 false positives and 251 false negatives). 


Feature Importance

![Importance](https://github.com/cvarney/telco_customer_churn/blob/main/Importance.png?raw=true)

# Conclusion
