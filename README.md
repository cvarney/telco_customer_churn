**Telco Customer Churn Project**

**Author:** Christopher Varney

# Introduction
In this project we are analyzing a dataset containing customer information for a fictional telecommunications company. The data set contains customer information about:
* what services the customer has signed up for
* whether a customer left the company in the last month, called `Churn`
* account information
* demographic information

The objective is to build a model that will predict what characteristics result in customer churn and allow for the development of customer retention programs.

# Dataset Description
The dataset contains 21 features and 7043 entries. The target feature is a Yes/No variable called `Churn`, and the remaining features can be broken up into 3 categories:

* Demographic: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* Account: `customerID`, `tenure`, `Contract`, `PaperlessBilling`, `Payment Method`, `MonthlyCharges`, `TotalCharges`
* Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

Upon examining the data, the `customerID` column can be removed as it is unique to each customer and has no analytical use. Next, I examined the data type for each feature and discovered that the `TotalCharges` feature was labeled as an `object` instead of `float64`. I converted the column to a float and removed 11 rows corresponding to a data mismatch in this column.

In order to capture whether the number of subscribed services was important, I added an aggregate feature `NumServices` which summed all `Yes` responses. Because the `InternetService` feature contained two service types (`DSL` and `Fiber optic`), I created an intermediate variable by mapping these responses to a `Yes`.

# Exploratory Data Analysis
## Numerical Features
First, let us consider the numerical quantities: `tenure`, `MonthlyCharges`, `TotalCharges`, and `NumServices`.

In the Figure below we examine the distributions of our numerical data as a histogram with the Churn rate stacked on each feature:
![Numerical Data Distributions](https://github.com/cvarney/telco_customer_churn/blob/main/Numerical.png?raw=true)

We observe that the Churn rate is highest for low tenure and total charges. For the number of services, the Churn rate is highest when the number of services is 2, and decreases linearly as the number of services decreases. Both 1 and 8 services have low Churn rates.

This information is aggregated for `Churn = Yes` and `Churn = No` in the two tables below, with the count, mean, and standard deviation shown:

| `Churn = Yes` |    tenure |   MonthlyCharges |   TotalCharges |   NumServices |
|:------:|:----------:|-----------------:|---------------:|--------------:|
| count  |  1869      |        1869      |        1869    |    1869       |
| mean   |    17.9791 |          74.4413 |        1531.8  |       3.61691 |
| std    |    19.5311 |          24.6661 |        1890.82 |       1.60994 |

|  `Churn = No`     |    tenure |   MonthlyCharges |   TotalCharges |   NumServices |
|:------|----------:|-----------------:|---------------:|--------------:|
| count | 5163      |        5163      |       5163     |    5163       |
| mean  |   37.65   |          61.3074 |       2555.34  |       3.76448 |
| std   |   24.0769 |          31.0946 |       2329.46  |       2.24953 |

Here we see that the average tenure is 17.98 months for a customer that has left and is 37.65 months for a customer that has stayed with the company. We also note that the monthly charges are smaller for customers that remain, but the number of services subscribed to by the customer are similar. The total charges are increased for customers that remained, but that is clearly due ot the increased tenure.

## Categorical Features
### Demographic Features
In the Figure below, I show a histogram of the Churn rate for each value of each feature belonging to Demographics. Gender appears to have no impact on Churn rate, but the Churn rate is higher if the customer does not have a `Partner` or `Dependents`. Additionaly, the Churn rate is higher if the customer is a `SeniorCitizen`.
![Demographic Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Demographic_Prop.png?raw=true)

### Account Features
In the Figure below, I show a histogram of the Churn rate for each value of each feature belonging to Service. Customers who select `PaperlessBilling`, `Month-to-Month` contracts, and `Electronic check` as `PaymentMethod` have higher Churn rates.

![Account Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Account_Prop.png?raw=true)

### Service Features
In the Figure below, I show a histogram of the Churn rate for each value of each feature belonging to Service. Customers who subscribe to `PhoneService` and `Fiber optic` as their `InternetService` are more likely to Churn. Customers who do not subscribe to `OnlineBaackup`, `OnlineSecurity`, or `TechSupport` have higher Churn rates. `StreamingTV` and `StreamingMovies` have similar Churn rates. 
![Service Category Churn](https://github.com/cvarney/telco_customer_churn/blob/main/Services_Prop.png?raw=true)

### Mutual Information Score for Feature Importance
In order to gain a basic understanding of the importance of each categorical feature, I calculated the mutual information for each categorical variable with the Churn variable using `mutual_info_score` from `sklearn.metrics`. This metric measures the similarity between the two labels and I regard it as a preliminary assessment. 
![Mutual Information Score](https://github.com/cvarney/telco_customer_churn/blob/main/MutualInfo.png?raw=true)

Here we see that the `Contract`, `OnlineSecurity`, `TechSupport`, `InternetService`, `OnlineBackup`, `PaymentMethod`, and `DeviceProtection` are likely to be relevant features in our model.

# Methodology
## Feature Engineering
Categorical data was transformed into numerical format using label encoding or one-hot encoding (for categorical data with multiple values). Data was split with a 75%/25% train/test ratio. Next, numerical data was transformed using `MinMaxScaler` to put it onto a $[0,1]$ interval. The transformation was established on the train data and used to transform the test data.

## Models
To analyze this data set, we will consider popular models for binary classification covered in this course: Perceptron, LogisticRegresion, Support Vector Machines (SVM), $k$-Nearest Neighbors (KNN), and Random Forest. 

* The Perceptron learning rule is based on the MCP neuron model and is guaranteed to converge if the data is linearly separable. This approach minimizes misclassification errors.However, the main flaw with this algorithm is that it will never converge if the classes are not perfectly separable.
* The logistic regression approach models the probability of an outcome being a certain value. It is a widely used model that is best applied for linearly separable classes and predicts a binary outcome.
* The SVM algorithm maximizes the margin between the decision boundary and the nearest training examples, this leads to lower generalization errors.
* The KNN method is a nonparametric model that memorizes the training set and makes no assumptions about the dataset, however it is a computationally expensive approach. For every item we wish to classify, it finds the k-nearest neighbors and determines the class label based on a majority of the neighbors.
* Random forest improves the accuracy compared to an individual decision tree by generating many trees and combining them. This reduces overfitting with increased computational costs.

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

Here we note that the accuracy of the model on the test set is 79.2%, which is lower than the accuracy on the training set 80.9%. The accuracies are similar, so we note that the model performs similarly on unseen data as it does with the known data. Additionally, I computed the confusion matrix, which is shown below:

![Confusion Matrix](https://github.com/cvarney/telco_customer_churn/blob/main/CM.png?raw=true)

Overall the model had 1144 true negatives and 248 true positives, for a total of 1392 correct classifications with 366 misclassifications (115 false positives and 251 false negatives). Overall, we see that the model predicts Churn 20.65% of the time, and the actual Churn rate is 28.38%. Our model is underpredicting the Churn rate on unseen data. This is likely due to overfitting in our model.

Finally, we examine the absolute value of the coefficients of the logistic regression to determine the features most important for the classification model:

![Importance](https://github.com/cvarney/telco_customer_churn/blob/main/Importance.png?raw=true)

Here we see that the most important features are: 

| Feature | Importance |
|:-------:|:----------:|
| tenure  | 1.704854 |
| Contract_Month-to-month | 0.681454 |
| Contract_Two year | 0.668443 |
| InternetService_Fiber optic | 0.450459 |
| InternetService_DSL | 0.365121 |
| PaymentMethod_Electronic check | 0.298266 |
| TotalCharges | 0.254261 |
| PaperlessBilling | 0.253542 |
| OnlineSecurity_No | 0.233404 |
| TechSupport_No | 0.233095 |
| SeniorCitizen | 0.229972 |

We note that many of these features were previously identified as being connected to high Churn rates: month-to-month contracts, fiber optic internet, paperless billing, no online security, no tech support, and senior citizen. I also find it interesting that a feature associated with very low Churn rates has high importance in the model: two year contracts. This feature is clearly an extremly important determining factor in retaining a customer and should be prioritized in all customer retention programs.

# Conclusions
In this machine learning project, I have evaluated the Telco customer Churn dataset. The data was cleaned and exploratory analysis was performed to identify important features. Categorical data was encoded into numerical data using label encoding or one hot encoding. Numerical data was transformed using the `MinMaxScaler`. Next, I employed five machine learning algorithms with default parameters and determined that LogisticRegression resulted in the highest accuracy, although SVM and RandomForest were close in performance. I tuned the hyperparameters the LogisticRegression, improving the accuracy on the training set to 80.9%. However, overfitting in the model resulted in the model underpredicting the Churn rate on test data. 

Overall, while the model could be improved through the use of dimensional reduction methods or by comparing with additional classification algorithms, it did identify important factors for customer retention that can be used to build a customer retention program for our fictional telecommunciations company.
