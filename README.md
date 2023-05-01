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

# Results & Discussion

# Conclusion
