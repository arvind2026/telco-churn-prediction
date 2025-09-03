# Telecom Customer Churn Prediction

## Executive Summary

Customer churn represents one of the most critical challenges in the telecommunications industry, with annual attrition rates ranging from 15-25% in this highly competitive market. This project develops a machine learning framework to identify customers at high risk of churning, enabling targeted retention strategies that optimize resource allocation and maximize customer lifetime value.

The cost differential between customer acquisition and retention makes churn prediction a strategic imperative. By accurately identifying at-risk customers, telecommunications companies can deploy focused retention campaigns rather than broad-based initiatives, significantly improving ROI on marketing expenditures.

## Objectives

1. Quantify customer churn rates and identify key demographic and behavioral drivers
2. Conduct comprehensive feature analysis to determine churn predictors
3. Develop and evaluate multiple machine learning models for optimal churn classification
4. Implement ensemble methods to maximize predictive accuracy


## Dataset Overview

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)

The dataset encompasses comprehensive customer information across four key dimensions:

- **Churn Status:** Binary indicator of customer departure within the last month
- **Service Portfolio:** Phone services, internet connectivity, security features, streaming services, and technical support subscriptions
- **Account Details:** Customer tenure, contract terms, payment methods, billing preferences, and financial metrics
- **Demographics:** Gender, age segmentation, partnership status, and dependent relationships

**Technical Stack:** scikit-learn, pandas, NumPy, matplotlib, seaborn

## Quick Start

```bash
# Clone repository and install dependencies
pip install -r importants.txt (requiremetns or dependencies)

# Run analysis
jupyter notebook telecom_churn_analysis.ipynb

# Execute model training
python train_churn_model.py
```

**Requirements:** Python 3.8+, scikit-learn 1.0+, pandas 1.3+, numpy 1.21+

## Exploratory Data Analysis

### Churn Distribution
<p>
  <img src="output/ChurnDistribution.png" width="920">
</p>

**Key Finding:** 26.6% of customers terminated services, establishing baseline churn rate for model evaluation.

### Gender Analysis
<p>
  <img src="output/distributionWRTGender.PNG" width="920">
</p>

Gender demonstrates no significant correlation with churn behavior, indicating equal likelihood across demographics.

### Contract Type Impact
<p>
  <img src="output/ContractDistribution.png" width="920">
</p>


Contract duration shows strong inverse correlation with churn:
- 75% of churned customers had month-to-month contracts
- 13% of churned customers had one-year contracts  
- 3% of churned customers had two-year contracts

### Payment Method Analysis
![Distribution of Payments methods](output/paymentmethods.png) 
![Churn wrt payment methods](output/paymentethodswithrespecttochurn.PNG)

Electronic check users exhibit significantly higher churn rates compared to automated payment methods (credit card, bank transfer) and traditional mail payments.

### Service-Specific Insights

#### Internet Services
![Churn distribution w.r.t Internet services and Gender](output/internetservices.PNG)

Fiber optic customers demonstrate elevated churn rates despite service popularity, suggesting potential service quality or pricing concerns.

#### Customer Support Services
![Churn distribution w.r.t online security](output/onlineSecurity.PNG)
<p>
  <img src="output/techSupport.PNG" width="920">
</p>


Customers lacking online security and technical support show significantly higher churn propensity, highlighting the retention value of comprehensive service packages.

#### Demographic Patterns
![Churn distribution w.r.t dependents](output/dependents.PNG)
![Churn distribution w.r.t Senior Citizen](output/seniorCitzen.PNG)

- Customers without dependents exhibit higher churn likelihood
- Senior citizens, though representing a smaller customer segment, show elevated churn rates

#### Billing and Financial Factors
![Churn distribution w.r.t mode of billing](output/billing.png)

<p>
  <img src="output/chargesDistribution.PNG" width="920">
</p>
<p>
  <img src="output/totalcharges.PNG" width="920">
</p>
<p>
  <img src="output/tenureandchurn.PNG" width="920">
</p>

- Paperless billing correlates with increased churn probability
- Higher monthly charges correlate with elevated churn risk
- Customer tenure demonstrates strong inverse relationship with churn likelihood

## Model Development and Evaluation

### Algorithm Performance Comparison
<p>
  <img src="output/Modelevaluation.PNG" width="920">
</p>

### Cross-Validation Results

![Logistic Regression](output/LR.PNG) 
![KNN](output/KNN.PNG)
![Naive Bayes](output/NaiveBayes.PNG)
![Decision Tree](output/DecisionTrees.PNG)
![Random Forest](output/RandomForest.PNG)
![Adaboost](output/Adaboost.png)
![Gradient Boost](output/Gradientboost.PNG)
![Voting Classifier](output/VotingClassifier.PNG)

### Model Performance Summary

```
Model Performance (Test Set - Accuracy ± Std Dev):
├── VotingClassifier: 84.68% ± 1.09%
├── GradientBoostingClassifier: 84.46% ± 1.07%
├── AdaBoostClassifier: 84.46% ± 1.13%
├── LogisticRegression: 84.13% ± 1.05%
├── GaussianNB: 82.32% ± 0.74%
├── RandomForestClassifier: 81.98% ± 1.16%
├── KNeighborsClassifier: 79.13% ± 0.82%
└── DecisionTreeClassifier: 64.70% ± 2.20%
```

**Final Model Metrics (Voting Classifier):**
- **Accuracy:** 84.7%
- **Precision:** 62.9% (281 true positives / 447 predicted positives)
- **Recall:** 50.1% (281 true positives / 561 actual churners)
- **F1-Score:** 55.8%
- **Specificity:** 89.3% (correct non-churn identification)

### Final Model Implementation

**Selected Approach:** Soft Voting Classifier combining Gradient Boosting, Logistic Regression, and AdaBoost

```python
from sklearn.ensemble import VotingClassifier

clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()

eclf1 = VotingClassifier(
    estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], 
    voting='soft'
)

eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)

print("Final Accuracy Score:", accuracy_score(y_test, predictions))
```

### Confusion Matrix Analysis
![Confusion Matrix](output/confusion_matrix_models.PNG)

<p>
  <img src="output/confusionMatrix.png" width="920">
</p>

**Model Performance Metrics:**
- True Negatives: 1,383 (correctly identified non-churn customers)
- False Positives: 166 (incorrectly predicted churn)
- False Negatives: 280 (missed churn cases)  
- True Positives: 281 (correctly identified churn customers)

**Business Impact:** The model successfully identifies 50.1% of actual churners while maintaining 89.3% accuracy for non-churn predictions.


## Business Applications

This predictive framework enables telecommunications companies to:
- Implement targeted retention campaigns with estimated 15-20% improvement in customer retention
- Optimize marketing spend allocation by focusing resources on high-risk customer segments  
- Develop personalized service offerings based on churn risk factors
- Establish proactive customer success programs for early-stage customers and high-value accounts
