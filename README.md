# Customer Churn Prediction using Logistic Regression

This repository contains the submission for **AI-ML Assignment - 2** on predicting customer churn using a Logistic Regression model.

---

## 📌 Objective
The goal of this assignment is to develop a predictive machine learning model that determines whether a telecommunications customer is likely to leave (churn) the company. The prediction is based on demographic information, subscribed services, and account/contract details.

---

## 📊 Dataset Link
The analysis is performed using the **Telco Customer Churn Dataset**.
* **Kaggle Dataset URL**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Dataset File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns)

*Note: In accordance with submission guidelines, the raw dataset file is not uploaded to the repository.*

---

## 🛠️ Libraries Used
The following Python libraries were used for this project:
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn` (specifically `LogisticRegression`, `StandardScaler`, `train_test_split`, and metrics utilities)

---

## ⚙️ Methodology
1. **Data Understanding**:
   * Analyzed column types and structures. Identified `tenure`, `MonthlyCharges`, and `TotalCharges` as numerical features, `Churn` as the target variable, and the remaining 17 columns as categorical features.
2. **Data Preprocessing**:
   * **Missing Values**: Handled 11 blank spaces in the `TotalCharges` column. These corresponded to customers with 0 months of tenure (new sign-ups). They were imputed with `0.0` after converting the column to numeric.
   * **Feature Selection**: Dropped `customerID` as it is a unique identifier and has no predictive power.
   * **Target Encoding**: Mapped the target variable `Churn` to binary labels (`Yes` -> `1`, `No` -> `0`).
   * **Categorical Encoding**: One-Hot encoded multi-level categorical variables using `drop_first=True` to avoid the dummy variable trap.
   * **Train-Test Split**: Partitioned the dataset into **80% training** and **20% testing** sets. Stratified sampling was applied to maintain the ratio of churned vs. non-churned customers in both splits.
   * **Feature Scaling**: Standardized numerical features using `StandardScaler` to ensure scale invariance and model convergence.
3. **Model Development**:
   * Trained a `LogisticRegression` model with a maximum of 1,000 iterations for convergence and set `random_state=42` for reproducibility.
4. **Evaluation**:
   * Calculated classification metrics (Accuracy, Precision, Recall, F1-Score) and plotted a confusion matrix.
   * Extracted coefficients to identify key factors influencing churn.

---

## 📈 Results

### Performance Metrics
The model was evaluated on the unseen 20% test dataset (1,409 records):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 80.62% |
| **Precision** | 65.93% |
| **Recall** | 55.88% |
| **F1-Score** | 60.49% |

### Confusion Matrix
```
[[927  108]   <-- Actual: No Churn [TN, FP]
 [165  209]]  <-- Actual: Churn    [FN, TP]
```

### Key Factors Influencing Churn (Top Coefficients)
* **Positive Drivers of Churn (Increase probability of leaving)**:
  1. `InternetService_Fiber optic` (+1.18)
  2. `TotalCharges` (+0.53)
  3. `PaymentMethod_Electronic check` (+0.39)
* **Negative Drivers of Churn (Increase retention/loyalty)**:
  1. `Contract_Two year` (-1.31)
  2. `tenure` (-1.26)
  3. `Contract_One year` (-0.68)

---

## 📝 Conclusion
This study built a Logistic Regression model to predict customer churn with 80.6% accuracy. Key findings indicate that longer customer tenure and two-year contracts strongly decrease churn probability, acting as key retention factors. Conversely, subscribing to Fiber Optic internet and paying via Electronic Check significantly increase churn likelihood, highlighting potential service dissatisfaction or billing friction. 

A major limitation of Logistic Regression in this scenario is its assumption of linearity between independent features and log-odds of the target. It cannot capture complex, non-linear relationships or multi-feature interactions (e.g., tenure combined with specific contract and support tickets) without manual, complex feature engineering, unlike non-linear algorithms such as Random Forests or Gradient Boosting Machines.
