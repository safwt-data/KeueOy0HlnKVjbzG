<h1 align="center">🏦 Customer Investment Subscription Prediction</h1>

<p align="center">
A machine learning classification project for predicting customer subscription to a term-deposit investment product,
identifying high-potential customer segments, and explaining the key factors influencing subscription predictions.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Classification-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Best%20Model-Tuned%20XGBoost-green" alt="Best Model">
  <img src="https://img.shields.io/badge/EDA-93%25-brightgreen" alt="EDA">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP%20%7C%20LIME-blueviolet" alt="Explainable AI">
  <img src="https://img.shields.io/badge/Class%20Imbalance-Handled-yellow" alt="Class Imbalance">
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Questions](#-business-questions)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [1. Data Preparation](#1-data-preparation)
- [2. Customer Segmentation](#2-customer-segmentation)
- [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
- [4. Missing Data & Outliers](#4-missing-data--outliers)
- [5. Preprocessing](#5-preprocessing)
- [6. Class Imbalance](#6-class-imbalance)
- [7. Machine Learning Models](#7-machine-learning-models)
- [8. Model Optimization](#8-model-optimization)
- [9. Model Explainability](#9-model-explainability)
- [Key Findings](#-key-findings)
- [Business Interpretation](#-business-interpretation)
- [Project Contribution](#-project-contribution)
- [Conclusion](#-conclusion)

---

## 🔎 Project Overview

The objective of this project is to predict whether a bank customer will subscribe to a term-deposit product.

The project goes beyond producing a classification score. It combines:

- **Customer segmentation** to identify groups with above-average subscription rates.
- **Exploratory data analysis** to understand distributions, missingness, outliers, and relationships.
- **Classification modeling** to compare multiple machine learning approaches.
- **Class-imbalance handling** to improve detection of the minority subscription class.
- **Hyperparameter optimization** using randomized search and cross-validation.
- **Explainable AI** using feature importance, permutation importance, LIME, and SHAP.

The final goal is to support a practical business decision:

> **Which customers are most promising to target, and what variables are driving the model's predictions?**

---

## 🎯 Business Questions

The analysis focuses on three main questions:

1. **Can customer subscription to a term deposit be predicted?**
2. **Which variables contribute most strongly to the prediction?**
3. **Which customer segments show the highest historical subscription rates and should receive greater marketing attention?**

---
## 📊 Dataset

The dataset contains **40,000 customer records** and **14 variables**, including:

- Age
- Job
- Marital status
- Education
- Credit default status
- Account balance
- Housing loan
- Personal loan
- Contact method
- Contact day and month
- Last-contact duration
- Number of contacts during the campaign
- Subscription outcome (`y`)

The target variable is binary:

- `0` = Not subscribed
- `1` = Subscribed

The target is strongly imbalanced:

- **92.76%** did not subscribe
- **7.24%** subscribed

This imbalance makes metrics such as **precision, recall, and F1-score** especially important in addition to overall accuracy.

---

## 🔄 Project Workflow

```text
Raw Customer Data
        │
        ▼
Data Inspection
        │
        ▼
Encoding & Feature Preparation
        │
        ├──────────────► Customer Segmentation
        │
        ▼
Exploratory Data Analysis
        │
        ├── Missing-Value Analysis
        ├── Distribution Analysis
        ├── Outlier Analysis
        └── Correlation Analysis
        │
        ▼
Train/Test Split
        │
        ▼
Imputation & Scaling
        │
        ▼
Class-Imbalance Handling
        │
        ▼
Model Training & Comparison
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Evaluation on Unseen Data
        │
        ▼
Feature Importance + Permutation Importance
        │
        ▼
LIME + SHAP Explainability
        │
        ▼
Business Interpretation
```

---

## 1. Data Preparation

Categorical variables were transformed into machine-readable representations.

Binary variables such as:

- credit default,
- housing loan,
- personal loan,
- subscription outcome

were mapped to `0` and `1`.

Nominal variables such as job, marital status, and contact method were encoded numerically, while ordered variables such as education and month were handled using ordinal mappings.

Values recorded as `"unknown"` in selected fields were treated as missing values so they could be assessed and processed explicitly rather than treated as meaningful categories.

---

## 2. Customer Segmentation

Before modeling, customer groups were examined to identify segments with stronger-than-average subscription behavior.

The overall subscription rate is **7.24%**, which provides a useful baseline for evaluating each segment.

Segments were constructed using combinations of customer and campaign characteristics, and each segment was evaluated using:

- **Subscription rate**
- **Number of customers**

Small segments were filtered out to avoid making business recommendations from very limited samples.

### Highest-performing segments identified

| Segment | Subscription Rate | Customers |
|---|---:|---:|
| Management, tertiary education, no credit default, housing loan, no personal loan, cellular contact in May | **12.08%** | 480 |
| Services, secondary education, no credit default, housing loan, no personal loan, cellular contact in May | **11.33%** | 415 |
| Admin, secondary education, no credit default, housing loan, no personal loan, cellular contact in May | **11.16%** | 421 |
| Blue-collar, secondary education, no credit default, housing loan, no personal loan, cellular contact in May | **7.27%** | 729 |

The first three segments substantially outperform the **7.24% overall subscription baseline**.

This part of the project translates model-oriented analysis into a more directly actionable marketing question: **who should receive greater attention?**

---

## 3. Exploratory Data Analysis

EDA was used to understand the structure and quality of the data before training the models.

The analysis included:

- Data types and variable inspection
- Target-class distribution
- Missing-value analysis
- Numerical distributions
- Outlier detection
- Correlation analysis

The numerical distributions revealed substantial right skew in variables such as **balance** and **duration**, which influenced the choice of outlier-analysis technique.

The correlation analysis did not reveal widespread strong linear relationships between predictors. The strongest relationship with the target was observed for **last-contact duration**.

---

## 4. Missing Data & Outliers

### Missing values

Missingness was investigated rather than automatically removing incomplete records.

The largest missing-data issue appeared in the **contact** variable, with approximately **32% missing values**. Because such a large proportion is missing, simple mode imputation may influence the original distribution and should be interpreted carefully.

### Outliers

Because several numerical variables are skewed, a traditional Z-score approach was not considered ideal.

Instead, an **IQR/quantile-based approach** was used to identify extreme observations because it is more robust to skewed distributions.

The analysis identified extreme observations particularly in:

- `balance`
- `duration`

while very few were identified for `age`, and none for `day` under the selected threshold.

---

## 5. Preprocessing

The dataset was split into training and test sets using a **stratified 80/20 split** to preserve the original target-class proportions.

Preprocessing included:

- Mode imputation for selected categorical variables
- Standardization of selected numerical variables
- Encoding of categorical features
- Separate training and test datasets to support evaluation on unseen observations

The numerical variables selected for standardization included:

- `balance`
- `age`
- `duration`

---

## 6. Class Imbalance

Only **7.24%** of customers subscribed, making the positive class considerably smaller than the non-subscriber class.

Resampling techniques were explored to reduce the impact of this imbalance during model training.

This was important because a model could achieve high accuracy by predicting the majority class while still performing poorly at identifying customers who actually subscribe.

For that reason, model performance was evaluated using:

- Precision
- Recall
- F1-score
- Accuracy

with particular attention paid to **class 1 — subscribed customers**.

---

## 7. Machine Learning Models

Several classification algorithms were trained and compared:

- **K-Nearest Neighbors**
- **Logistic Regression**
- **Support Vector Machine**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

### Model comparison on the test set

| Model | Accuracy | Precision — Subscribed | Recall — Subscribed | F1 — Subscribed |
|---|---:|---:|---:|---:|
| KNN | 83% | 0.27 | 0.78 | 0.41 |
| Logistic Regression | 84% | 0.29 | 0.79 | 0.42 |
| SVM | 84% | 0.30 | **0.85** | 0.44 |
| Decision Tree | 92% | 0.41 | 0.34 | 0.37 |
| Random Forest | **93%** | **0.56** | 0.42 | 0.48 |
| XGBoost | 91% | 0.43 | 0.75 | **0.55** |
| Tuned XGBoost | **93%** | 0.49 | 0.61 | 0.54 |

The results show why accuracy alone is not enough for this task.

For example, Random Forest produced high overall accuracy and precision, while XGBoost captured a much larger share of actual subscribers. This illustrates the trade-off between minimizing false positives and finding more potential buyers.

---

## 8. Model Optimization

XGBoost was explored further using **`RandomizedSearchCV`** with cross-validation.

Parameters investigated included:

- `max_depth`
- `learning_rate`
- `scale_pos_weight`

Different optimization objectives were explored, including accuracy and F1-score.

The tuned XGBoost model achieved:

- **93% accuracy**
- **49% precision** for subscribers
- **61% recall** for subscribers
- **54% F1-score** for subscribers

Rather than relying only on the headline accuracy figure, the analysis considers the business trade-off between **precision and recall** when identifying prospective subscribers.

---

## 9. Model Explainability

Prediction alone is not sufficient for a business-facing classification project. The project therefore includes several complementary explainability methods.

### Feature Importance

Tree-based feature importance was used to examine which variables contributed most strongly to the fitted XGBoost model.

### Permutation Importance

Permutation importance was calculated on the **unseen test set** by repeatedly shuffling each feature and measuring the resulting deterioration in model performance.

The strongest signals were:

| Feature | Mean decrease in model score |
|---|---:|
| Duration | **0.0543** |
| Month | **0.0424** |
| Day | **0.0381** |
| Housing | 0.0032 |
| Education | 0.0016 |

`duration` produced the largest decline in performance when shuffled, indicating that the model depends strongly on this variable.

### LIME

**LIME** was used to explain individual predictions, helping show why the model classified a specific customer as more or less likely to subscribe.

### SHAP

**SHAP** was applied to both training and unseen test data to understand:

- Global feature influence
- Direction of feature effects
- The contribution of individual feature values to model output

Using SHAP alongside permutation importance provides two complementary perspectives:

- **Permutation importance:** How much does model performance depend on the feature?
- **SHAP:** How does the feature influence predictions and in which direction?

---

## 🔑 Key Findings

### 1. Subscription is a minority outcome

Only **7.24%** of customers subscribed, making this an imbalanced classification problem rather than a simple accuracy-maximization task.

### 2. Some customer segments considerably outperform the baseline

Several customer groups achieved subscription rates above **11%**, compared with the overall **7.24%** rate.

### 3. Duration is the strongest predictive signal

Permutation importance on unseen data showed that shuffling `duration` reduced model performance by approximately **5.4 percentage points**, the largest reduction among the tested variables.

### 4. Month and day also contain substantial predictive information

After duration, `month` and `day` produced the next-largest decreases in model performance when permuted.

### 5. Different models create different business trade-offs

The model with the highest accuracy is not automatically the model with the strongest ability to identify subscribers.

XGBoost provided a stronger balance for detecting the minority class, while Random Forest produced higher subscriber precision but lower recall.

---

## 💼 Business Interpretation

The project separates two related but different decisions.

### Who should be targeted?

Customer segmentation can identify customer profiles whose historical subscription rates are substantially above the overall baseline.

### How likely is an individual customer to subscribe?

The classification models provide a predictive framework for estimating subscription behavior at customer level.

### Why did the model make that prediction?

Permutation importance, LIME, and SHAP help make model behavior more transparent by identifying the variables and feature values that influence predictions.

One important business consideration is that **contact duration is only known after or during a marketing interaction**. Although it is highly predictive, it should therefore be treated carefully when the objective is to decide **whom to contact before the campaign begins**. Pre-contact customer attributes are more appropriate for prospective targeting decisions.

---

## 👨‍💻 Project Contribution

This project demonstrates an end-to-end approach to a realistic customer classification problem.

My contribution covered:

- Translating the business problem into a binary classification task
- Preparing and encoding customer and campaign data
- Investigating missing data and skewed numerical distributions
- Designing customer segments and comparing them with the overall subscription baseline
- Handling severe class imbalance
- Training and comparing six classification algorithms
- Optimizing XGBoost using randomized hyperparameter search and cross-validation
- Evaluating performance using class-specific precision, recall, F1-score, and accuracy
- Testing feature reliance using permutation importance on unseen data
- Applying LIME and SHAP for local and global model explainability
- Translating technical findings into customer-targeting and marketing insights

The project is intended to demonstrate not only **model building**, but also the ability to connect **data preparation, predictive modeling, explainability, and business decision-making**.

## ✅ Conclusion

This project develops a complete machine learning workflow for predicting term-deposit subscription in a highly imbalanced customer dataset.

The analysis showed that model quality cannot be judged from accuracy alone. Different algorithms produced different precision-recall trade-offs for the relatively rare subscriber class, with XGBoost providing a strong balance between overall performance and the ability to identify potential subscribers.

The project also moves beyond prediction by identifying customer segments with above-average historical subscription rates and by applying multiple explainability techniques to understand the model's behavior.

For hiring managers and recruiters, this project demonstrates practical experience across the full analytical workflow: **business-question framing, data preparation, EDA, segmentation, classification, imbalance handling, model evaluation, hyperparameter tuning, explainable AI, and business interpretation**.

The focus is not simply on building a model, but on turning its results into information that can support a real customer-targeting decision.
