# Credit Card Fraud Detection System

## Overview
This project implements a machine learning pipeline to detect fraudulent credit card transactions. It covers data preprocessing, feature engineering, exploratory data analysis (EDA), and model training using multiple classification algorithms.

The objective is to classify transactions as fraudulent or non-fraudulent with high reliability, especially considering the strong class imbalance in the dataset.

---

## Dataset
- File: `creditcard.csv` , Source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Description: Real-world anonymized credit card transaction data
- Features:
  - `V1` to `V28`: PCA-transformed numerical features
  - `Time`: Time elapsed (in seconds) between transactions
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Normal, 1 = Fraud)

Note: The dataset is highly imbalanced, with fraudulent transactions making up approximately 0.17% of the total.

---

## Workflow

### 1. Data Loading and Inspection
- Loaded using pandas
- Basic inspection using:
  - `head()`
  - `info()`
  - `describe()`

---

### 2. Train-Test Split
- Data split into training (80%) and testing (20%) sets
- Stratified sampling used to preserve class distribution

---

### 3. Feature Engineering
- Extracted time-based features:
  - Hour = (Time % 86400) // 3600
  - Day = Time // 86400
- Dropped original `Time` column

---

### 4. Outlier Handling
- Applied IQR-based capping on the `Amount` feature:
  - Lower bound = Q1 − 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR

---

### 5. Feature Scaling
- Standardization using `StandardScaler`
- Scaled features:
  - `Amount`
  - `Hour` and `Day`

---

### 6. Exploratory Data Analysis
Performed visualization to understand data patterns:
- Class distribution (training and testing sets)
- Amount distribution before and after scaling
- Fraud occurrences by hour and day
- Correlation heatmap
- Missing values heatmap
- Fraud probability over time
- Feature-wise comparison between fraud and normal transactions

---

## Models Used

### 1. Random Forest Classifier
- Handles class imbalance using `class_weight='balanced'`

### 2. Logistic Regression
- Configured with `max_iter=1000` to ensure convergence

### 3. Decision Tree Classifier
- Simple and interpretable baseline model

---

## Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Note: Due to class imbalance, precision, recall, and F1 score are more important than accuracy.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## How to Run

1. Install dependencies:
   pip install numpy pandas scikit-learn matplotlib seaborn

2. Place the dataset file:
   creditcard.csv

3. Run the script:
   python main.py

---

## Notes
- No missing values in the dataset
- Stratified splitting prevents distribution bias
- Class imbalance is handled using model class

---

## Author
Vansh Hardwani
Vaanya Kshatriya

## Reproducing the paper

1. Download `creditcard.csv` from Kaggle (mlg-ulb/creditcardfraud) into this folder.
2. Install the dependencies: `pip install -r requirements.txt`
3. Figures 1-4: `python make_figures.py creditcard.csv`
4. Tables II-IV (hold-out, 5-fold CV, ablations): `python reproduce_tables.py creditcard.csv`
5. Tables V-VI (leakage experiment, temporal validation):
   `python run_full_experiments.py creditcard.csv`, then `python summarize_results.py`.
   Raw results from our run are in `full_results.jsonl`.

The scripts import `leakage_and_temporal_experiments.py`, so keep all files in the same folder.

The leaky scenarios in `leakage_and_temporal_experiments.py` are deliberate methodological
demonstrations; their scores are invalid and are labelled as such in the paper.
Tree-based scores can differ slightly across library versions and platforms.
