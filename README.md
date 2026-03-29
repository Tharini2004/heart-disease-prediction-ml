# Heart Disease Prediction Using Machine Learning

Comparative analysis of 5 ML algorithms with SMOTE and ADASYN oversampling 
for early Coronary Heart Disease (CHD) prediction — achieving 93.79% accuracy 
with Random Forest + SMOTE.

---

## Overview

Coronary Heart Disease (CHD) is a leading cause of death worldwide. This project 
applies supervised machine learning on two datasets — a large Canadian dataset 
(70,000 records) and a smaller Bangladeshi dataset (1,008 records) — to build 
and compare predictive models for early CHD detection.

SMOTE and ADASYN oversampling techniques handle class imbalance in the 
Bangladeshi dataset.

---

## Key Results

### Canadian Dataset (70,000 records — Balanced)

| Algorithm | Accuracy |
|---|---|
| Logistic Regression | 72.33% |
| Random Forest | 71.5% |
| KNN | 70.1% |
| Decision Tree | 68.4% |
| Naive Bayes | 67.2% |

### Bangladeshi Dataset (1,008 records — Imbalanced)

| Algorithm | Technique | Accuracy | F1 Score |
|---|---|---|---|
| Random Forest | SMOTE | 93.79% | 0.83 |
| Random Forest | ADASYN | 88.12% | 0.93 |

Best model: Random Forest + SMOTE — 93.79% accuracy

---

## Tech Stack

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | SMOTE | ADASYN

---

## Project Structure

heart-disease-prediction-ml/
├── heart_disease_prediction.py
├── heart_disease_prediction.ipynb
├── requirements.txt
├── algorithm_comparison.png
├── confusion_matrix_rf.png
└── README.md

---

## How to Run

pip install -r requirements.txt
python heart_disease_prediction.py

Download dataset: Kaggle Heart Disease Dataset → save as heart.csv

---

## Datasets

- Canadian Dataset: 70,000 balanced records
- Bangladeshi Dataset: 1,008 records (imbalanced)

Features: Age, Gender, Cholesterol, Blood Pressure, Smoking, Blood Sugar, ECG, Heart Rate
