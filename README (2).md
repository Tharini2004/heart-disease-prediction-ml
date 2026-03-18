# ❤️ Heart Disease Prediction Using Machine Learning

> Comparative analysis of 5 ML algorithms with SMOTE & ADASYN oversampling for early Coronary Heart Disease (CHD) prediction — achieving **93.79% accuracy** with Random Forest + SMOTE.

**Authors:** M Sundeep (1NC22CI032), Preethi M (1NC22CI045), Samarth V H (1NC22CI052), Tharini G (1NC22CI061)
**Institution:** Nagarjuna College of Engineering and Technology, Bengaluru
**Department:** CSE (AI & ML) | **Academic Year:** 2024–25
**Guide:** Manjunath K N, Assistant Professor

---

## 📌 Overview

Coronary Heart Disease (CHD) is a leading cause of death worldwide. This project applies supervised machine learning on two datasets — a large Canadian dataset (70,000 records) and a smaller Bangladeshi dataset (1,008 records) — to build and compare predictive models for early CHD detection.

SMOTE and ADASYN oversampling techniques handle class imbalance in the Bangladeshi dataset.

---

## 🎯 Key Results

### Canadian Dataset (70,000 records — Balanced)
| Algorithm | Accuracy |
|---|---|
| **Logistic Regression** | **72.33%** |
| Random Forest | 71.5% |
| KNN | 70.1% |
| Decision Tree | 68.4% |
| Naive Bayes | 67.2% |

### Bangladeshi Dataset (1,008 records — Imbalanced)
| Algorithm | Technique | Accuracy | F1 Score |
|---|---|---|---|
| **Random Forest** | **SMOTE** | **93.79%** | **0.83** |
| Random Forest | ADASYN | 88.12% | 0.93 |

> ✅ **Best model: Random Forest + SMOTE — 93.79% accuracy**

---

## 📊 Datasets

- **Canadian Dataset:** 70,000 balanced records — [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Bangladeshi Dataset:** 1,008 records (imbalanced) — collected with medical assistance

**Features:** Age, Gender, Cholesterol, Blood Pressure, Smoking Status, Blood Sugar, ECG, Heart Rate

---

## 🤖 Algorithms

| Algorithm | Description |
|---|---|
| Random Forest | Ensemble of decision trees — best performer |
| KNN | Classifies based on nearest neighbors |
| Decision Tree | Tree-based decision model |
| Naive Bayes | Probabilistic classifier |
| Logistic Regression | Binary classification using sigmoid function |

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **SMOTE & ADASYN** — imbalanced-learn
- **Validation** — Cross-validation

---

## 🗂️ Project Structure

```
heart-disease-prediction-ml/
├── heart_disease_prediction.py
├── heart_disease_prediction.ipynb
├── requirements.txt
├── algorithm_comparison.png
├── confusion_matrix_rf.png
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python heart_disease_prediction.py
```

Download dataset: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) → save as `heart.csv`

---

## 📄 License

Academic project — Nagarjuna College of Engineering and Technology, 2024–25.
