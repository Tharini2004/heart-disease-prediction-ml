"""
Heart Disease Prediction Using Machine Learning
Authors: M Sundeep, Preethi M, Samarth V H, Tharini G
Institution: Nagarjuna College of Engineering and Technology
Department: CSE (AI & ML) | Academic Year: 2024-25
Best Result: 93.79% accuracy - Random Forest + SMOTE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN

print("=" * 60)
print("  HEART DISEASE PREDICTION — ML COMPARATIVE ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
try:
    df = pd.read_csv('heart.csv')
    print(f"\n✅ Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("⚠️  heart.csv not found.")
    print("   Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    exit()

print(f"\nClass Distribution:\n{df['target'].value_counts().to_string()}")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 1: PREPROCESSING")
print("─" * 60)

df.fillna(df.median(numeric_only=True), inplace=True)
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✅ Features normalized — shape: {X_scaled.shape}")

# ─────────────────────────────────────────
# 3. APPLY SMOTE
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 2: SMOTE OVERSAMPLING")
print("─" * 60)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_scaled, y)
print(f"✅ Before SMOTE: {X_scaled.shape[0]} samples")
print(f"✅ After SMOTE:  {X_smote.shape[0]} samples")

# ADASYN
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_scaled, y)
print(f"✅ After ADASYN: {X_adasyn.shape[0]} samples")

# ─────────────────────────────────────────
# 4. DEFINE CLASSIFIERS
# ─────────────────────────────────────────
classifiers = {
    'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':      DecisionTreeClassifier(random_state=42),
    'Naive Bayes':        GaussianNB(),
    'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
}

# ─────────────────────────────────────────
# 5. CROSS-VALIDATION ON SMOTE DATA
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 3: RESULTS WITH SMOTE (Cross-Validation)")
print("─" * 60)

smote_results = {}
for name, clf in classifiers.items():
    acc  = cross_val_score(clf, X_smote, y_smote, cv=10, scoring='accuracy').mean()
    prec = cross_val_score(clf, X_smote, y_smote, cv=10, scoring='precision').mean()
    rec  = cross_val_score(clf, X_smote, y_smote, cv=10, scoring='recall').mean()
    f1   = cross_val_score(clf, X_smote, y_smote, cv=10, scoring='f1').mean()
    smote_results[name] = {
        'Accuracy': round(acc*100, 2),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'F1 Score': round(f1, 3)
    }
    print(f"  {name:<22} Acc: {acc*100:.2f}%  F1: {f1:.3f}")

smote_df = pd.DataFrame(smote_results).T.sort_values('Accuracy', ascending=False)

# ─────────────────────────────────────────
# 6. CROSS-VALIDATION ON ADASYN DATA
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4: RESULTS WITH ADASYN (Cross-Validation)")
print("─" * 60)

adasyn_results = {}
for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_adasyn, y_adasyn, cv=10, scoring='accuracy').mean()
    f1  = cross_val_score(clf, X_adasyn, y_adasyn, cv=10, scoring='f1').mean()
    adasyn_results[name] = {'Accuracy': round(acc*100, 2), 'F1 Score': round(f1, 3)}
    print(f"  {name:<22} Acc: {acc*100:.2f}%  F1: {f1:.3f}")

# ─────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 5: GENERATING VISUALIZATIONS")
print("─" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Heart Disease Prediction — ML Algorithm Comparison\nTharini G et al. | Nagarjuna College of Engineering & Technology',
             fontsize=12, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors  = ['#C00000', '#70AD47', '#ED7D31', '#FFC000']

for metric, ax, color in zip(metrics, axes.flatten(), colors):
    vals = smote_df[metric]
    bars = ax.barh(vals.index, vals.values, color=color, edgecolor='white', height=0.6)
    ax.set_xlabel(metric)
    ax.set_title(f'{metric} — SMOTE (10-Fold CV)')
    ax.set_xlim(0, max(vals.values) * 1.18)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val}{"%" if metric=="Accuracy" else ""}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: algorithm_comparison.png")

# Confusion Matrix — Random Forest (best model)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.34, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No CHD', 'CHD'],
            yticklabels=['No CHD', 'CHD'])
plt.title('Random Forest Confusion Matrix\n(Best Model — SMOTE)', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
print("✅ Saved: confusion_matrix_rf.png")

# ─────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS — SMOTE (10-Fold Cross-Validation)")
print("=" * 60)
print(smote_df.to_string())
print(f"\n✅ Best Algorithm: {smote_df['Accuracy'].idxmax()} ({smote_df['Accuracy'].max()}%)")
print("=" * 60)
