import warnings
warnings.filterwarnings("ignore")

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(__file__).with_name('creditcard.csv')
df = pd.read_csv(csv_path)

print('First 5 rows of the DataFrame:')
print(df.head())

print('\nDataFrame info:')
df.info()

from sklearn.preprocessing import StandardScaler

df.dropna(inplace=True)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

print('\nFirst 5 rows after scaling:')
print(df.head())

print('\nClass Distribution:')
print(df['Class'].value_counts())
print('\nClass distribution percentage:')
print((df['Class'].value_counts(normalize=True) * 100))

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original training set class distribution:")
print(y_train.value_counts())

train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['Class'] == 0]
minority = train_data[train_data['Class'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_upsampled = pd.concat([majority, minority_upsampled])

X_train_res = train_upsampled.drop('Class', axis=1)
y_train_res = train_upsampled['Class']

print("\nResampled training set class distribution:")
print(y_train_res.value_counts())

print("\nTest set class distribution:")
print(y_test.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

print("Logistic Regression Model Evaluation:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
)
rf_model.fit(X_train_res, y_train_res)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Model Evaluation:")
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
cm_lr = confusion_matrix(y_test, y_pred)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Genuine', 'Predicted Fraud'],
            yticklabels=['Actual Genuine', 'Actual Fraud'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted Genuine', 'Predicted Fraud'],
            yticklabels=['Actual Genuine', 'Actual Fraud'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

