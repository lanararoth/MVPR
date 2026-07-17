import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("Breast_cancer_dataset.csv")

# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Convert target to numeric
# M = 1, B = 0
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("Dataset Shape:", X.shape)

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------
# Feature Selection
# Select top 10 features
# -------------------------------------------------------
selector = SelectKBest(score_func=f_classif, k=10)

X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print("\nSelected Features:")
print(selected_features)

# -------------------------------------------------------
# Apply LDA
# -------------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=1)

X_train_lda = lda.fit_transform(X_train_fs, y_train)
X_test_lda = lda.transform(X_test_fs)

# -------------------------------------------------------
# Train Bayes Classifier
# -------------------------------------------------------
model = GaussianNB()
model.fit(X_train_lda, y_train)

# Posterior probabilities
prob = model.predict_proba(X_test_lda)

# -------------------------------------------------------
# Loss Matrix
#
#            Predicted
#            B    M
# True B     0    1
# True M     5    0
#
# Missing a malignant tumor is more costly.
# -------------------------------------------------------
loss_matrix = np.array([
    [0, 1],
    [5, 0]
])

# -------------------------------------------------------
# Risk-Based Prediction
# -------------------------------------------------------
predictions = []

for p in prob:
    risk = loss_matrix @ p
    predictions.append(np.argmin(risk))

predictions = np.array(predictions)

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")
print(accuracy)

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("\nClassification Report")
print(classification_report(y_test, predictions))

# -------------------------------------------------------
# Expected Loss
# -------------------------------------------------------
loss = 0

for actual, pred in zip(y_test, predictions):
    loss += loss_matrix[actual, pred]

expected_loss = loss / len(y_test)

print("Expected Loss =", expected_loss)

# -------------------------------------------------------
# LDA Visualization
# -------------------------------------------------------
plt.figure(figsize=(8,5))

plt.scatter(
    X_test_lda[y_test==0],
    np.zeros(sum(y_test==0)),
    label="Benign"
)

plt.scatter(
    X_test_lda[y_test==1],
    np.zeros(sum(y_test==1)),
    label="Malignant"
)

plt.xlabel("LDA Component")
plt.title("LDA Projection")
plt.yticks([])
plt.legend()
plt.grid(True)
plt.show()