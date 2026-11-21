import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib


# Load dataset
data = pd.read_csv("dataset.csv")

# Input features
X = data[
    [
        "Study_Time_Hours",
        "Attendance_Percentage",
        "Previous_Exam_Score"
    ]
]

# Target
y = data["Pass_Fail"]

# Convert Pass/Fail into numbers
y = y.map({
    "Fail": 0,
    "Pass": 1
})


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)


# Save model
joblib.dump(model, "model.pkl")

print("model.pkl saved!")