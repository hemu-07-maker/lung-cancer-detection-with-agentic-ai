import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset (3000 records)
np.random.seed(42)
n = 3000

data = pd.DataFrame({
    "GENDER":               np.random.randint(0, 2, n),
    "AGE":                  np.random.randint(20, 80, n),
    "SMOKING":              np.random.randint(1, 3, n),
    "YELLOW_FINGERS":       np.random.randint(1, 3, n),
    "ANXIETY":              np.random.randint(1, 3, n),
    "PEER_PRESSURE":        np.random.randint(1, 3, n),
    "CHRONIC DISEASE":      np.random.randint(1, 3, n),
    "FATIGUE":              np.random.randint(1, 3, n),
    "ALLERGY":              np.random.randint(1, 3, n),
    "WHEEZING":             np.random.randint(1, 3, n),
    "ALCOHOL CONSUMING":    np.random.randint(1, 3, n),
    "COUGHING":             np.random.randint(1, 3, n),
    "SHORTNESS OF BREATH":  np.random.randint(1, 3, n),
    "SWALLOWING DIFFICULTY":np.random.randint(1, 3, n),
    "CHEST PAIN":           np.random.randint(1, 3, n),
})

# Generate labels based on risk factors
risk = (
    (data["SMOKING"] == 2) * 3 +
    (data["CHEST PAIN"] == 2) * 2 +
    (data["SHORTNESS OF BREATH"] == 2) * 2 +
    (data["WHEEZING"] == 2) * 1.5 +
    (data["COUGHING"] == 2) * 1.5 +
    (data["AGE"] > 55) * 2 +
    (data["CHRONIC DISEASE"] == 2) * 1 +
    (data["YELLOW_FINGERS"] == 2) * 1
)
data["LUNG_CANCER"] = (risk + np.random.normal(0, 0.5, n) > 5).astype(int)

X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

os.makedirs("model", exist_ok=True)
with open("model/lung_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/lung_cancer_model.pkl")
