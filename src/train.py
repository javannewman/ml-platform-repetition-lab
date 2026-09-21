import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


DATA_FILE = "data/equipment_data.csv"
MODEL_FILE = "models/equipment_failure_model.joblib"


# 1. Load the validated data
data = pd.read_csv(DATA_FILE)


# 2. Separate features from target
X = data[
    [
        "temperature",
        "vibration",
        "machine_age",
        "error_count",
    ]
]

y = data["failure"]


# 3. Split historical data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 4. Create the model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
)


# 5. Train
model.fit(X_train, y_train)


# 6. Evaluate on data it did not train on
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print(f"Accuracy: {accuracy:.3f}")


# 7. Save the trained artifact
joblib.dump(model, MODEL_FILE)

print(f"Model saved to {MODEL_FILE}")