import os

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = "data/equipment_data.csv"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = "equipment-failure-experiment"
MODEL_NAME = "equipment-failure-model"


# -------------------------------------------------
# CONNECT TO MLFLOW
# -------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

data = pd.read_csv(DATA_FILE)

X = data[
    [
        "temperature",
        "vibration",
        "machine_age",
        "error_count",
    ]
]

y = data["failure"]


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# -------------------------------------------------
# MODEL PARAMETERS
# -------------------------------------------------

params = {
    "max_depth": 2,
    "random_state": 42,
}


# -------------------------------------------------
# START MLFLOW RUN
# -------------------------------------------------

with mlflow.start_run() as run:

    # Create model
    model = DecisionTreeClassifier(
        **params
    )

    # Train model
    model.fit(
        X_train,
        y_train,
    )

    # Run inference on test data
    predictions = model.predict(
        X_test
    )


    # -------------------------------------------------
    # EVALUATE MODEL
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )


    # -------------------------------------------------
    # LOG PARAMETERS
    # -------------------------------------------------

    mlflow.log_params(
        params
    )


    # -------------------------------------------------
    # LOG METRICS
    # -------------------------------------------------

    mlflow.log_metrics(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


    # -------------------------------------------------
    # CREATE MODEL SIGNATURE
    # -------------------------------------------------

    signature = infer_signature(
        X_train,
        model.predict(X_train),
    )


    # -------------------------------------------------
    # LOG + REGISTER MODEL
    # -------------------------------------------------

    model_info = mlflow.sklearn.log_model(
        model,
        name="equipment_failure_model",
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,

        # We trained this DecisionTree ourselves,
        # so we explicitly trust this sklearn type.
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )


    # -------------------------------------------------
    # GET REGISTERED MODEL VERSION
    # -------------------------------------------------

    model_version = (
        model_info.registered_model_version
    )


    # -------------------------------------------------
    # MARK MODEL AS CANDIDATE
    # -------------------------------------------------

    client = MlflowClient()

    client.set_model_version_tag(
        MODEL_NAME,
        str(model_version),
        "validation_status",
        "candidate",
    )


    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    print(
        f"Run ID: {run.info.run_id}"
    )

    print(
        f"Registered model: {MODEL_NAME}"
    )

    print(
        f"Model version: {model_version}"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall: {recall:.3f}"
    )

    print(
        f"F1: {f1:.3f}"
    )

    print(
        "Status: candidate"
    )