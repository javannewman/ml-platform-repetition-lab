import os
import sys

import mlflow

from mlflow import MlflowClient


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "equipment-failure-model"


mlflow.set_tracking_uri(
    TRACKING_URI
)


if len(sys.argv) != 2:

    print(
        "Usage: python src/promote_model.py <version>"
    )

    sys.exit(1)


version = sys.argv[1]

client = MlflowClient()


# Confirm that the requested model version exists
model_version = client.get_model_version(
    MODEL_NAME,
    version,
)


# Mark the model as approved
client.set_model_version_tag(
    MODEL_NAME,
    version,
    "validation_status",
    "approved",
)


# Assign the production alias
client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    version,
)


print(
    f"Model {MODEL_NAME} version {version} "
    "is now approved and assigned alias 'champion'"
)