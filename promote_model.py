from mlflow import MlflowClient

client = MlflowClient()

#version 3'ni Productionga o'tkazamiz  (chunki eng yaxshi natija bergan)
client.transition_model_version_stage(
    name="FraudDetector",
    version=3,
    stage="Production"
)

print("Version 3 endi Production holatida!")