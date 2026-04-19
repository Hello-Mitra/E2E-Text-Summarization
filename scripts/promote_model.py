import os
import mlflow

def promote_model():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # ✅ Updated to your repo
    mlflow.set_tracking_uri(
        "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
    )

    client     = mlflow.MlflowClient()
    model_name = "my_model"

    # Get latest version in Staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model in Staging — nothing to promote")
        return

    latest_staging = staging_versions[0].version

    # Archive current Production model
    for version in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote Staging → Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging,
        stage="Production"
    )
    print(f"Model version {latest_staging} promoted to Production ✅")


if __name__ == "__main__":
    promote_model()