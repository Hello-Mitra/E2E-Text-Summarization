import os
import mlflow

def promote_model():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
    )

    client     = mlflow.MlflowClient()
    model_name = "my_model"

    # Get latest version in Staging (still works even if deprecated)
    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        print("No model in Staging — nothing to promote")
        return

    new_version = staging[0].version

    # ✅ Set alias "production" on the new version
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=new_version
    )

    print(f"Model version {new_version} aliased as 'production' ✅")


if __name__ == "__main__":
    promote_model()