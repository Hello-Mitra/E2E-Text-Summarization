import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
os.environ["DAGSHUB_USER_TOKEN"]        = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
)

client = mlflow.MlflowClient()

# Delete all versions first
versions = client.search_model_versions("name='my_model'")
for v in versions:
    print(f"Deleting version {v.version}...")
    client.delete_model_version(name="my_model", version=v.version)

# Delete the registered model itself
client.delete_registered_model(name="my_model")
print("✅ my_model deleted completely")