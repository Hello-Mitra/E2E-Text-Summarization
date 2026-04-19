import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # ✅ Updated to your repo
        mlflow.set_tracking_uri(
            "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
        )

        cls.new_model_name    = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri     = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model         = mlflow.pyfunc.load_model(cls.new_model_uri)
        cls.vectorizer        = pickle.load(open("models/vectorizer.pkl", "rb"))

        # ✅ Updated — your pipeline saves test_tfidf.csv not test_bow.csv
        cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client   = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "this movie was absolutely brilliant"
        input_data = self.vectorizer.transform([input_text])
        input_df   = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )
        prediction = self.new_model.predict(input_df)
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.new_model.predict(X_holdout)

        accuracy  = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall    = recall_score(y_holdout, y_pred)
        f1        = f1_score(y_holdout, y_pred)

        # Thresholds — raise these once you know your model's actual performance
        self.assertGreaterEqual(accuracy,  0.40, "Accuracy below threshold")
        self.assertGreaterEqual(precision, 0.40, "Precision below threshold")
        self.assertGreaterEqual(recall,    0.40, "Recall below threshold")
        self.assertGreaterEqual(f1,        0.40, "F1 below threshold")


if __name__ == "__main__":
    unittest.main()