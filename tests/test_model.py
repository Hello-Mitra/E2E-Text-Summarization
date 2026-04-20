import unittest
import mlflow
import os
import tempfile
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
        os.environ["DAGSHUB_USER_TOKEN"]        = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
        )

        client = mlflow.MlflowClient()

        # ── Load challenger + its own vectorizer ──────────────────────────
        try:
            challenger            = client.get_model_version_by_alias("my_model", "challenger")
            cls.new_model_version = challenger.version
        except Exception:
            raise RuntimeError("No model with alias 'challenger' found — cannot run tests")

        cls.new_model      = mlflow.pyfunc.load_model("models:/my_model@challenger")
        cls.new_vectorizer = cls.load_vectorizer_for_version("my_model", cls.new_model_version)
        print(f"\nChallenger — version: {cls.new_model_version} | "
              f"features: {len(cls.new_vectorizer.vocabulary_) if cls.new_vectorizer else 'N/A'}")

        # ── Load champion + its own vectorizer ────────────────────────────
        try:
            champion               = client.get_model_version_by_alias("my_model", "champion")
            cls.prod_model_version = champion.version
        except Exception:
            cls.prod_model_version = None

        if cls.prod_model_version and cls.prod_model_version != cls.new_model_version:
            prod_vectorizer = cls.load_vectorizer_for_version("my_model", cls.prod_model_version)

            if prod_vectorizer is not None:
                cls.prod_model      = mlflow.pyfunc.load_model("models:/my_model@champion")
                cls.prod_vectorizer = prod_vectorizer
                print(f"Champion    — version: {cls.prod_model_version} | "
                      f"features: {len(cls.prod_vectorizer.vocabulary_)}")
            else:
                cls.prod_model      = None
                cls.prod_vectorizer = None
                print(f"Champion version {cls.prod_model_version} has no vectorizer "
                      f"artifact — skipping comparison")
        else:
            cls.prod_model      = None
            cls.prod_vectorizer = None
            print("No separate champion model found — floor check only")

        # ── Holdout data — raw cleaned text ───────────────────────────────
        cls.holdout_data = pd.read_csv("data/interim/test_processed.csv")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def load_vectorizer_for_version(model_name: str, model_version: str):
        """Download the vectorizer artifact logged with this specific model version."""
        client = mlflow.MlflowClient()
        mv     = client.get_model_version(model_name, model_version)
        run_id = mv.run_id

        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="vectorizer/vectorizer.pkl",
                    dst_path=tmp
                )
                with open(local_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Could not download vectorizer for version {model_version}: {e}")
            return None

    @staticmethod
    def vectorize(vectorizer, texts) -> pd.DataFrame:
        """Transform raw text using a specific vectorizer."""
        matrix = vectorizer.transform(texts)
        return pd.DataFrame(
            matrix.toarray(),
            columns=[str(i) for i in range(matrix.shape[1])]
        )

    # ── Tests ─────────────────────────────────────────────────────────────

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        if self.new_vectorizer is None:
            self.skipTest("No vectorizer available for challenger")

        input_df   = self.vectorize(self.new_vectorizer, ["this movie was absolutely brilliant"])
        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.new_vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        if self.new_vectorizer is None:
            self.skipTest("No vectorizer available for challenger")

        X_raw     = self.holdout_data['review'].values
        y_holdout = self.holdout_data['sentiment'].values

        # ── Challenger evaluation ─────────────────────────────────────────
        X_new      = self.vectorize(self.new_vectorizer, X_raw)
        y_pred_new = self.new_model.predict(X_new)

        accuracy_new  = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new    = recall_score(y_holdout, y_pred_new)
        f1_new        = f1_score(y_holdout, y_pred_new)

        print(f"\nChallenger — accuracy: {accuracy_new:.4f} | "
              f"precision: {precision_new:.4f} | "
              f"recall: {recall_new:.4f} | f1: {f1_new:.4f}")

        # Layer 1 — absolute floor
        self.assertGreaterEqual(accuracy_new,  0.75, "Accuracy below floor")
        self.assertGreaterEqual(precision_new, 0.75, "Precision below floor")
        self.assertGreaterEqual(recall_new,    0.75, "Recall below floor")
        self.assertGreaterEqual(f1_new,        0.75, "F1 below floor")

        # ── Champion comparison ───────────────────────────────────────────
        if self.prod_model is not None and self.prod_vectorizer is not None:
            X_prod      = self.vectorize(self.prod_vectorizer, X_raw)
            y_pred_prod = self.prod_model.predict(X_prod)

            accuracy_prod  = accuracy_score(y_holdout, y_pred_prod)
            precision_prod = precision_score(y_holdout, y_pred_prod)
            recall_prod    = recall_score(y_holdout, y_pred_prod)
            f1_prod        = f1_score(y_holdout, y_pred_prod)

            print(f"Champion    — accuracy: {accuracy_prod:.4f} | "
                  f"precision: {precision_prod:.4f} | "
                  f"recall: {recall_prod:.4f} | f1: {f1_prod:.4f}")

            self.assertGreaterEqual(accuracy_new,  accuracy_prod  - 0.01, "Accuracy regression")
            self.assertGreaterEqual(precision_new, precision_prod - 0.01, "Precision regression")
            self.assertGreaterEqual(recall_new,    recall_prod    - 0.01, "Recall regression")
            self.assertGreaterEqual(f1_new,        f1_prod        - 0.01, "F1 regression")
        else:
            print("No champion to compare against — floor check only")


if __name__ == "__main__":
    unittest.main()