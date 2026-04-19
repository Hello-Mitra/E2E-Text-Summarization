"""
test_app.py — FastAPI app tests.

Replaces the old test_flask_app.py.
Patches MLflow + dagshub so no real model loading happens during CI.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

os.environ.setdefault("CAPSTONE_TEST", "dummy-token-for-testing")


class TestFastAPIApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Patch all external dependencies before importing the FastAPI app.
        Prevents real MLflow / DagsHub calls during CI.
        """
        # Patch dagshub.init
        patcher_dagshub = patch("dagshub.init")
        patcher_dagshub.start()

        # Patch mlflow.set_tracking_uri
        patcher_uri = patch("mlflow.set_tracking_uri")
        patcher_uri.start()

        # Patch mlflow.MlflowClient
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [MagicMock(version="4")]
        patcher_client = patch("mlflow.MlflowClient", return_value=mock_client)
        patcher_client.start()

        # Patch mlflow.pyfunc.load_model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        patcher_model = patch("mlflow.pyfunc.load_model", return_value=mock_model)
        patcher_model.start()

        # Patch pickle.load for vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock(
            toarray=lambda: np.zeros((1, 100)),
            shape=(1, 100)
        )
        patcher_pickle = patch("pickle.load", return_value=mock_vectorizer)
        patcher_pickle.start()

        # Patch open so pickle.load doesn't need a real file
        patcher_open = patch("builtins.open", unittest.mock.mock_open())
        patcher_open.start()

        # Now safe to import
        from fastapi.testclient import TestClient
        from backend.main import app
        cls.client = TestClient(app)

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_returns_ok(self):
        resp = self.client.get("/health")
        self.assertIn("status", resp.json())
        self.assertEqual(resp.json()["status"], "ok")

    def test_predict_returns_200(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        self.assertEqual(resp.status_code, 200)

    def test_predict_returns_sentiment(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        data = resp.json()
        self.assertIn("sentiment", data)
        self.assertIn(data["sentiment"], ["Positive", "Negative"])

    def test_predict_returns_confidence(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        data = resp.json()
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], float)

    def test_predict_returns_clean_text(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        data = resp.json()
        self.assertIn("clean_text", data)

    def test_predict_missing_text_returns_422(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 422)

    def test_metrics_endpoint_returns_200(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
