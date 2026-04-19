import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

os.environ.setdefault("CAPSTONE_TEST", "dummy-token-for-testing")


class TestFastAPIApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        patch("dagshub.init").start()
        patch("mlflow.set_tracking_uri").start()
        patch("backend.main.get_latest_model_version", return_value="4").start()

        # Patch lifespan to do nothing — skip real model loading entirely
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def mock_lifespan(app):
            yield
        patch("backend.main.lifespan", mock_lifespan).start()

        from fastapi.testclient import TestClient
        import backend.main as main_module
        from backend.main import app

        # ✅ Directly inject mocks into the module globals
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])

        mock_vectorizer = MagicMock()
        mock_transform = MagicMock()
        mock_transform.toarray.return_value = np.zeros((1, 100))
        mock_transform.shape = (1, 100)
        mock_vectorizer.transform.return_value = mock_transform

        main_module.model         = mock_model
        main_module.vectorizer    = mock_vectorizer
        main_module.model_version = "4"

        # Mock normalize_text so NLTK is never called
        patch(
            "backend.main.normalize_text",
            return_value="this movie great"
        ).start()

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        patch.stopall()

    def test_health_returns_200(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_health_returns_ok(self):
        self.assertEqual(
            self.client.get("/health").json()["status"], "ok"
        )

    def test_predict_returns_200(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        self.assertEqual(resp.status_code, 200)

    def test_predict_returns_sentiment(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        self.assertIn(resp.json()["sentiment"], ["Positive", "Negative"])

    def test_predict_returns_confidence(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        self.assertIsInstance(resp.json()["confidence"], float)

    def test_predict_returns_clean_text(self):
        resp = self.client.post("/predict", json={"text": "This movie was great!"})
        self.assertIn("clean_text", resp.json())

    def test_predict_missing_text_returns_422(self):
        self.assertEqual(
            self.client.post("/predict", json={}).status_code, 422
        )

    def test_metrics_returns_200(self):
        self.assertEqual(
            self.client.get("/metrics").status_code, 200
        )


if __name__ == "__main__":
    unittest.main()