import importlib.util
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import repo


class ImpressionLoggingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.routers.logs.repo.insert_impression_log")
    def test_impression_log_uses_impression_writer(self, insert_impression_log):
        response = self.client.post(
            "/api/logs",
            json={
                "session_id": "sess_test",
                "event_type": "impression",
                "article_id": None,
                "event_data": {
                    "articles": [
                        {"article_id": "a1", "position": 0},
                        {"article_id": "a2", "position": 1},
                    ]
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        insert_impression_log.assert_called_once()

    @patch("backend.services.repo.insert_log")
    def test_impression_payload_is_normalized_for_training(self, insert_log):
        repo.insert_impression_log({
            "session_id": "sess_test",
            "event_type": "impression",
            "article_id": "should_be_ignored",
            "event_data": {
                "source": "main_list",
                "context_key": "main_list|page=2|category=경제|query=",
                "article_ids": ["a1", "a2"],
            },
        })

        normalized = insert_log.call_args.args[0]
        self.assertEqual(normalized["event_type"], "impression")
        self.assertIsNone(normalized["article_id"])
        self.assertEqual(
            normalized["event_data"]["articles"],
            [
                {"article_id": "a1", "position": 0},
                {"article_id": "a2", "position": 1},
            ],
        )


class RecommendationFallbackTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.services.recommend.repo.get_latest_articles", return_value=[])
    @patch("backend.services.recommend.repo.get_recent_user_history", return_value=[])
    def test_recommendations_cold_start_falls_back_to_latest(self, _history, latest):
        response = self.client.get("/api/recommendations?session_id=sess_test&limit=3")
        self.assertEqual(response.status_code, 200)
        latest.assert_called_once()

    @patch("backend.services.recommend.encoder_inference.is_model_ready", return_value=False)
    @patch("backend.services.recommend.repo.search_similar_chunks")
    @patch("backend.services.recommend.repo.get_recent_user_history")
    def test_recommendations_normalizes_vector_fallback_records(self, history, search, _model_ready):
        history.return_value = [
            {"article_id": f"h{idx}", "embed_summary": "[0.1,0.2]"}
            for idx in range(5)
        ]
        search.return_value = pd.DataFrame([{
            "article_id": "a1",
            "title": "Recommended",
            "source": "ET",
            "published_at": "2026-06-19",
            "summary_text": "summary",
            "trust_score": 90,
            "trust_verdict": "ok",
        }])

        response = self.client.get("/api/recommendations?session_id=sess_test&limit=1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["article_id"], "a1")
        self.assertIn("url", body[0])

    @patch("backend.services.recommend.encoder_inference.is_model_ready", return_value=False)
    @patch("backend.services.recommend.repo.get_latest_articles")
    @patch("backend.services.recommend.repo.get_latest_articles_by_categories")
    @patch("backend.services.recommend.repo.search_similar_chunks")
    @patch("backend.services.recommend.repo.get_recent_user_history")
    def test_recommendations_use_history_category_when_vectors_are_missing(
        self,
        history,
        search,
        category_latest,
        latest,
        _model_ready,
    ):
        history.return_value = [
            {"article_id": "h1", "category": "economy", "embed_summary": "[]"}
        ]
        search.return_value = pd.DataFrame()
        category_latest.return_value = [{
            "article_id": "a2",
            "title": "Category match",
            "source": "ET",
            "url": "https://example.com/a2",
            "published_at": "2026-06-19",
            "summary_text": "summary",
            "trust_score": 88,
            "trust_verdict": "ok",
            "category": "economy",
        }]

        response = self.client.get("/api/recommendations?session_id=sess_test&limit=1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["article_id"], "a2")
        category_latest.assert_called_once_with(
            categories=["economy"],
            limit=1,
            exclude_article_ids=["h1"],
        )
        latest.assert_not_called()


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class AttentionEncoderShapeTests(unittest.TestCase):
    def test_news_encoder_outputs_normalized_768_vector(self):
        import torch
        from backend.training.news_encoder import NewsEncoder

        encoder = NewsEncoder(category_dim=6)
        output = encoder(torch.randn(2, 768), torch.randn(2, 768), torch.eye(6)[:2])
        self.assertEqual(tuple(output.shape), (2, 768))
        norms = torch.linalg.norm(output, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_user_encoder_outputs_normalized_768_vector(self):
        import torch
        from backend.training.user_encoder import UserEncoder

        encoder = UserEncoder(max_history=20)
        history = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.bool)
        output = encoder(history, mask)
        self.assertEqual(tuple(output.shape), (2, 768))
        norms = torch.linalg.norm(output, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
