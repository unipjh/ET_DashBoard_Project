import importlib.util
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import recommend, repo


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
        self.assertEqual(body[0]["rec_source"], "profile")

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


class RecommendationRankingLogicTests(unittest.TestCase):
    def test_profile_vector_weights_recent_article_higher(self):
        old = [0.0] * 768
        old[0] = 1.0
        recent = [0.0] * 768
        recent[1] = 1.0
        # 시간순(과거→최근) 입력이므로 마지막 기사(recent)의 차원이 더 커야 한다
        profile = recommend.build_profile_vector([
            {"embed_summary": old},
            {"embed_summary": recent},
        ])
        self.assertGreater(profile[1], profile[0])

    def test_profile_vector_stays_in_raw_embedding_space(self):
        raw = [0.0] * 768
        raw[0] = 1.0
        learned = [0.0] * 768
        learned[1] = 1.0

        profile = recommend.build_profile_vector([{
            "embed_summary": raw,
            "learned_embedding": learned,
        }])

        self.assertEqual(profile[0], 1.0)
        self.assertEqual(profile[1], 0.0)

    def test_blend_vectors_prefers_short_term(self):
        short = [1.0, 0.0]
        long = [0.0, 1.0]
        blended = recommend._blend_vectors(short, long, 0.65)
        self.assertAlmostEqual(blended[0], 0.65)
        self.assertAlmostEqual(blended[1], 0.35)

    def test_blend_vectors_falls_back_when_one_side_empty(self):
        short = [1.0, 0.0]
        empty = [0.0, 0.0]
        self.assertEqual(recommend._blend_vectors(short, empty, 0.65), short)
        self.assertEqual(recommend._blend_vectors(empty, short, 0.65), short)

    def test_trust_guardrail_demotes_low_trust_articles(self):
        records = [
            {"article_id": "low", "trust_score": 20},
            {"article_id": "high", "trust_score": 85},
            {"article_id": "unanalyzed", "trust_score": 0},
        ]
        ordered = recommend._apply_trust_guardrail(records)
        self.assertEqual([r["article_id"] for r in ordered], ["high", "unanalyzed", "low"])

    def test_profile_reranker_combines_semantic_and_history_category(self):
        records = [
            {
                "article_id": "semantic_only",
                "score": 0.90,
                "category": "IT/과학",
                "trust_score": 80,
                "published_at": "2026-07-01",
            },
            {
                "article_id": "history_match",
                "score": 0.40,
                "category": "경제",
                "trust_score": 80,
                "published_at": "2026-07-01",
            },
        ]
        history = [{"article_id": "h1", "category": "경제"}]

        ordered = recommend._rerank_profile_candidates(records, history)

        self.assertEqual(ordered[0]["article_id"], "history_match")
        self.assertGreater(ordered[0]["_rank_score"], ordered[1]["_rank_score"])

    def test_diversify_caps_single_category_share(self):
        records = [
            {"article_id": f"it{i}", "category": "IT/과학"} for i in range(5)
        ] + [{"article_id": "eco1", "category": "경제"}]
        ordered = recommend._diversify_by_category(records, limit=5)
        top5_categories = [r["category"] for r in ordered[:5]]
        self.assertIn("경제", top5_categories)

    @patch("backend.services.recommend.repo.get_latest_articles")
    def test_explore_slot_replaces_last_item_with_new_category(self, latest):
        latest.return_value = [
            {"article_id": "seen_cat", "category": "IT/과학"},
            {"article_id": "fresh", "category": "세계"},
        ]
        records = [
            {"article_id": f"r{i}", "category": "IT/과학", "rec_source": "profile"}
            for i in range(5)
        ]
        history = [{"article_id": "h1", "category": "IT/과학"}]
        result = recommend._add_explore_slot(records, history, limit=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[-1]["article_id"], "fresh")
        self.assertEqual(result[-1]["rec_source"], "explore")


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
