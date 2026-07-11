import unittest

from backend.training.evaluate_offline import (
    rank_by_category,
    rank_by_latest,
    rank_by_random,
)
from backend.training.samples import InteractionSample


class OfflineEvaluationRankingTests(unittest.TestCase):
    def setUp(self):
        self.sample = InteractionSample(
            subject_id="s1",
            history_article_ids=["history"],
            positive_article_id="a_positive",
            negative_article_ids=["z_negative"],
        )
        self.meta = {
            "history": {"category": "경제", "published_at": "2026-01-01"},
            "a_positive": {"category": "경제", "published_at": "2026-02-01"},
            "z_negative": {"category": "경제", "published_at": "2026-02-01"},
        }

    def test_category_tie_does_not_preserve_positive_first_input_order(self):
        ranked = rank_by_category(
            self.sample,
            self.meta,
            [self.sample.positive_article_id, *self.sample.negative_article_ids],
        )
        self.assertEqual(ranked, ["z_negative", "a_positive"])

    def test_latest_tie_uses_deterministic_article_id(self):
        ranked = rank_by_latest(
            self.sample,
            self.meta,
            [self.sample.positive_article_id, *self.sample.negative_article_ids],
        )
        self.assertEqual(ranked, ["z_negative", "a_positive"])

    def test_random_baseline_is_repeatable(self):
        candidates = ["a_positive", "z_negative", "m_other"]
        first = rank_by_random(self.sample, self.meta, candidates)
        second = rank_by_random(self.sample, self.meta, candidates)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
