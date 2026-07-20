import unittest

from backend.training.samples import build_temporal_samples_from_rows


class TemporalRecommendationSampleTests(unittest.TestCase):
    def test_uses_only_impressions_observed_before_positive(self):
        events = [
            {
                "session_id": "s1",
                "event_type": "impression",
                "event_data": {"article_ids": ["n_before"]},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "h1",
                "created_at": "2026-01-01T00:01:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "positive",
                "created_at": "2026-01-01T00:02:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "impression",
                "event_data": {"article_ids": ["n_future"]},
                "created_at": "2026-01-01T00:03:00+00:00",
            },
        ]

        samples = build_temporal_samples_from_rows(events, [], negative_count=1)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].positive_article_id, "positive")
        self.assertEqual(samples[0].negative_article_ids, ["n_before"])
        self.assertNotIn("n_future", samples[0].negative_article_ids)

    def test_deduplicates_click_and_view_for_same_article(self):
        events = [
            {
                "session_id": "s1",
                "event_type": "impression",
                "event_data": {"article_ids": ["n1"]},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "a1",
                "created_at": "2026-01-01T00:01:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "view_article_detail",
                "article_id": "a1",
                "created_at": "2026-01-01T00:01:01+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "a2",
                "created_at": "2026-01-01T00:02:00+00:00",
            },
        ]

        samples = build_temporal_samples_from_rows(events, [], negative_count=1)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].history_article_ids, ["a1"])
        self.assertEqual(samples[0].positive_article_id, "a2")

    def test_skips_sample_when_observed_negatives_are_insufficient(self):
        events = [
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "a1",
                "created_at": "2026-01-01T00:01:00+00:00",
            },
            {
                "session_id": "s1",
                "event_type": "click_article",
                "article_id": "a2",
                "created_at": "2026-01-01T00:02:00+00:00",
            },
        ]

        samples = build_temporal_samples_from_rows(events, [], negative_count=1)

        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
