import unittest

import numpy as np

from backend.training.evaluate_grounded_trust import _json_object, metrics


class GroundedTrustEvaluationTests(unittest.TestCase):
    def test_parses_json_response(self):
        result = _json_object('{"factuality_score": 2, "status": "unsupported", "reason": "반증"}')
        self.assertEqual(result["factuality_score"], 2)

    def test_parses_markdown_fallback_response(self):
        result = _json_object("**판단:** unverifiable\n\n**사실성 점수:** 4")
        self.assertEqual(result["factuality_score"], 4)
        self.assertEqual(result["status"], "unverifiable")

    def test_extreme_auc_uses_quartile_endpoints(self):
        labels = np.asarray([0, 0.25, 0.5, 0.75, 1.0])
        scores = np.asarray([1, 2, 3, 4, 5])
        result = metrics(scores, labels)
        self.assertEqual(result["extreme_n"], 4)
        self.assertEqual(result["extreme_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
