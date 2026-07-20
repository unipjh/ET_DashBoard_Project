import unittest

from backend.services import trust


def criteria(source=10, evidence=10, style=10, logic=10, clickbait=0):
    return {
        "source_credibility": {"score": source},
        "evidence_support": {"score": evidence},
        "style_neutrality": {"score": style},
        "logical_consistency": {"score": logic},
        "clickbait_risk": {"score": clickbait},
    }


class TrustScoringTests(unittest.TestCase):
    def test_current_weights_match_selected_w2(self):
        self.assertEqual(
            trust.WEIGHTS,
            {
                "source_credibility": 0.20,
                "evidence_support": 0.20,
                "style_neutrality": 0.25,
                "logical_consistency": 0.15,
                "clickbait_risk": -0.20,
            },
        )

    def test_low_evidence_cannot_be_likely_true(self):
        score = trust._weighted_sum_score(criteria(evidence=3))
        self.assertEqual(score, 69)
        self.assertEqual(trust._verdict(score), "uncertain")

    def test_response_example_matches_predicate_counts(self):
        for key, questions in trust.PREDICATES.items():
            self.assertEqual(len(trust._RESPONSE_EXAMPLE[key]), len(questions))


if __name__ == "__main__":
    unittest.main()
