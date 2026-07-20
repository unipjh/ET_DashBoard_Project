import unittest

import numpy as np

from backend.training.persona_benchmark import (
    PERSONAS,
    ArticleMeta,
    generate_persona_samples,
)


class PersonaBenchmarkTests(unittest.TestCase):
    def test_persona_behavior_weights_sum_to_one(self):
        for persona in PERSONAS:
            total = (
                persona.category_weight
                + persona.semantic_weight
                + persona.trust_weight
                + persona.freshness_weight
                + persona.novelty_weight
            )
            self.assertAlmostEqual(total, 1.0, places=7, msg=persona.name)

    def test_generation_is_reproducible_for_same_seed(self):
        categories = ["IT/과학", "경제", "정치", "사회", "세계", "생활/문화"]
        catalog = {}
        for index in range(20):
            vector = np.zeros(768, dtype=np.float32)
            vector[index] = 1.0
            article_id = f"a{index:02d}"
            catalog[article_id] = ArticleMeta(
                article_id=article_id,
                category=categories[index % len(categories)],
                embedding=vector,
                trust=0.5 + (index % 5) * 0.1,
                freshness=index / 19,
            )

        first = generate_persona_samples(
            catalog,
            users_per_persona=1,
            sessions_per_user=4,
            impression_size=8,
            negative_count=3,
            seed=7,
        )
        second = generate_persona_samples(
            catalog,
            users_per_persona=1,
            sessions_per_user=4,
            impression_size=8,
            negative_count=3,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(PERSONAS) * 3)


if __name__ == "__main__":
    unittest.main()
