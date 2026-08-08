import unittest

import numpy as np

from scoring import fuse_hybrid_scores, min_max_normalize


class ScoringTests(unittest.TestCase):
    def test_min_max_normalization(self):
        result = min_max_normalize(np.array([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_hybrid_fusion_respects_alpha(self):
        bm25 = np.array([0.0, 10.0])
        dense = np.array([9.0, 1.0])
        np.testing.assert_allclose(fuse_hybrid_scores(bm25, dense, alpha=0.25), [0.75, 0.25])

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_hybrid_scores(np.array([1.0]), np.array([1.0]), alpha=1.5)


if __name__ == "__main__":
    unittest.main()
