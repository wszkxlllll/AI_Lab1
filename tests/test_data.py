"""检查分层划分的互斥性与可复现性。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import stratified_indices, validation_mask


class SplitTests(unittest.TestCase):
    def test_stratification_and_reproducibility(self):
        labels = np.repeat(np.arange(10), 20)
        train, valid = stratified_indices(labels, 0.2, 42)
        repeated_train, repeated_valid = stratified_indices(labels, 0.2, 42)
        np.testing.assert_array_equal(train, repeated_train)
        np.testing.assert_array_equal(valid, repeated_valid)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(set(train) | set(valid), set(range(len(labels))))
        np.testing.assert_array_equal(np.bincount(labels[valid]), np.full(10, 4))

    def test_header_changes_do_not_hide_content_overlap(self):
        texts = np.array(["From: first\nSubject: Topic\n\nSome body", 
                          "From: second\nSubject: Re: TOPIC\n\n some  BODY ",
                          "Subject: Different\n\nanother text"])
        included = validation_mask(texts, np.array([0]), np.array([1, 2]))
        np.testing.assert_array_equal(included, [False, True])


if __name__ == "__main__":
    unittest.main()
