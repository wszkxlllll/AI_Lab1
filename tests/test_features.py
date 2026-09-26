"""检查邮件字段解析、归一化和验证集词表隔离。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_features import TextFeatures, message_view, split_message


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(max_features=100, title_max_features=30, min_df=1, max_df=1.0)

    def test_header_fields_and_body_are_separate(self):
        text = "From: author@example.org\r\nSubject: Re: Re: Motor\r\n speed\r\nOrganization: School\r\n\r\nSubject: keep this body line\r\nengine"
        title, body = split_message(text)
        self.assertEqual(title, "Motor speed")
        self.assertEqual(body, "Subject: keep this body line\nengine")
        self.assertNotIn("author@example.org", message_view(text, "joined"))

    def test_plain_text_is_not_discarded(self):
        self.assertEqual(split_message("a plain paragraph\n\nanother paragraph"),
                         ("", "a plain paragraph\n\nanother paragraph"))

    def test_validation_words_cannot_enter_vocabulary(self):
        features = TextFeatures(self.config)
        features.fit_transform(["Subject: train\n\napple orange", "Subject: other\n\nengine wheel"])
        before = dict(features.vectorizer.vocabulary_)
        transformed = features.transform(["Subject: unseenword\n\nneverobserved"])
        self.assertEqual(before, features.vectorizer.vocabulary_)
        self.assertNotIn("neverobserved", before)
        self.assertEqual(transformed.nnz, 0)

    def test_bow_and_tfidf_share_vocabulary_and_norm(self):
        texts = ["apple apple orange", "orange engine", "engine wheel"]
        bow, tfidf = TextFeatures(self.config, "bow"), TextFeatures(self.config, "tfidf")
        a, b = bow.fit_transform(texts), tfidf.fit_transform(texts)
        self.assertEqual(bow.feature_names(), tfidf.feature_names())
        np.testing.assert_allclose(np.asarray(a.multiply(a).sum(axis=1)), 1)
        np.testing.assert_allclose(np.asarray(b.multiply(b).sum(axis=1)), 1)

    def test_separate_view_handles_missing_title(self):
        texts = ["Subject: title\n\napple orange", "plain body engine"]
        features = TextFeatures(self.config, view="separate", title_weight=2)
        matrix = features.fit_transform(texts)
        np.testing.assert_allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)), 1)
        names = features.feature_names()
        self.assertIn("title:title", names)
        self.assertIn("body:engine", names)

    def test_word_vectors_do_not_learn_from_validation(self):
        config = {**self.config, "word2vec": dict(vector_size=8, window=2, min_count=1, epochs=3)}
        texts = ["apple orange apple", "engine wheel engine"]
        first, second = TextFeatures(config, "word2vec"), TextFeatures(config, "word2vec")
        a, b = first.fit_transform(texts), second.fit_transform(texts)
        np.testing.assert_array_equal(a, b)
        result = first.transform(["neverobserved"])
        self.assertNotIn("neverobserved", first.embedding.wv)
        np.testing.assert_array_equal(result, np.zeros((1, 8)))


if __name__ == "__main__":
    unittest.main()
