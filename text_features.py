"""分词、邮件头解析和三种文本表示。"""

import re
import zlib
from email.parser import Parser

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


TOKEN_PATTERN = r"(?u)\b\w\w+\b"
TOKENIZER = re.compile(TOKEN_PATTERN)


def split_message(text):
    """仅解析首个空行前的邮件头，正文中的同名行仍保留。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    header, separator, body = text.partition("\n\n")
    if not separator or not re.match(r"^[\w-]+:", header):
        return "", text
    title = str(Parser().parsestr(header + "\n\n", headersonly=True).get("Subject", ""))
    title = re.sub(r"^(?:\s*re\s*:\s*)+", "", title, flags=re.I)
    return " ".join(title.split()), body


def message_view(text, view):
    if view == "raw":
        return text
    title, body = split_message(text)
    if view == "body":
        return body
    if view == "joined":
        return title + "\n" + body
    raise ValueError(f"Unknown text view: {view}")


def stable_hash(word):
    return zlib.crc32(word.encode("utf-8"))


class TextFeatures:
    def __init__(self, config, representation="tfidf", view="joined", title_weight=1.0, seed=66):
        self.config = config
        self.representation = representation
        self.view = view
        self.title_weight = title_weight
        self.seed = seed

    def _vectorizer(self, title=False):
        return TfidfVectorizer(
            max_features=self.config["title_max_features" if title else "max_features"],
            min_df=self.config["min_df"], max_df=self.config["max_df"],
            token_pattern=TOKEN_PATTERN, lowercase=True, ngram_range=(1, 1),
            use_idf=self.representation != "bow", sublinear_tf=False, norm="l2",
            dtype=np.float64,
        )

    def fit_transform(self, texts):
        if self.representation == "word2vec":
            documents = [TOKENIZER.findall(message_view(t, self.view).lower()) for t in texts]
            self.embedding = Word2Vec(
                sentences=documents, workers=1, seed=self.seed, hashfxn=stable_hash,
                sg=1, **self.config["word2vec"],
            )
            return self._average(documents)
        if self.view == "separate":
            parts = [split_message(t) for t in texts]
            self.title_vectorizer = self._vectorizer(title=True)
            self.body_vectorizer = self._vectorizer()
            title = self.title_vectorizer.fit_transform([p[0] for p in parts])
            body = self.body_vectorizer.fit_transform([p[1] for p in parts])
            return self._combine(title, body)
        self.vectorizer = self._vectorizer()
        return self.vectorizer.fit_transform([message_view(t, self.view) for t in texts])

    def transform(self, texts):
        if self.representation == "word2vec":
            return self._average([TOKENIZER.findall(message_view(t, self.view).lower()) for t in texts])
        if self.view == "separate":
            parts = [split_message(t) for t in texts]
            return self._combine(
                self.title_vectorizer.transform([p[0] for p in parts]),
                self.body_vectorizer.transform([p[1] for p in parts]),
            )
        return self.vectorizer.transform([message_view(t, self.view) for t in texts])

    def _combine(self, title, body):
        # 拼接后再次归一化，使不同标题权重下的文档长度可比。
        return normalize(sparse.hstack([title * self.title_weight, body], format="csr"))

    def _average(self, documents):
        matrix = np.zeros((len(documents), self.config["word2vec"]["vector_size"]), dtype=np.float64)
        for i, tokens in enumerate(documents):
            known = [t for t in tokens if t in self.embedding.wv]
            if known:
                matrix[i] = self.embedding.wv[known].mean(axis=0)
        return normalize(matrix)

    def feature_names(self):
        if self.representation == "word2vec":
            return [f"embedding_{i}" for i in range(self.config["word2vec"]["vector_size"])]
        if self.view == "separate":
            return (["title:" + s for s in self.title_vectorizer.get_feature_names_out()]
                    + ["body:" + s for s in self.body_vectorizer.get_feature_names_out()])
        return list(self.vectorizer.get_feature_names_out())
