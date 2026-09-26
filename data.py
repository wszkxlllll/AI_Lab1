"""数据读取、基本检查与分层划分。"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from text_features import message_view


def load_training(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if set(frame.columns) != {"text", "target"}:
        raise ValueError("Expected text,target columns")
    if frame.text.str.strip().eq("").any() or frame.target.isna().any():
        raise ValueError("Training data contains an empty text or label")
    if not set(frame.target).issubset(set(range(10))):
        raise ValueError("Expected integer class labels in 0..9")
    return frame


def stratified_indices(labels, fraction, seed):
    return train_test_split(np.arange(len(labels)), test_size=fraction,
                            stratify=labels, random_state=seed)


def validation_mask(texts, train, valid):
    """标记清洗后与训练文本重复的验证样本。"""
    def canonical(text):
        return " ".join(message_view(text, "joined").lower().split())

    known = {canonical(texts[i]) for i in train}
    return np.array([canonical(texts[i]) not in known for i in valid], dtype=bool)
