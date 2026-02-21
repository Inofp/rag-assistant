from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from api.intent.model import IntentModel


def load_jsonl(path: str) -> Tuple[List[str], List[str]]:
    xs, ys = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        xs.append(obj["text"])
        ys.append(obj["label"])
    return xs, ys


def train(train_path: str) -> IntentModel:
    x_train, y_train = load_jsonl(train_path)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    pipe.fit(x_train, y_train)
    return IntentModel(pipe)