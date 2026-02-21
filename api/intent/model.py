from __future__ import annotations

from dataclasses import dataclass
from typing import List
import joblib


@dataclass
class IntentModel:
    pipeline: object

    def predict(self, texts: List[str]) -> List[str]:
        return self.pipeline.predict(texts).tolist()

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path: str) -> "IntentModel":
        return IntentModel(joblib.load(path))