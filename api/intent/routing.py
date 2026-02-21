from __future__ import annotations

import re
from pathlib import Path

from api.intent.model import IntentModel


PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{8,}\d)")
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")


class IntentRouter:
    def __init__(self, model_path: str = "reports/intent_model.joblib") -> None:
        self.model_path = model_path
        self.model = IntentModel.load(model_path) if Path(model_path).exists() else None

    def route(self, text: str) -> str:
        t = text.strip()
        if PHONE_RE.search(t) or EMAIL_RE.search(t):
            return "CTA"
        low = t.lower()
        if any(x in low for x in ["коммерческ", "кп", "счет", "счёт", "прайс", "связ", "контакт", "менеджер"]):
            return "CTA"
        if any(x in low for x in ["оператор", "человек", "позвоните", "соедините"]):
            return "OPERATOR"
        if self.model:
            try:
                return self.model.predict([t])[0]
            except Exception:
                return "RAG"
        return "RAG"