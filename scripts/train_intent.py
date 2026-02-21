from api.intent.train import train
from sklearn.metrics import classification_report
import json
from pathlib import Path


def load_jsonl(path: str):
    xs, ys = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        xs.append(obj["text"])
        ys.append(obj["label"])
    return xs, ys


def main():
    model = train("data/intents/train.jsonl")
    x_val, y_val = load_jsonl("data/intents/val.jsonl")
    y_pred = model.predict(x_val)
    print(classification_report(y_val, y_pred, digits=4))
    Path("reports").mkdir(parents=True, exist_ok=True)
    model.save("reports/intent_model.joblib")


if __name__ == "__main__":
    main()