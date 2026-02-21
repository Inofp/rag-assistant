import json
from pathlib import Path
import numpy as np

from api.app.deps import deps


def recall_at_k(targets, ranked, k):
    top = set(ranked[:k])
    return 1.0 if any(t in top for t in targets) else 0.0


def mrr_at_k(targets, ranked, k):
    for i, x in enumerate(ranked[:k], start=1):
        if x in targets:
            return 1.0 / i
    return 0.0


def main():
    p = deps.pipeline()
    qa = [json.loads(x) for x in Path("data/eval/qa.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    ks = [1, 3, 5, 10]
    rec = {k: [] for k in ks}
    mrr = {k: [] for k in ks}

    for item in qa:
        q = item["question"]
        targets = item["relevant_doc_ids"]
        hits = p.retriever.retrieve(q, top_k=max(ks))
        ranked = [h.doc_id for h in hits]
        for k in ks:
            rec[k].append(recall_at_k(targets, ranked, k))
            mrr[k].append(mrr_at_k(targets, ranked, k))

    report = {
        "n": len(qa),
        "recall": {str(k): float(np.mean(v)) for k, v in rec.items()},
        "mrr": {str(k): float(np.mean(v)) for k, v in mrr.items()},
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/retrieval_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()