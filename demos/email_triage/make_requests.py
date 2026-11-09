"""Writes requests.jsonl: 10,000 emails of the Enron-Spam corpus (half spam, half ham), each with the questions of
questions.json, one request per line.

    python make_requests.py [--n 10000] [--out requests.jsonl]
"""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="requests.jsonl")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    questions = json.loads((HERE / "questions.json").read_text())

    seen, pools = set(), {0: [], 1: []}
    for name in ("train.jsonl", "test.jsonl"):
        path = hf_hub_download("SetFit/enron_spam", name, repo_type="dataset")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            subject, body = (r.get("subject") or "").strip(), (r.get("message") or "").strip()
            key = " ".join((subject + " " + body).split())
            if len(key) < 20 or key in seen:
                continue
            seen.add(key)
            pools[int(r["label"])].append((r["message_id"], subject, body))

    rng = random.Random(a.seed)
    rows = []
    for label in (1, 0):
        for mid, subject, body in rng.sample(pools[label], a.n // 2):
            state = (f"Subject: {subject}\n\n" if subject else "") + body[: a.max_chars]
            rows.append({"id": f"email-{label}-{mid}", "label": "spam" if label else "ham", "state": state,
                         "questions": questions})
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"wrote {len(rows)} emails to {a.out}")


if __name__ == "__main__":
    main()
