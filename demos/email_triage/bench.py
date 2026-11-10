"""Times a server with the /v1/systemone API on the requests of requests.jsonl: each client sends its next email
as soon as its last one is answered. Prints the median time of one email and the time for all of them.

    python bench.py --url http://localhost:8000 --rows requests.jsonl --clients 64 [--model NAME] [--out answers.jsonl]
"""
import argparse
import http.client
import json
import statistics
import threading
import time
from urllib.parse import urlparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rows", default="requests.jsonl")
    ap.add_argument("--clients", type=int, default=64)
    ap.add_argument("--model", help="the model name, for servers that ask for one")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write every answer here, one line per email")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")][: a.limit]
    url = urlparse(a.url)
    results, failed, lock, todo = [None] * len(rows), [], threading.Lock(), iter(range(len(rows)))

    def client():
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=600)
        while True:
            with lock:
                i = next(todo, None)
            if i is None:
                return
            body = {"state": rows[i]["state"], "questions": rows[i]["questions"]}
            if a.model:
                body["model"] = a.model
            t = time.perf_counter()
            try:
                conn.request("POST", "/v1/systemone", json.dumps(body), {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                ok = resp.status == 200
            except OSError:
                conn.close()
                conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=600)
                ok, data = False, b""
            ms = (time.perf_counter() - t) * 1000
            if ok:
                results[i] = (ms, json.loads(data))
            else:
                failed.append(i)

    start = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(a.clients)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - start
    done = [r for r in results if r is not None]
    print(f"{len(done)} emails in {wall:.1f} s with {a.clients} clients, {len(done) / wall:.1f} emails/s, "
          f"median {statistics.median(ms for ms, _ in done):.1f} ms per email, {len(failed)} failed")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for row, r in zip(rows, results):
                if r is not None:
                    f.write(json.dumps({"id": row.get("id"), "label": row.get("label"), **r[1]}) + "\n")


if __name__ == "__main__":
    main()
