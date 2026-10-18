"""Send test traffic to the API so the dashboards have something to show.

Examples
    python scripts/generate_traffic.py --n 200
    python scripts/generate_traffic.py --n 200 --drift      # shifted patient population
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter


def make_record(drift: bool = False) -> dict:
    """Random patient record. With drift=True the population is older, with higher
    cholesterol and lower peak heart rate than the training data."""
    return {
        "age": random.randint(60, 76) if drift else random.randint(29, 77),
        "sex": random.randint(0, 1),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(94, 200),
        "chol": random.randint(300, 560) if drift else random.randint(126, 400),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(71, 120) if drift else random.randint(71, 202),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6), 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n", type=int, default=100, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    parser.add_argument("--drift", action="store_true", help="send a shifted population")
    parser.add_argument("--bad-rate", type=float, default=0.05,
                        help="fraction of deliberately invalid requests (422s)")
    args = parser.parse_args()

    statuses = Counter()
    for _ in range(args.n):
        payload = make_record(args.drift)
        if random.random() < args.bad_rate:
            payload["age"] = 500                      # out of range -> 422
        statuses[post(args.url, payload)] += 1
        time.sleep(args.delay)

    print("Done. Status codes:", dict(statuses))


if __name__ == "__main__":
    main()
