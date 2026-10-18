"""Compare the API's predictions for a normal and a shifted (drift) patient population.

The API must be running, for example:   docker compose up -d
Then, from the repo root with the venv active:
    python scripts/drift_check.py
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_traffic import make_record  # noqa: E402


def run_batch(url: str, n: int, drift: bool) -> dict:
    disease, conf, risk, failed = 0, [], [], 0
    for _ in range(n):
        req = urllib.request.Request(url, data=json.dumps(make_record(drift)).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, ValueError):
            failed += 1
            continue
        disease += int(body["prediction"])
        conf.append(float(body["confidence"]))
        risk.append(float(body["risk_probability"]))
    ok = len(conf)
    if ok == 0:
        sys.exit("No successful responses - is the API running on " + url + " ?")
    return {"requests_ok": ok, "failed": failed, "pct_disease": 100 * disease / ok,
            "mean_confidence": sum(conf) / ok, "mean_risk": sum(risk) / ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    normal = run_batch(args.url, args.n, drift=False)
    shifted = run_batch(args.url, args.n, drift=True)

    print(f"{'batch':<10}{'requests':>10}{'% disease':>12}{'mean confidence':>18}{'mean risk':>12}")
    for name, r in (("normal", normal), ("shifted", shifted)):
        print(f"{name:<10}{r['requests_ok']:>10}{r['pct_disease']:>11.1f}%"
              f"{r['mean_confidence']:>18.3f}{r['mean_risk']:>12.3f}")

    up = "up" if shifted["pct_disease"] > normal["pct_disease"] else "down"
    conf = "rose" if shifted["mean_confidence"] > normal["mean_confidence"] else "fell"
    print("\nSentence to adapt for the report (check it matches what you see on the dashboard):")
    print(f"  After the shifted batch the share of disease predictions went {up} from "
          f"{normal['pct_disease']:.0f}% to {shifted['pct_disease']:.0f}% and the average "
          f"confidence {conf} from {normal['mean_confidence']:.2f} to {shifted['mean_confidence']:.2f}.")


if __name__ == "__main__":
    main()
