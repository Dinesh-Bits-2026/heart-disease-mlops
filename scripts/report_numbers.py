"""Print the numbers you need for the report text blanks.

Run from the repo root with the venv active (after python -m src.train):
    python scripts/report_numbers.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "heart-disease-classification"


def disease_rates():
    df = pd.read_csv(ROOT / "data" / "heart_clean.csv")
    print("=" * 70)
    print("1) DISEASE RATE BY CATEGORY  (share of patients with target = 1)")
    print("=" * 70)
    for col in ["thal", "ca", "cp", "exang", "slope", "sex", "fbs"]:
        g = df.groupby(col)["target"].agg(["mean", "count"])
        g["mean"] = (g["mean"] * 100).round(1)
        print(f"\n{col}:")
        print(g.rename(columns={"mean": "% with disease", "count": "patients"}).to_string())


def tracking():
    db = ROOT / "mlflow.db"
    print("\n" + "=" * 70)
    print("2) MLFLOW: training runs and grid-search spread")
    print("=" * 70)
    if not db.exists():
        print("mlflow.db not found - run: python -m src.train")
        return
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    df = mlflow.search_runs(experiment_names=[EXPERIMENT])
    parents = df[df["params.model_type"].notna()].sort_values("start_time")

    print("\nTraining runs (oldest first) - did you change seed or settings between runs?")
    cols = [c for c in ["start_time", "params.model_type", "params.seed", "params.test_size",
                        "metrics.cv_roc_auc", "metrics.cv_recall", "metrics.test_roc_auc"]
            if c in parents.columns]
    shown = parents[cols].copy()
    shown["start_time"] = shown["start_time"].astype(str).str[:19]
    print(shown.round(3).to_string(index=False))
    print("\nDistinct seeds used:", sorted(parents["params.seed"].dropna().unique()))

    grid = df[df["tags.stage"] == "grid_search"]
    for model in ["logreg", "random_forest"]:
        mp = parents[parents["params.model_type"] == model]
        if mp.empty:
            continue
        run_id = mp.iloc[-1]["run_id"]
        kids = grid[grid["tags.mlflow.parentRunId"] == run_id].copy()
        param_cols = [c for c in kids.columns if c.startswith("params.") and kids[c].notna().any()]
        kids = kids.sort_values("metrics.cv_roc_auc", ascending=False)
        print(f"\nGrid search of the LATEST {model} run: {len(kids)} settings tried")
        show = kids[param_cols + ["metrics.cv_roc_auc"]].round(4)
        show.columns = [c.replace("params.", "").replace("metrics.", "") for c in show.columns]
        print(show.to_string(index=False))
        if len(kids) > 1:
            best, worst = kids["metrics.cv_roc_auc"].iloc[0], kids["metrics.cv_roc_auc"].iloc[-1]
            print(f"-> best {best:.3f}, worst {worst:.3f}, difference {best - worst:.3f} ROC-AUC")


if __name__ == "__main__":
    disease_rates()
    tracking()
