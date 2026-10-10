"""Shared fixtures: a small synthetic dataset so tests never depend on the network."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import FEATURES, TARGET, build_preprocessor


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n).astype(float),
        "trestbps": rng.normal(130, 15, n).round(),
        "chol": rng.normal(245, 45, n).round(),
        "thalach": rng.normal(150, 20, n).round(),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "cp": rng.integers(1, 5, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "slope": rng.integers(1, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "ca": rng.integers(0, 4, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
    })
    score = (df["age"] / 10 + df["oldpeak"] + df["exang"] * 1.5
             - df["thalach"] / 50 + rng.normal(0, 0.5, n))
    df[TARGET] = (score > score.median()).astype(int)
    return df


@pytest.fixture
def xy(sample_df):
    """Fresh copies of features and target for each test (safe to modify)."""
    return sample_df[FEATURES].copy(), sample_df[TARGET].copy()


@pytest.fixture(scope="session")
def fitted_model(sample_df):
    """A small trained pipeline, shared by model and API tests."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    return pipe.fit(sample_df[FEATURES], sample_df[TARGET])
