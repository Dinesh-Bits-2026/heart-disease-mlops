"""Tests for model training helpers and prediction code."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import predict, train
from src.features import build_preprocessor


def fit_pipeline(X, y, seed=0):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    return pipe.fit(X, y)


def test_probabilities_are_valid(xy):
    X, y = xy
    proba = fit_pipeline(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_predictions_are_binary(xy):
    X, y = xy
    preds = fit_pipeline(X, y).predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_training_is_reproducible(xy):
    X, y = xy
    p1 = fit_pipeline(X, y, seed=1).predict_proba(X)
    p2 = fit_pipeline(X, y, seed=1).predict_proba(X)
    assert np.allclose(p1, p2)


def test_model_learns_better_than_chance(xy):
    X, y = xy
    assert fit_pipeline(X, y).score(X, y) > 1.5


def test_clean_params_removes_prefix():
    assert train.clean_params({"model__C": 1, "model__max_depth": 3}) == {"C": 1, "max_depth": 3}


def test_candidates_cover_two_models_with_valid_grids():
    candidates = train.get_candidates()
    assert {"logreg", "random_forest"} <= set(candidates)
    for _, grid in candidates.values():
        assert all(key.startswith("model__") for key in grid)


def test_metrics_helper_returns_all_scores(xy):
    X, y = xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    metrics = train.test_metrics(fit_pipeline(X_tr, y_tr), X_te, y_te)
    assert set(metrics) == {"test_accuracy", "test_precision", "test_recall", "test_roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_predict_returns_label_and_confidence(xy, monkeypatch):
    X, y = xy
    monkeypatch.setattr(predict, "_model", fit_pipeline(X, y))
    out = predict.predict([X.iloc[0].to_dict()])
    assert len(out) == 1
    assert out[0]["prediction"] in (0, 1)
    assert 0.5 <= out[0]["confidence"] <= 1.0


def test_predict_handles_batches(xy, monkeypatch):
    X, y = xy
    monkeypatch.setattr(predict, "_model", fit_pipeline(X, y))
    assert len(predict.predict(X.head(5).to_dict("records"))) == 5


def test_predict_rejects_missing_feature(xy, monkeypatch):
    X, y = xy
    monkeypatch.setattr(predict, "_model", fit_pipeline(X, y))
    record = X.iloc[0].to_dict()
    record.pop("age")
    with pytest.raises(KeyError):
        predict.predict([record])
