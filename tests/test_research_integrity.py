"""Regression tests for the frozen Mpro QSAR research record.

These tests use archived data and predictions only.
They never train models or regenerate held-out predictions.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_csv(relative_path):
    with (ROOT / relative_path).open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def load_json(relative_path):
    return json.loads(
        (ROOT / relative_path).read_text(encoding="utf-8")
    )


def sha256(relative_path):
    return hashlib.sha256(
        (ROOT / relative_path).read_bytes()
    ).hexdigest()


def test_train_test_groups_are_separate():
    """Chemical groups must not cross the held-out boundary."""
    rows = load_csv("data/splits/butina_070_seed42.csv")

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    assert len(train) == 969
    assert len(test) == 267

    train_ids = {r["compound_id"] for r in train}
    test_ids = {r["compound_id"] for r in test}

    assert len(train_ids) == 969
    assert len(test_ids) == 267
    assert train_ids.isdisjoint(test_ids)

    train_groups = {r["group_id"] for r in train}
    test_groups = {r["group_id"] for r in test}

    assert train_groups.isdisjoint(test_groups)
    assert len(test_groups) == 136


def test_cv_assignments_cover_development():
    """Every development compound needs a CV assignment."""
    splits = load_csv("data/splits/butina_070_seed42.csv")
    folds = load_csv("data/splits/development_cv5.csv")

    development_ids = {
        r["compound_id"]
        for r in splits
        if r["split"] == "train"
    }

    fold_ids = [r["compound_id"] for r in folds]

    assert len(fold_ids) == 969
    assert len(set(fold_ids)) == 969
    assert set(fold_ids) == development_ids


def test_feature_archive_integrity():
    """All compound IDs and feature dimensions must agree."""
    manifest_path = "data/features/feature_manifest.json"
    features_path = "data/features/moonshot_features.npz"

    manifest = load_json(manifest_path)
    compounds = load_csv(
        "data/processed/moonshot_quantitative_compounds.csv"
    )

    assert manifest["feature_artifact_sha256"] == sha256(
        features_path
    )

    with np.load(ROOT / features_path, allow_pickle=False) as data:
        ids = data["compound_ids"].tolist()
        descriptors = data["descriptors"]
        fingerprints = data["fingerprints"]

        assert descriptors.shape == (1236, 10)
        assert fingerprints.shape == (1236, 2048)

        assert np.isfinite(descriptors).all()
        assert np.isfinite(fingerprints).all()
        assert np.all(
            (fingerprints == 0) | (fingerprints == 1)
        )

    compound_ids = {r["compound_id"] for r in compounds}

    assert len(ids) == len(set(ids)) == 1236
    assert set(ids) == compound_ids


def test_final_model_was_selected_by_cv_mae():
    """The archived winner must match the original selection rule."""
    selection = load_json("results/final_model_selection.json")
    initial = load_csv(
        "results/benchmarks/initial_cv_summary.csv"
    )
    tuning = load_json(
        "results/tuning/tuning_verification.json"
    )

    scores = [float(r["mean_mae"]) for r in initial]
    scores.extend(
        s["best_tuned_cv_mae"]
        for s in tuning["studies"]
    )

    assert len(scores) == 15

    winner = selection["winning_model"]

    assert winner["model"] == "svr"
    assert winner["representation"] == "hybrid"
    assert winner["source"] == "initial"
    assert winner["mean_cv_mae"] == pytest.approx(min(scores))

    assert selection["test_used_for_model_selection"] is False


def test_saved_test_metrics_are_reproducible():
    """Recalculate test metrics without loading or running the model."""
    metrics = load_json("results/final_test_metrics.json")
    predictions = load_csv(
        "results/final_test_predictions.csv"
    )
    splits = load_csv("data/splits/butina_070_seed42.csv")

    test_groups = {
        r["compound_id"]: r["group_id"]
        for r in splits
        if r["split"] == "test"
    }

    assert len(predictions) == 267

    prediction_ids = [r["compound_id"] for r in predictions]
    assert len(set(prediction_ids)) == 267
    assert set(prediction_ids) == set(test_groups)

    for row in predictions:
        assert row["group_id"] == test_groups[row["compound_id"]]

    actual = np.array([
        float(r["true_pic50"]) for r in predictions
    ])
    predicted = np.array([
        float(r["predicted_pic50"]) for r in predictions
    ])

    errors = predicted - actual

    assert np.isfinite(actual).all()
    assert np.isfinite(predicted).all()

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    r2 = 1 - np.sum(errors ** 2) / np.sum(
        (actual - np.mean(actual)) ** 2
    )

    assert mae == pytest.approx(metrics["mae"], abs=1e-10)
    assert rmse == pytest.approx(metrics["rmse"], abs=1e-10)
    assert r2 == pytest.approx(metrics["r2"], abs=1e-10)

    assert metrics["test_split_sha256"] == sha256(
        "data/splits/butina_070_seed42.csv"
    )


def test_archived_provenance_is_consistent():
    """Evaluation and diagnostics must reference the same files."""
    selection_path = "results/final_model_selection.json"
    manifest = load_json("results/final_training_manifest.json")
    metrics = load_json("results/final_test_metrics.json")
    diagnostics = load_json(
        "results/diagnostics/heldout_error_profile.json"
    )

    assert manifest["selection_sha256"] == sha256(
        selection_path
    )
    assert metrics["selection_sha256"] == sha256(
        selection_path
    )
    assert metrics["model_sha256"] == manifest["model_sha256"]

    assert diagnostics["source_predictions_sha256"] == sha256(
        "results/final_test_predictions.csv"
    )
    assert diagnostics["source_metrics_sha256"] == sha256(
        "results/final_test_metrics.json"
    )

    assert diagnostics["primary_test_evaluation_unchanged"]
