"""Evaluate the frozen SVR on the held-out Moonshot test compounds.

Usage:
    python scripts/evaluate_final_test.py --check
    python scripts/evaluate_final_test.py --run

The --check mode verifies the model and data without evaluating
test targets. The --run mode performs the primary test evaluation
and refuses to overwrite existing results.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from benchmark_models import prepare


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SELECTION = RESULTS / "final_model_selection.json"
MANIFEST = RESULTS / "final_training_manifest.json"
MODEL = ROOT / "artifacts/final_svr_hybrid_dev969.joblib"

METRICS = RESULTS / "final_test_metrics.json"
PREDICTIONS = RESULTS / "final_test_predictions.csv"


def sha256(path):
    """Return a file's SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    """Load a CSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def preflight():
    """Check the frozen model, data, and train/test separation."""

    # This existing function checks our frozen source files,
    # model configurations, package versions, and development folds.
    protocol, _, development_targets, _ = prepare()

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    winner = selection["winning_model"]

    assert selection["status"] == "frozen_before_held_out_evaluation"
    assert winner["model"] == "svr"
    assert winner["representation"] == "hybrid"
    assert winner["source"] == "initial"

    assert manifest["selection_sha256"] == sha256(SELECTION)
    assert manifest["model_sha256"] == sha256(MODEL)
    assert manifest["training_compounds"] == 969
    assert manifest["test_evaluated"] is False

    assert len(development_targets) == 969

    # Load only the model saved by our own training script.
    model = joblib.load(MODEL)

    assert repr(model.named_steps["model"]) == selection["estimator"]
    assert (
        repr(model.named_steps["preprocess"])
        == selection["preprocessing"]
    )

    scaler = model.named_steps["preprocess"].named_transformers_[
        "descriptors"
    ]
    assert int(scaler.n_samples_seen_) == 969

    split_rows = read_csv(
        ROOT / "data/splits/butina_070_seed42.csv"
    )

    train_rows = [r for r in split_rows if r["split"] == "train"]
    test_rows = [r for r in split_rows if r["split"] == "test"]

    assert len(train_rows) == 969
    assert len(test_rows) == 267

    train_ids = {r["compound_id"] for r in train_rows}
    test_ids = {r["compound_id"] for r in test_rows}

    assert len(train_ids) == 969
    assert len(test_ids) == 267
    assert train_ids.isdisjoint(test_ids)

    # No Butina group can occur on both sides.
    train_groups = {r["group_id"] for r in train_rows}
    test_groups = {r["group_id"] for r in test_rows}

    assert train_groups.isdisjoint(test_groups)
    assert len(test_groups) == 136

    # Read molecular features without reading test targets.
    with np.load(
        ROOT / "data/features/moonshot_features.npz",
        allow_pickle=False,
    ) as saved:
        ids = saved["compound_ids"].tolist()
        descriptors = saved["descriptors"]
        fingerprints = saved["fingerprints"]

    assert len(ids) == len(set(ids)) == 1236
    assert set(ids) == train_ids | test_ids

    # Align all test features by the stable compound ID.
    index = {compound_id: i for i, compound_id in enumerate(ids)}
    ordered_ids = sorted(test_ids)
    positions = [index[cid] for cid in ordered_ids]

    test_features = np.column_stack([
        descriptors[positions],
        fingerprints[positions],
    ])

    assert test_features.shape == (267, 2058)
    assert np.isfinite(test_features).all()

    # Refuse to overwrite a prior final evaluation.
    if METRICS.exists() or PREDICTIONS.exists():
        raise RuntimeError(
            "Final test results already exist. "
            "The primary evaluation must not be repeated."
        )

    print("=== FINAL TEST PREFLIGHT ===")
    print("Selected model: SVR + Hybrid")
    print("Training compounds: 969")
    print("Held-out compounds:", len(ordered_ids))
    print("Held-out chemical groups:", len(test_groups))
    print("Model checksum: PASS")
    print("Frozen protocol: PASS")
    print("Train/test group separation: PASS")

    return model, ordered_ids, test_features, test_rows


def evaluate(model, ids, features, test_rows):
    """Predict once and record the held-out evaluation."""

    # Test targets are accessed only in this function,
    # after the model has been frozen and verified.
    all_compounds = read_csv(
        ROOT / "data/processed/moonshot_quantitative_compounds.csv"
    )

    test_ids = set(ids)

    targets = {
        row["compound_id"]: float(row["pic50_molar"])
        for row in all_compounds
        if row["compound_id"] in test_ids
    }

    assert set(targets) == test_ids

    y_true = np.array([targets[cid] for cid in ids])
    y_pred = model.predict(features)

    assert y_pred.shape == (267,)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    errors = y_pred - y_true
    absolute_errors = np.abs(errors)

    metrics = {
        "evaluation": "primary_held_out_test",
        "model": "SVR + Hybrid",
        "n_training_compounds": 969,
        "n_test_compounds": 267,
        "n_test_groups": 136,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_signed_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "p90_absolute_error": float(
            np.percentile(absolute_errors, 90)
        ),
        "selection_sha256": sha256(SELECTION),
        "model_sha256": sha256(MODEL),
        "test_split_sha256": sha256(
            ROOT / "data/splits/butina_070_seed42.csv"
        ),
    }

    group_by_id = {
        row["compound_id"]: row["group_id"]
        for row in test_rows
    }

    prediction_rows = [
        {
            "compound_id": cid,
            "group_id": group_by_id[cid],
            "true_pic50": float(y_true[i]),
            "predicted_pic50": float(y_pred[i]),
            "signed_error_pic50": float(errors[i]),
            "absolute_error_pic50": float(absolute_errors[i]),
        }
        for i, cid in enumerate(ids)
    ]

    # Save the predictions for later diagnostics.
    # Subsequent analyses should use this saved file rather
    # than rerunning the primary test evaluation.
    with PREDICTIONS.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(prediction_rows[0])
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    METRICS.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== FINAL HELD-OUT TEST RESULTS ===")
    print(f"MAE:  {metrics['mae']:.4f} pIC50")
    print(f"RMSE: {metrics['rmse']:.4f} pIC50")
    print(f"R2:   {metrics['r2']:.4f}")
    print(
        "Median absolute error:",
        round(metrics["median_absolute_error"], 4),
    )
    print(
        "90th percentile absolute error:",
        round(metrics["p90_absolute_error"], 4),
    )
    print("\nSaved:", METRICS)
    print("Saved:", PREDICTIONS)
    print("Primary test evaluation completed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")

    args = parser.parse_args()

    model, ids, features, test_rows = preflight()

    if args.check:
        print("\nPASS: Ready for the primary held-out evaluation.")
        print("Test targets: NOT EVALUATED")
    else:
        evaluate(model, ids, features, test_rows)


if __name__ == "__main__":
    main()
