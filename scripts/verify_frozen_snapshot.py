"""Check the archived QSAR evaluation without running the model.

This script verifies file provenance, train/test separation,
and previously saved evaluation metrics.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = ROOT / "results"

    selection_path = results / "final_model_selection.json"
    manifest_path = results / "final_training_manifest.json"
    metrics_path = results / "final_test_metrics.json"
    predictions_path = results / "final_test_predictions.csv"
    diagnostics_path = (
        results / "diagnostics/heldout_error_profile.json"
    )
    split_path = ROOT / "data/splits/butina_070_seed42.csv"

    selection = read_json(selection_path)
    manifest = read_json(manifest_path)
    metrics = read_json(metrics_path)
    diagnostics = read_json(diagnostics_path)

    # The final model must match the frozen selection.
    winner = selection["winning_model"]

    require(winner["model"] == "svr", "Unexpected model")
    require(
        winner["representation"] == "hybrid",
        "Unexpected feature representation",
    )
    require(
        winner["source"] == "initial",
        "Unexpected model-selection source",
    )
    require(
        selection["test_used_for_model_selection"] is False,
        "Test data influenced model selection",
    )
    require(
        manifest["selection_sha256"] == sha256(selection_path),
        "Training selection checksum mismatch",
    )
    require(
        metrics["model_sha256"] == manifest["model_sha256"],
        "Model checksum mismatch",
    )
    require(
        metrics["test_split_sha256"] == sha256(split_path),
        "Test split checksum mismatch",
    )

    # Confirm the saved predictions correspond exactly to
    # the 267 reserved compounds and their chemical groups.
    splits = read_csv(split_path)
    predictions = read_csv(predictions_path)

    train = [row for row in splits if row["split"] == "train"]
    test = [row for row in splits if row["split"] == "test"]

    require(len(train) == 969, "Wrong training count")
    require(len(test) == 267, "Wrong test count")
    require(len(predictions) == 267, "Wrong prediction count")

    train_groups = {row["group_id"] for row in train}
    test_groups = {row["group_id"] for row in test}

    require(
        train_groups.isdisjoint(test_groups),
        "Chemical-group leakage detected",
    )
    require(len(test_groups) == 136, "Wrong test group count")

    expected = {
        row["compound_id"]: row["group_id"]
        for row in test
    }
    observed = {
        row["compound_id"]: row["group_id"]
        for row in predictions
    }

    require(
        len(observed) == 267 and observed == expected,
        "Test prediction IDs or groups do not match the split",
    )

    # Independently reconstruct the three primary metrics
    # using the archived predictions, not the fitted model.
    actual = [
        float(row["true_pic50"])
        for row in predictions
    ]
    predicted = [
        float(row["predicted_pic50"])
        for row in predictions
    ]

    require(
        all(math.isfinite(v) for v in actual + predicted),
        "Non-finite predictions or targets",
    )

    errors = [
        estimate - truth
        for truth, estimate in zip(actual, predicted)
    ]

    n = len(actual)
    squared_error = sum(e * e for e in errors)
    average_target = sum(actual) / n

    total_variance = sum(
        (value - average_target) ** 2
        for value in actual
    )

    calculated = {
        "mae": sum(abs(e) for e in errors) / n,
        "rmse": math.sqrt(squared_error / n),
        "r2": 1 - squared_error / total_variance,
    }

    for name, value in calculated.items():
        require(
            math.isclose(
                value, metrics[name],
                rel_tol=0,
                abs_tol=1e-9,
            ),
            f"Archived {name} does not match predictions",
        )

    # The descriptive diagnostics must refer to the same
    # archived primary evaluation.
    require(
        diagnostics["source_predictions_sha256"]
        == sha256(predictions_path),
        "Diagnostic prediction checksum mismatch",
    )
    require(
        diagnostics["source_metrics_sha256"]
        == sha256(metrics_path),
        "Diagnostic metrics checksum mismatch",
    )

    report = {
        "status": "PASS",
        "purpose": "verification_of_frozen_research_results",
        "model": "SVR + Hybrid",
        "training_compounds": 969,
        "test_compounds": 267,
        "test_groups": 136,
        "primary_test_metrics": calculated,
        "selection_sha256": sha256(selection_path),
        "predictions_sha256": sha256(predictions_path),
        "metrics_sha256": sha256(metrics_path),
        "model_retrained": False,
        "test_predictions_regenerated": False,
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PASS: Frozen model-selection record verified.")
    print("PASS: Train/test chemical groups remain separate.")
    print("PASS: Archived test metrics independently reproduced.")
    print("PASS: Diagnostics match archived test predictions.")
    print("No model training or new predictions performed.")
    print("Report:", output)


if __name__ == "__main__":
    main()
