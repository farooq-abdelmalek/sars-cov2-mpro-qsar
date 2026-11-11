"""Verify and archive the completed held-out evaluation.

Reads saved predictions only. Does not refit a model,
generate new predictions, or access the original test targets.
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "final_test_metrics.json"
PREDICTIONS = RESULTS / "final_test_predictions.csv"
MANIFEST = RESULTS / "final_training_manifest.json"
SELECTION = RESULTS / "final_model_selection.json"
MODEL = ROOT / "artifacts/final_svr_hybrid_dev969.joblib"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


metrics = json.loads(METRICS.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

with PREDICTIONS.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 267
assert len({r["compound_id"] for r in rows}) == 267
assert len({r["group_id"] for r in rows}) == 136

assert metrics["evaluation"] == "primary_held_out_test"
assert metrics["n_training_compounds"] == 969
assert metrics["n_test_compounds"] == 267
assert metrics["model_sha256"] == sha256(MODEL)
assert metrics["selection_sha256"] == sha256(SELECTION)
assert manifest["model_sha256"] == sha256(MODEL)

y_true = np.array([float(r["true_pic50"]) for r in rows])
y_pred = np.array([float(r["predicted_pic50"]) for r in rows])

assert np.isfinite(y_true).all()
assert np.isfinite(y_pred).all()

errors = y_pred - y_true
absolute_errors = np.abs(errors)

recalculated = {
    "mae": mean_absolute_error(y_true, y_pred),
    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    "r2": r2_score(y_true, y_pred),
    "mean_signed_error": np.mean(errors),
    "median_absolute_error": np.median(absolute_errors),
    "p90_absolute_error": np.percentile(absolute_errors, 90),
}

for name, value in recalculated.items():
    assert np.isclose(
        value, metrics[name], rtol=0, atol=1e-10
    ), f"Mismatch in {name}"

for i, row in enumerate(rows):
    assert np.isclose(
        float(row["signed_error_pic50"]),
        errors[i], rtol=0, atol=1e-10
    )
    assert np.isclose(
        float(row["absolute_error_pic50"]),
        absolute_errors[i], rtol=0, atol=1e-10
    )

print("Saved predictions and metrics: PASS")
print("Frozen model and selection checksums: PASS")

# Keep MLflow metadata and artifacts outside version control.
tracking_dir = ROOT / "tracking"
tracking_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{tracking_dir / 'mlflow.db'}"
)

experiment_name = "mpro-qsar-final-evaluation"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    artifact_dir = tracking_dir / "mlflow_artifacts"
    artifact_dir.mkdir(exist_ok=True)

    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_dir.resolve().as_uri(),
    )
else:
    experiment_id = experiment.experiment_id

# One primary evaluation record only.
existing_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    output_format="list",
    max_results=1000,
)

if existing_runs:
    assert len(existing_runs) == 1
    existing = existing_runs[0]
    assert existing.info.status == "FINISHED"
    assert (
        existing.data.tags.get("test_metrics_sha256")
        == sha256(METRICS)
    )
    assert (
        existing.data.tags.get("test_predictions_sha256")
        == sha256(PREDICTIONS)
    )
    print("Matching MLflow evaluation already recorded.")
    print("MLflow run ID:", existing.info.run_id)
else:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, text=True
    ).strip()

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="frozen_svr_hybrid_primary_test",
    ) as run:
        mlflow.log_params({
            "model": "svr",
            "features": "hybrid",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "training_compounds": 969,
            "test_compounds": 267,
            "test_groups": 136,
        })

        mlflow.log_metrics({
            name: float(value)
            for name, value in recalculated.items()
        })
        mlflow.log_metric(
            "development_cv_mae",
            manifest["selected_model_cv_mae"],
        )

        mlflow.set_tags({
            "evaluation": "primary_held_out_test",
            "test_metrics_sha256": sha256(METRICS),
            "test_predictions_sha256": sha256(PREDICTIONS),
            "frozen_model_sha256": sha256(MODEL),
            "selection_sha256": sha256(SELECTION),
            "pre_evaluation_git_commit": git_commit,
        })

        mlflow.log_artifact(str(METRICS))
        mlflow.log_artifact(str(PREDICTIONS))
        mlflow.log_artifact(str(MANIFEST))
        mlflow.log_artifact(str(SELECTION))
        mlflow.log_artifact(str(MODEL))

        print("MLflow run ID:", run.info.run_id)

print("\nPASS: Primary evaluation preserved in MLflow.")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R2: {metrics['r2']:.4f}")
