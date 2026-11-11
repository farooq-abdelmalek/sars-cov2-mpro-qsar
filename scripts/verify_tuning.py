"""Verify completed Optuna experiments against CSV and MLflow.

This script does not fit models or evaluate test compounds.
It checks that all tuning results are complete and consistent
before we select the final QSAR model.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import mlflow
import optuna
from optuna.trial import TrialState


ROOT = Path(__file__).resolve().parents[1]
TUNING_DIR = ROOT / "results/tuning"
OUTPUT = TUNING_DIR / "tuning_verification.json"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def equal_number(a, b):
    return math.isclose(
        float(a), float(b),
        rel_tol=0, abs_tol=1e-9
    )


tuning_path = ROOT / "results/tuning_protocol.json"
benchmark_path = ROOT / "results/experiment_protocol.json"
selection_path = (
    ROOT / "results/benchmarks/selected_finalists.json"
)

tuning = json.loads(tuning_path.read_text())
selection = json.loads(selection_path.read_text())

# Confirm that the selected finalists still refer to the
# exact experimental protocols used before optimization.
assert selection["tuning_protocol_sha256"] == checksum(tuning_path)
assert selection["benchmark_protocol_sha256"] == checksum(benchmark_path)

assert tuning["optimization"]["trials_per_finalist"] == 20
assert selection["held_out_test_used"] is False

optuna_storage = f"sqlite:///{ROOT / 'tracking/optuna.db'}"
mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'tracking/mlflow.db'}"
)

experiment = mlflow.get_experiment_by_name(
    "mpro-qsar-optuna-tuning"
)
assert experiment is not None

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="list",
    max_results=1000,
)

client = mlflow.tracking.MlflowClient()
reports = []

for finalist in selection["finalists"]:
    family = finalist["model"]
    representation = finalist["representation"]

    study = optuna.load_study(
        study_name=f"mpro_{family}_{representation}_cv5",
        storage=optuna_storage,
    )

    trials = study.trials

    # Every planned trial must have completed successfully.
    assert len(trials) == 20, (family, len(trials))
    assert all(
        trial.state == TrialState.COMPLETE
        for trial in trials
    ), f"Incomplete trials: {family}"

    csv_path = TUNING_DIR / f"{family}_trials.csv"
    csv_rows = read_csv(csv_path)

    assert len(csv_rows) == 20

    exported = {
        int(row["trial"]): row
        for row in csv_rows
    }
    assert set(exported) == set(range(20))

    family_runs = {
        int(run.data.params["trial"]): run
        for run in runs
        if run.data.params.get("family") == family
        and run.data.params.get("representation") == representation
    }
    assert set(family_runs) == set(range(20)), (
        f"Missing or duplicate MLflow runs: {family}"
    )

    for trial in trials:
        row = exported[trial.number]
        run = family_runs[trial.number]

        fold_mae = trial.user_attrs["fold_mae"]
        assert len(fold_mae) == 5

        # Independent recalculation of the trial objective.
        assert equal_number(mean(fold_mae), trial.value)

        # Verify the exported trial data.
        assert row["state"] == "COMPLETE"
        assert equal_number(row["mean_cv_mae"], trial.value)
        assert json.loads(row["parameters_json"]) == trial.params

        assert all(
            equal_number(a, b)
            for a, b in zip(
                json.loads(row["fold_mae_json"]),
                fold_mae
            )
        )

        for metric in ("mean_cv_rmse", "mean_cv_r2"):
            assert equal_number(
                row[metric], trial.user_attrs[metric]
            )

        # Verify MLflow's stored trial score and parameters.
        assert run.info.status == "FINISHED"
        assert equal_number(
            run.data.metrics["mean_cv_mae"],
            trial.value,
        )
        assert (
            run.data.params["tuning_protocol_sha256"]
            == checksum(tuning_path)
        )

        # Verify all five individual validation-fold errors.
        history = sorted(
            client.get_metric_history(
                run.info.run_id, "fold_mae"
            ),
            key=lambda metric: metric.step,
        )

        assert [m.step for m in history] == [1, 2, 3, 4, 5]
        assert all(
            equal_number(record.value, value)
            for record, value in zip(history, fold_mae)
        )

    best = study.best_trial

    report = {
        "model": family,
        "representation": representation,
        "completed_trials": len(trials),
        "initial_cv_mae": finalist["initial_mean_cv_mae"],
        "best_tuned_cv_mae": best.value,
        "best_trial_number": best.number,
        "best_parameters": best.params,
        "best_mean_cv_rmse": best.user_attrs["mean_cv_rmse"],
        "best_mean_cv_r2": best.user_attrs["mean_cv_r2"],
        "trial_csv_sha256": checksum(csv_path),
        "optuna_csv_mlflow_consistency": "PASS",
    }

    reports.append(report)

    print(f"\n=== {family.upper()} ===")
    print("Representation:", representation)
    print("Completed trials:", len(trials))
    print("Initial CV MAE:", round(report["initial_cv_mae"], 4))
    print("Best tuned CV MAE:", round(best.value, 4))
    print("Best tuned CV RMSE:", round(report["best_mean_cv_rmse"], 4))
    print("Best tuned CV R2:", round(report["best_mean_cv_r2"], 4))
    print("Optuna / CSV / MLflow: PASS")

verification = {
    "status": "verified_before_final_model_freeze",
    "tuning_protocol_sha256": checksum(tuning_path),
    "selection_sha256": checksum(selection_path),
    "studies": reports,
    "held_out_test_evaluated": False,
}

# Refuse to silently replace a previous verification.
if OUTPUT.exists():
    assert json.loads(OUTPUT.read_text()) == verification
else:
    OUTPUT.write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )

print("\nPASS: Both optimization studies verified.")
print("Saved:", OUTPUT)
print("Held-out test: NOT USED")
