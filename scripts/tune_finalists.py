"""Tune the selected Mpro QSAR models using development CV only.

Usage:
    python scripts/tune_finalists.py --check
    python scripts/tune_finalists.py --run svr
    python scripts/tune_finalists.py --run xgboost

Hyperparameter ranges come from the committed tuning protocol.
The held-out test targets are never used.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import mlflow
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from benchmark_models import prepare
from sarscov2_qsar.models import make_model


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results/tuning"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_verify():
    """Verify our frozen protocol and selected finalists."""

    # Reuse the existing benchmark checks, including the
    # source-file hashes and development-only target loading.
    protocol, features, targets, folds = prepare()

    tuning = read_json(ROOT / "results/tuning_protocol.json")
    selection = read_json(
        ROOT / "results/benchmarks/selected_finalists.json"
    )

    # The selection must still refer to the exact benchmark
    # and tuning documents that we committed before tuning.
    source_files = {
        "benchmark_protocol_sha256":
            "results/experiment_protocol.json",
        "tuning_protocol_sha256":
            "results/tuning_protocol.json",
        "fold_results_sha256":
            "results/benchmarks/initial_cv_folds.csv",
        "summary_results_sha256":
            "results/benchmarks/initial_cv_summary.csv",
    }

    for field, filename in source_files.items():
        if sha256(ROOT / filename) != selection[field]:
            raise RuntimeError(
                f"Frozen selection checksum changed: {filename}"
            )

    finalists = selection["finalists"]

    assert [
        (f["model"], f["representation"])
        for f in finalists
    ] == [
        ("svr", "hybrid"),
        ("xgboost", "fingerprints"),
    ]

    assert selection["held_out_test_used"] is False
    assert tuning["evaluation"]["data"] == "development_only"
    assert tuning["evaluation"]["test_set_access"] == "prohibited"
    assert tuning["evaluation"]["primary_metric"] == "mean_fold_mae"
    assert tuning["evaluation"]["cv_file"] == (
        "data/splits/development_cv5.csv"
    )
    assert tuning["optimization"]["method"] == "Optuna TPESampler"
    assert tuning["optimization"]["trials_per_finalist"] == 20

    assert len(targets) == 969
    assert sorted(set(folds)) == [1, 2, 3, 4, 5]

    for finalist in finalists:
        family = finalist["model"]
        assert tuning["optimization"]["search_spaces"][family]

    return features, targets, folds, tuning, finalists


def suggest_parameters(trial, search_space):
    """Sample only parameters declared in the tuning protocol."""

    parameters = {}

    for name, rule in search_space.items():

        if isinstance(rule, list):
            parameters[name] = trial.suggest_categorical(
                name, rule
            )

        elif rule["type"] == "float_log":
            parameters[name] = trial.suggest_float(
                name,
                rule["low"],
                rule["high"],
                log=True,
            )

        elif rule["type"] == "float":
            parameters[name] = trial.suggest_float(
                name,
                rule["low"],
                rule["high"],
            )

        else:
            raise ValueError(
                f"Unsupported search rule: {name}"
            )

    return parameters


def save_trials(study, family):
    """Export persisted trial results for later analysis."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = RESULTS_DIR / f"{family}_trials.csv"
    temporary = output.with_suffix(".tmp")

    fields = [
        "trial",
        "state",
        "mean_cv_mae",
        "parameters_json",
        "fold_mae_json",
        "mean_cv_rmse",
        "mean_cv_r2",
    ]

    with temporary.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for trial in study.trials:
            writer.writerow({
                "trial": trial.number,
                "state": trial.state.name,
                "mean_cv_mae": (
                    trial.value
                    if trial.value is not None else ""
                ),
                "parameters_json": json.dumps(
                    trial.params, sort_keys=True
                ),
                "fold_mae_json": json.dumps(
                    trial.user_attrs.get("fold_mae", [])
                ),
                "mean_cv_rmse": trial.user_attrs.get(
                    "mean_cv_rmse", ""
                ),
                "mean_cv_r2": trial.user_attrs.get(
                    "mean_cv_r2", ""
                ),
            })

    temporary.replace(output)


def run_one_model(family, features, targets, folds,
                  tuning, finalists):
    """Tune one finalist with five-fold development CV."""

    finalist = next(
        f for f in finalists if f["model"] == family
    )

    representation = finalist["representation"]
    X = features[representation]

    search_space = tuning["optimization"]["search_spaces"][
        family
    ]
    budget = tuning["optimization"]["trials_per_finalist"]

    tracking = ROOT / "tracking"
    tracking.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking / 'mlflow.db'}"
    )
    mlflow.set_experiment("mpro-qsar-optuna-tuning")

    # Optuna stores completed trials in a local database.
    # Each finalist gets a separate study.
    study = optuna.create_study(
        study_name=f"mpro_{family}_{representation}_cv5",
        storage=f"sqlite:///{tracking / 'optuna.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        direction="minimize",
    )

    if any(
        trial.state == TrialState.RUNNING
        for trial in study.trials
    ):
        raise RuntimeError(
            "An earlier trial is still marked RUNNING. "
            "Inspect the study before resuming."
        )

    remaining = budget - len(study.trials)

    if remaining < 0:
        raise RuntimeError(
            "The study exceeds the frozen trial budget."
        )

    def objective(trial):
        """Return mean validation MAE for one parameter set."""

        parameters = suggest_parameters(trial, search_space)

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        with mlflow.start_run(
            run_name=(
                f"{family}-{representation}-"
                f"trial{trial.number:02d}"
            )
        ):
            mlflow.log_params({
                "family": family,
                "representation": representation,
                "trial": trial.number,
                "tuning_protocol_sha256": sha256(
                    ROOT / "results/tuning_protocol.json"
                ),
                **parameters,
            })

            for fold in range(1, 6):
                train = folds != fold
                validation = folds == fold

                # A fresh pipeline prevents scaling statistics
                # or fitted model state leaking between folds.
                model = make_model(family, representation)

                model.set_params(**{
                    f"model__{name}": value
                    for name, value in parameters.items()
                })

                model.fit(X[train], targets[train])
                predictions = model.predict(X[validation])

                fold_mae.append(float(
                    mean_absolute_error(
                        targets[validation], predictions
                    )
                ))
                fold_rmse.append(float(
                    np.sqrt(mean_squared_error(
                        targets[validation], predictions
                    ))
                ))
                fold_r2.append(float(
                    r2_score(targets[validation], predictions)
                ))

                mlflow.log_metric(
                    "fold_mae", fold_mae[-1], step=fold
                )

            cv_mae = float(mean(fold_mae))

            trial.set_user_attr("fold_mae", fold_mae)
            trial.set_user_attr(
                "mean_cv_rmse", float(mean(fold_rmse))
            )
            trial.set_user_attr(
                "mean_cv_r2", float(mean(fold_r2))
            )

            mlflow.log_metrics({
                "mean_cv_mae": cv_mae,
                "std_cv_mae": float(stdev(fold_mae)),
                "mean_cv_rmse": float(mean(fold_rmse)),
                "mean_cv_r2": float(mean(fold_r2)),
            })

        print(
            f"{family} trial {trial.number + 1:02d}/"
            f"{budget}: CV MAE={cv_mae:.4f}",
            flush=True,
        )

        return cv_mae

    print(
        f"Starting {family} ({representation}): "
        f"{remaining} trials remaining.",
        flush=True,
    )

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            callbacks=[
                lambda study, trial: save_trials(study, family)
            ],
        )

    save_trials(study, family)

    completed = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
    ]

    print(
        f"Completed {len(completed)} of "
        f"{len(study.trials)} attempted trials."
    )

    if completed:
        print(f"Best CV MAE: {study.best_value:.4f}")
        print("Best parameters:", study.best_params)

    print("Held-out test set: NOT USED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify inputs without training",
    )
    parser.add_argument(
        "--run",
        choices=("svr", "xgboost"),
        help="Tune one finalist",
    )

    args = parser.parse_args()

    if args.check == bool(args.run):
        parser.error(
            "Choose --check or --run MODEL, not both."
        )

    features, targets, folds, tuning, finalists = (
        load_and_verify()
    )

    print("=== TUNING PREFLIGHT ===")
    print("Development compounds:", len(targets))
    print("CV folds:", sorted(int(f) for f in set(folds)))

    for finalist in finalists:
        print(
            "Finalist:",
            finalist["model"],
            "+",
            finalist["representation"],
        )

    print(
        "Trials per finalist:",
        tuning["optimization"]["trials_per_finalist"],
    )
    print("Optuna version:", optuna.__version__)
    print("Held-out test targets: NOT USED")

    if args.check:
        print(
            "\nPASS: Ready for frozen hyperparameter optimization."
        )
    else:
        run_one_model(
            args.run, features, targets, folds,
            tuning, finalists,
        )


if __name__ == "__main__":
    main()
