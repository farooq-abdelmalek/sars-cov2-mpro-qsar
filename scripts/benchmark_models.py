"""Frozen five-fold development benchmark for Mpro QSAR."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import mlflow
import numpy as np
import sklearn
import xgboost
from rdkit import rdBase
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sarscov2_qsar.models import make_model
from sarscov2_qsar.tanimoto_knn import TanimotoKNNRegressor


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/experiment_protocol.json"
RESULTS = ROOT / "results/benchmarks"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    """Verify frozen inputs and construct development-only arrays."""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    paths = {
        "compounds": "data/processed/moonshot_quantitative_compounds.csv",
        "outer_split": "data/splits/butina_070_seed42.csv",
        "cv_folds": "data/splits/development_cv5.csv",
        "features": "data/features/moonshot_features.npz",
        "feature_manifest": "data/features/feature_manifest.json",
        "model_code": "src/sarscov2_qsar/models.py",
        "preprocessing_code": "src/sarscov2_qsar/preprocessing.py",
        "knn_code": "src/sarscov2_qsar/tanimoto_knn.py",
        "feature_code": "src/sarscov2_qsar/features.py",
    }

    for name, path in paths.items():
        actual = sha256(ROOT / path)
        expected = protocol["input_and_code_sha256"][name]
        if actual != expected:
            raise RuntimeError(f"Frozen checksum mismatch: {path}")

    versions = {
        "rdkit": rdBase.rdkitVersion,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    assert versions == protocol["software_versions"]

    splits = read_csv(ROOT / paths["outer_split"])
    folds = read_csv(ROOT / paths["cv_folds"])

    dev_rows = [r for r in splits if r["split"] == "train"]
    test_rows = [r for r in splits if r["split"] == "test"]

    assert len(dev_rows) == 969
    assert len(test_rows) == 267

    dev_groups = {r["group_id"] for r in dev_rows}
    test_groups = {r["group_id"] for r in test_rows}
    assert dev_groups.isdisjoint(test_groups)

    dev_ids = {r["compound_id"] for r in dev_rows}
    test_ids = {r["compound_id"] for r in test_rows}
    assert dev_ids.isdisjoint(test_ids)

    fold_map = {r["compound_id"]: int(r["cv_fold"]) for r in folds}
    assert len(folds) == len(fold_map) == 969
    assert set(fold_map) == dev_ids
    assert set(fold_map.values()) == {1, 2, 3, 4, 5}

    group_fold = {}
    for row in folds:
        group = row["group_id"]
        fold = int(row["cv_fold"])
        if group in group_fold:
            assert group_fold[group] == fold
        group_fold[group] = fold

    # Parse target values for development compounds only.
    y_map = {}
    for row in read_csv(ROOT / paths["compounds"]):
        cid = row["compound_id"]
        if cid in dev_ids:
            y_map[cid] = float(row["pic50_molar"])

    assert len(y_map) == 969

    with np.load(ROOT / paths["features"], allow_pickle=False) as data:
        all_ids = data["compound_ids"].tolist()
        all_desc = data["descriptors"]
        all_fp = data["fingerprints"]

    assert len(all_ids) == len(set(all_ids)) == 1236

    index = {cid: i for i, cid in enumerate(all_ids)}
    ids = sorted(dev_ids)

    assert set(ids).issubset(index)

    positions = [index[cid] for cid in ids]

    X_desc = all_desc[positions]
    X_fp = all_fp[positions]
    X_hybrid = np.column_stack((X_desc, X_fp))

    features = {
        "descriptors": X_desc,
        "fingerprints": X_fp,
        "hybrid": X_hybrid,
    }

    y = np.array([y_map[cid] for cid in ids])
    fold_numbers = np.array([fold_map[cid] for cid in ids])

    assert np.isfinite(y).all()

    configurations = protocol["initial_configurations"]
    assert len(configurations) == 13

    # Confirm implementation still matches frozen estimator settings.
    for config in configurations:
        name = config["model"]
        representation = config["representation"]

        if name == "tanimoto_knn":
            estimator = TanimotoKNNRegressor(n_neighbors=5)
            assert representation == "fingerprints"
            assert repr(estimator) == config["estimator"]
        else:
            pipeline = make_model(name, representation)
            assert repr(pipeline.named_steps["model"]) == config["estimator"]
            assert (
                repr(pipeline.named_steps["preprocess"])
                == config["preprocessing"]
            )

    return protocol, features, y, fold_numbers


def run_benchmark(protocol, features, y, fold_numbers):
    """Fit initial models and record identical-fold comparisons."""
    output = RESULTS / "initial_cv_folds.csv"

    if output.exists():
        raise RuntimeError(
            "Benchmark result already exists. Refusing to overwrite."
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'tracking/mlflow.db'}"
    )
    mlflow.set_experiment("mpro-qsar-initial-benchmark")

    results = []

    for config in protocol["initial_configurations"]:
        name = config["model"]
        representation = config["representation"]
        X = features[representation]

        for fold in range(1, 6):
            train = fold_numbers != fold
            val = fold_numbers == fold

            if name == "tanimoto_knn":
                model = TanimotoKNNRegressor(n_neighbors=5)
            else:
                model = make_model(name, representation)

            with mlflow.start_run(
                run_name=f"{name}-{representation}-fold{fold}"
            ) as run:
                mlflow.log_params({
                    "model_family": name,
                    "representation": representation,
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    "split": "butina_070_seed42",
                    "protocol_sha256": sha256(PROTOCOL),
                })

                model.fit(X[train], y[train])
                pred = model.predict(X[val])

                metrics = {
                    "mae": float(
                        mean_absolute_error(y[val], pred)
                    ),
                    "rmse": float(
                        np.sqrt(mean_squared_error(y[val], pred))
                    ),
                    "r2": float(r2_score(y[val], pred)),
                }

                mlflow.log_metrics(metrics)

                results.append({
                    "model": name,
                    "representation": representation,
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    "run_id": run.info.run_id,
                    **metrics,
                })

            print(
                f"{name:16s} {representation:12s} "
                f"fold={fold} MAE={metrics['mae']:.4f}",
                flush=True,
            )

    assert len(results) == 65

    fields = list(results[0])
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = []
    for config in protocol["initial_configurations"]:
        name = config["model"]
        rep = config["representation"]

        subset = [
            r for r in results
            if r["model"] == name
            and r["representation"] == rep
        ]

        assert len(subset) == 5

        summary.append({
            "model": name,
            "representation": rep,
            "mean_mae": mean(r["mae"] for r in subset),
            "std_mae": stdev(r["mae"] for r in subset),
            "mean_rmse": mean(r["rmse"] for r in subset),
            "mean_r2": mean(r["r2"] for r in subset),
        })

    summary.sort(key=lambda r: r["mean_mae"])

    summary_file = RESULTS / "initial_cv_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("\n=== INITIAL BENCHMARK LEADERBOARD ===")
    for i, row in enumerate(summary, start=1):
        print(
            f"{i:2d}. {row['model']:16s} "
            f"{row['representation']:12s} "
            f"MAE={row['mean_mae']:.4f}"
        )

    print("\nSaved:", output)
    print("Saved:", summary_file)
    print("Held-out test evaluation: NOT PERFORMED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("check", "run"), required=True
    )
    args = parser.parse_args()

    protocol, features, y, folds = prepare()

    print("=== FROZEN PROTOCOL PREFLIGHT ===")
    print("Configurations:", len(protocol["initial_configurations"]))
    print("Development compounds:", len(y))
    print("CV folds:", sorted(set(folds)))
    print("Descriptor shape:", features["descriptors"].shape)
    print("Fingerprint shape:", features["fingerprints"].shape)
    print("Hybrid shape:", features["hybrid"].shape)
    print("Checksums and frozen model settings: PASS")
    print("Held-out test targets: NOT USED")

    if args.mode == "check":
        print("\nPASS: Ready for 65 benchmark fits.")
    else:
        run_benchmark(protocol, features, y, folds)


if __name__ == "__main__":
    main()
