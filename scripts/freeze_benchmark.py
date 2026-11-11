"""Freeze and verify the initial QSAR benchmark configuration."""

import csv
import hashlib
import json
from pathlib import Path

import sklearn
import xgboost
from rdkit import rdBase

from sarscov2_qsar.models import (
    MODEL_NAMES,
    REPRESENTATIONS,
    make_model,
)
from sarscov2_qsar.tanimoto_knn import TanimotoKNNRegressor

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "results/experiment_protocol.json"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = {
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

hashes = {
    name: checksum(ROOT / relative_path)
    for name, relative_path in files.items()
}

splits = read_csv(ROOT / files["outer_split"])
folds = read_csv(ROOT / files["cv_folds"])

development = {
    row["compound_id"]
    for row in splits if row["split"] == "train"
}
test = {
    row["compound_id"]
    for row in splits if row["split"] == "test"
}

assert len(development) == 969
assert len(test) == 267
assert development.isdisjoint(test)
assert len(splits) == 1236

assert len(folds) == 969
assert {r["compound_id"] for r in folds} == development
assert {int(r["cv_fold"]) for r in folds} == set(range(1, 6))

group_to_fold = {}

for row in folds:
    group = row["group_id"]
    fold = int(row["cv_fold"])

    if group in group_to_fold:
        assert group_to_fold[group] == fold
    else:
        group_to_fold[group] = fold

configurations = []

for model_name in MODEL_NAMES:
    for representation in REPRESENTATIONS:
        pipeline = make_model(model_name, representation)

        configurations.append({
            "model": model_name,
            "representation": representation,
            "estimator": repr(pipeline.named_steps["model"]),
            "preprocessing": repr(
                pipeline.named_steps["preprocess"]
            ),
        })

configurations.append({
    "model": "tanimoto_knn",
    "representation": "fingerprints",
    "estimator": repr(TanimotoKNNRegressor(n_neighbors=5)),
    "preprocessing": "none",
})

assert len(configurations) == 13

protocol = {
    "status": "frozen_initial_benchmark",
    "task": "SARS-CoV-2 Mpro quantitative pIC50 regression",
    "grouping": "Butina Tanimoto 0.70",
    "n_development": 969,
    "n_test": 267,
    "n_cv_folds": 5,
    "primary_metric": "mean_fold_mae",
    "secondary_metrics": [
        "mean_fold_rmse",
        "mean_fold_r2",
        "fold_metric_std",
    ],
    "reference_baseline_mean_mae": 0.7889,
    "model_selection": "development_cv_only",
    "held_out_test_evaluation": "after_final_model_freeze",
    "initial_configurations": configurations,
    "initial_model_fits": 65,
    "software_versions": {
        "rdkit": rdBase.rdkitVersion,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    },
    "input_and_code_sha256": hashes,
    "tuning": {
        "status": "not_yet_specified",
        "rule": "freeze search space and budget before tuning",
    },
}

output.parent.mkdir(parents=True, exist_ok=True)

if output.exists():
    existing = json.loads(output.read_text(encoding="utf-8"))
    if existing != protocol:
        raise RuntimeError(
            "Existing protocol differs. Refusing to overwrite it."
        )
    print("Existing frozen protocol verified.")
else:
    output.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print("New benchmark protocol frozen.")

print("Model configurations:", len(configurations))
print("CV folds:", protocol["n_cv_folds"])
print("Planned model fits:", protocol["initial_model_fits"])
print("Input/code checksums:", len(hashes))
print("Held-out test evaluation: prohibited during selection")
print("Saved:", output)
print("\nPASS: Initial benchmark protocol verified.")
