"""Five-fold, group-aware median baseline on development data."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
compound_path = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
split_path = ROOT / "data/splits/butina_070_seed42.csv"
fold_path = ROOT / "data/splits/development_cv5.csv"
output = ROOT / "results/baselines/median_cv5.json"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


compounds = read_csv(compound_path)
splits = read_csv(split_path)
folds = read_csv(fold_path)

development_ids = {
    row["compound_id"]
    for row in splits if row["split"] == "train"
}

fold_map = {
    row["compound_id"]: int(row["cv_fold"])
    for row in folds
}

assert len(development_ids) == 969
assert len(folds) == len(fold_map) == 969
assert set(fold_map) == development_ids
assert sorted(set(fold_map.values())) == [1, 2, 3, 4, 5]

# Construct targets strictly from development compounds.
development = sorted(
    [
        row for row in compounds
        if row["compound_id"] in development_ids
    ],
    key=lambda row: row["compound_id"],
)

assert len(development) == 969

y = np.array(
    [float(row["pic50_molar"]) for row in development]
)
fold_numbers = np.array(
    [fold_map[row["compound_id"]] for row in development]
)

assert np.isfinite(y).all()

predictions = np.full(len(y), np.nan)
fold_results = []

for fold in range(1, 6):
    train_mask = fold_numbers != fold
    val_mask = fold_numbers == fold

    # DummyRegressor ignores molecular features.
    X_train = np.zeros((train_mask.sum(), 1))
    X_val = np.zeros((val_mask.sum(), 1))

    model = DummyRegressor(strategy="median")
    model.fit(X_train, y[train_mask])

    y_val = y[val_mask]
    y_pred = model.predict(X_val)
    predictions[val_mask] = y_pred

    result = {
        "fold": fold,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "training_median_pic50": float(model.constant_[0, 0]),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_val, y_pred))
        ),
        "r2": float(r2_score(y_val, y_pred)),
    }

    fold_results.append(result)

    print(
        f"Fold {fold}: "
        f"n_val={result['n_validation']}, "
        f"MAE={result['mae']:.4f}, "
        f"RMSE={result['rmse']:.4f}, "
        f"R2={result['r2']:.4f}"
    )

assert np.isfinite(predictions).all()

summary = {
    "model": "DummyRegressor(strategy='median')",
    "evaluation": "development-only group-aware 5-fold CV",
    "primary_metric": "MAE in pIC50 units",
    "n_development": len(y),
    "compound_sha256": sha256(compound_path),
    "split_sha256": sha256(split_path),
    "fold_sha256": sha256(fold_path),
    "folds": fold_results,
    "mean_fold_mae": float(
        np.mean([r["mae"] for r in fold_results])
    ),
    "std_fold_mae": float(
        np.std([r["mae"] for r in fold_results], ddof=1)
    ),
    "pooled_oof_mae": float(
        mean_absolute_error(y, predictions)
    ),
    "pooled_oof_rmse": float(
        np.sqrt(mean_squared_error(y, predictions))
    ),
    "pooled_oof_r2": float(
        r2_score(y, predictions)
    ),
}

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

print("\n=== MEDIAN BASELINE SUMMARY ===")
print(f"Mean fold MAE: {summary['mean_fold_mae']:.4f}")
print(f"Std fold MAE:  {summary['std_fold_mae']:.4f}")
print(f"Pooled OOF MAE: {summary['pooled_oof_mae']:.4f}")
print(f"Pooled OOF RMSE: {summary['pooled_oof_rmse']:.4f}")
print(f"Pooled OOF R2: {summary['pooled_oof_r2']:.4f}")
print("\nPASS: Development-only baseline completed.")
print("Saved:", output)
