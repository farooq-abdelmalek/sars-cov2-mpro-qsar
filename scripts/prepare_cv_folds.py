import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]

compound_file = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
split_file = ROOT / "data/splits/butina_070_seed42.csv"

output = ROOT / "data/splits/development_cv5.csv"
report_path = ROOT / "results/data_qualification/cv_fold_summary.json"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


compounds = read_csv(compound_file)
assignments = read_csv(split_file)

compound_ids = [r["compound_id"] for r in compounds]
split_ids = [r["compound_id"] for r in assignments]

assert len(compound_ids) == len(set(compound_ids)) == 1236
assert len(split_ids) == len(set(split_ids)) == 1236
assert set(compound_ids) == set(split_ids)

development = sorted(
    [r for r in assignments if r["split"] == "train"],
    key=lambda r: r["compound_id"],
)

test = [r for r in assignments if r["split"] == "test"]

assert len(development) == 969
assert len(test) == 267

dev_groups = {r["group_id"] for r in development}
test_groups = {r["group_id"] for r in test}

assert dev_groups.isdisjoint(test_groups)
assert len(dev_groups) == 282
assert len(test_groups) == 136

# CV uses only development-set groups.
groups = [r["group_id"] for r in development]
X = np.zeros((len(development), 1))

cv = GroupKFold(n_splits=5)

fold_assignments = []
summary = []

for fold_number, (train_idx, val_idx) in enumerate(
    cv.split(X, groups=groups), start=1
):
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}

    assert train_groups.isdisjoint(val_groups)

    for i in val_idx:
        fold_assignments.append({
            "compound_id": development[i]["compound_id"],
            "group_id": development[i]["group_id"],
            "cv_fold": fold_number,
        })

    fold_report = {
        "fold": fold_number,
        "training_compounds": len(train_idx),
        "validation_compounds": len(val_idx),
        "training_groups": len(train_groups),
        "validation_groups": len(val_groups),
    }

    summary.append(fold_report)
    print(f"Fold {fold_number}: {fold_report}")

# Every development compound must appear exactly once in validation.
fold_ids = [r["compound_id"] for r in fold_assignments]

assert len(fold_ids) == 969
assert len(set(fold_ids)) == 969
assert set(fold_ids) == {
    r["compound_id"] for r in development
}

# Each chemical group must belong to exactly one validation fold.
group_folds = {}

for row in fold_assignments:
    group = row["group_id"]
    fold = row["cv_fold"]

    if group in group_folds:
        assert group_folds[group] == fold
    else:
        group_folds[group] = fold

assert len(group_folds) == 282

fold_assignments.sort(key=lambda r: r["compound_id"])

report = {
    "scikit_learn_version": sklearn.__version__,
    "method": "GroupKFold",
    "n_splits": 5,
    "development_compounds": 969,
    "held_out_test_compounds": 267,
    "source_compound_sha256": checksum(compound_file),
    "source_split_sha256": checksum(split_file),
    "folds": summary,
    "group_leakage_check": "PASS",
}

output.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

# Never silently replace previously saved fold assignments.
if output.exists():
    existing = read_csv(output)
    expected = [
        {key: str(value) for key, value in row.items()}
        for row in fold_assignments
    ]
    if existing != expected:
        raise RuntimeError(
            "Existing CV assignments differ. No files overwritten."
        )

else:
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fold_assignments[0].keys()
        )
        writer.writeheader()
        writer.writerows(fold_assignments)

report_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("\nPASS: Five group-aware folds verified.")
print("Development compounds:", len(fold_assignments))
print("Held-out test compounds:", len(test))
print("Saved:", output)
