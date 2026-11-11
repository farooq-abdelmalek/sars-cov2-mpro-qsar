"""Select tuning finalists from the frozen initial benchmark.

The two best distinct model families are selected using
mean development CV MAE. The held-out test set is not used.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/benchmarks"

fold_file = RESULTS / "initial_cv_folds.csv"
summary_file = RESULTS / "initial_cv_summary.csv"
benchmark_protocol = ROOT / "results/experiment_protocol.json"
tuning_protocol = ROOT / "results/tuning_protocol.json"
output_file = RESULTS / "selected_finalists.json"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


benchmark = json.loads(benchmark_protocol.read_text())
tuning = json.loads(tuning_protocol.read_text())

fold_rows = read_csv(fold_file)
summary_rows = read_csv(summary_file)

# Confirm that the results cover every frozen configuration.
expected = {
    (row["model"], row["representation"])
    for row in benchmark["initial_configurations"]
}

observed = defaultdict(list)

for row in fold_rows:
    key = (row["model"], row["representation"])
    fold = int(row["fold"])
    score = float(row["mae"])

    assert key in expected
    assert math.isfinite(score)
    assert fold in range(1, 6)

    observed[key].append((fold, score))

assert len(fold_rows) == 65
assert set(observed) == expected

for key, entries in observed.items():
    assert sorted(fold for fold, _ in entries) == [1, 2, 3, 4, 5]

# Independently recompute the leaderboard's mean MAE.
summary_scores = {}

for row in summary_rows:
    key = (row["model"], row["representation"])
    assert key not in summary_scores
    summary_scores[key] = float(row["mean_mae"])

assert set(summary_scores) == expected

for key, entries in observed.items():
    calculated = mean(score for _, score in entries)
    assert math.isclose(
        calculated,
        summary_scores[key],
        rel_tol=0,
        abs_tol=1e-10,
    ), f"Summary mismatch: {key}"

# Rank the best representation of each model family.
# Tie-breaking follows the predeclared family order.
family_order = [
    "ridge", "svr", "random_forest",
    "xgboost", "tanimoto_knn"
]

best_by_family = {}

for (family, representation), score in summary_scores.items():
    candidate = {
        "model": family,
        "representation": representation,
        "initial_mean_cv_mae": score,
    }

    current = best_by_family.get(family)

    if current is None or score < current["initial_mean_cv_mae"]:
        best_by_family[family] = candidate

ranked = sorted(
    best_by_family.values(),
    key=lambda row: (
        row["initial_mean_cv_mae"],
        family_order.index(row["model"]),
    ),
)

n_finalists = tuning["finalist_selection"]["number_of_families"]
assert n_finalists == 2

finalists = ranked[:n_finalists]

selection = {
    "status": "selected_before_hyperparameter_tuning",
    "selection_rule": "top_two_distinct_model_families",
    "metric": "mean_development_cv_mae",
    "finalists": finalists,
    "benchmark_protocol_sha256": checksum(benchmark_protocol),
    "tuning_protocol_sha256": checksum(tuning_protocol),
    "fold_results_sha256": checksum(fold_file),
    "summary_results_sha256": checksum(summary_file),
    "held_out_test_used": False,
}

if output_file.exists():
    existing = json.loads(output_file.read_text())
    assert existing == selection, "Existing selection differs."
else:
    output_file.write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )

print("=== VERIFIED FINALISTS ===")

for rank, candidate in enumerate(finalists, start=1):
    print(
        f"{rank}. {candidate['model']} | "
        f"{candidate['representation']} | "
        f"CV MAE={candidate['initial_mean_cv_mae']:.4f}"
    )

print("\nInitial configurations verified:", len(expected))
print("Fold results verified:", len(fold_rows))
print("Held-out test used: NO")
print("Saved:", output_file)
print("\nPASS: Finalists selected reproducibly.")
