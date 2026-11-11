"""Select and freeze the final model using development results only.

Compare the 13 original configurations and the two best
Optuna trial results. The model with the lowest development
CV MAE is selected without evaluating the held-out test set.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/final_model_selection.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


benchmark_path = ROOT / "results/experiment_protocol.json"
tuning_path = ROOT / "results/tuning_protocol.json"
summary_path = ROOT / "results/benchmarks/initial_cv_summary.csv"
verification_path = ROOT / "results/tuning/tuning_verification.json"

benchmark = load_json(benchmark_path)
tuning = load_json(tuning_path)
verification = load_json(verification_path)

# Verify that tuning used the original frozen protocol.
assert verification["tuning_protocol_sha256"] == sha256(tuning_path)
assert verification["held_out_test_evaluated"] is False

# Collect the 13 original benchmark configurations.
candidates = []

for row in load_csv(summary_path):
    candidates.append({
        "model": row["model"],
        "representation": row["representation"],
        "source": "initial",
        "mean_cv_mae": float(row["mean_mae"]),
    })

assert len(candidates) == 13

# Add the best trial from each preselected tuning finalist.
for study in verification["studies"]:
    assert study["completed_trials"] == 20
    assert study["optuna_csv_mlflow_consistency"] == "PASS"

    candidates.append({
        "model": study["model"],
        "representation": study["representation"],
        "source": "tuned",
        "mean_cv_mae": study["best_tuned_cv_mae"],
        "parameters": study["best_parameters"],
    })

assert len(candidates) == 15

# MAE is the frozen primary selection criterion.
winner = min(candidates, key=lambda item: item["mean_cv_mae"])

assert winner["model"] == "svr"
assert winner["representation"] == "hybrid"
assert winner["source"] == "initial"

# Retrieve the exact original configuration.
original = next(
    config
    for config in benchmark["initial_configurations"]
    if config["model"] == winner["model"]
    and config["representation"] == winner["representation"]
)

selection = {
    "status": "frozen_before_held_out_evaluation",
    "selection_metric": "mean_five_fold_development_cv_mae",
    "candidates_compared": len(candidates),
    "winning_model": winner,
    "estimator": original["estimator"],
    "preprocessing": original["preprocessing"],
    "training_compounds": 969,
    "held_out_test_compounds": 267,
    "train_on": "all_development_compounds",
    "test_used_for_model_selection": False,
    "input_sha256": {
        "benchmark_protocol": sha256(benchmark_path),
        "tuning_protocol": sha256(tuning_path),
        "initial_summary": sha256(summary_path),
        "tuning_verification": sha256(verification_path),
    },
}

if OUTPUT.exists():
    assert load_json(OUTPUT) == selection
    print("Existing final selection verified.")
else:
    OUTPUT.write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Final model selection frozen.")

print("\n=== SELECTED MODEL ===")
print("Model:", winner["model"])
print("Features:", winner["representation"])
print("Source:", winner["source"])
print("Development CV MAE:", round(winner["mean_cv_mae"], 4))
print("Estimator:", original["estimator"])
print("Training compounds:", selection["training_compounds"])
print("Reserved test compounds:", selection["held_out_test_compounds"])
print("Held-out test evaluation: NOT PERFORMED")
print("\nPASS: Final model selected reproducibly.")
