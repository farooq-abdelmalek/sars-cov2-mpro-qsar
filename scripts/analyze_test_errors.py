"""Analyze saved held-out prediction errors.

This is descriptive, post-evaluation analysis.
It does not retrain models, generate predictions, or
change the previously frozen model-selection decision.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/final_test_predictions.csv"
METRICS = ROOT / "results/final_test_metrics.json"
OUTPUT = ROOT / "results/diagnostics/heldout_error_profile.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with PREDICTIONS.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

metrics = json.loads(METRICS.read_text(encoding="utf-8"))

assert len(rows) == 267
assert len({row["compound_id"] for row in rows}) == 267

y_true = np.array([float(r["true_pic50"]) for r in rows])
y_pred = np.array([float(r["predicted_pic50"]) for r in rows])

errors = y_pred - y_true
absolute_errors = np.abs(errors)

assert np.isfinite(y_true).all()
assert np.isfinite(y_pred).all()

# Verify against the already archived primary evaluation.
assert np.isclose(absolute_errors.mean(), metrics["mae"])
assert np.isclose(
    np.sqrt(np.mean(errors ** 2)), metrics["rmse"]
)
assert np.isclose(
    np.mean(errors), metrics["mean_signed_error"]
)

for i, row in enumerate(rows):
    assert np.isclose(
        float(row["signed_error_pic50"]), errors[i]
    )
    assert np.isclose(
        float(row["absolute_error_pic50"]), absolute_errors[i]
    )


def summarize(indices):
    """Calculate descriptive statistics for selected compounds."""
    actual = y_true[indices]
    predicted = y_pred[indices]
    signed = errors[indices]
    absolute = absolute_errors[indices]

    return {
        "n": len(indices),
        "observed_pic50_min": float(np.min(actual)),
        "observed_pic50_max": float(np.max(actual)),
        "observed_pic50_mean": float(np.mean(actual)),
        "predicted_pic50_mean": float(np.mean(predicted)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(signed ** 2))),
        "mean_signed_error": float(np.mean(signed)),
        "median_absolute_error": float(np.median(absolute)),
        "fraction_overpredicted": float(np.mean(signed > 0)),
        "fraction_absolute_error_above_1": float(
            np.mean(absolute > 1)
        ),
    }


# Divide compounds into four equal-sized groups by their
# experimentally observed pIC50, lowest to highest.
#
# These quartiles are exploratory: they were not used
# for training, model selection, or hyperparameter tuning.
order = np.argsort(y_true, kind="stable")
quartile_indices = np.array_split(order, 4)

labels = [
    "Q1_lowest_potency",
    "Q2_lower_middle",
    "Q3_upper_middle",
    "Q4_highest_potency",
]

quartiles = []

for label, indices in zip(labels, quartile_indices):
    result = {"group": label, **summarize(indices)}
    quartiles.append(result)

# Inspect the largest errors without changing the model.
worst_indices = np.argsort(-absolute_errors)[:10]

worst_compounds = [
    {
        "compound_id": rows[i]["compound_id"],
        "chemical_group": rows[i]["group_id"],
        "observed_pic50": float(y_true[i]),
        "predicted_pic50": float(y_pred[i]),
        "signed_error": float(errors[i]),
        "absolute_error": float(absolute_errors[i]),
    }
    for i in worst_indices
]

overall = summarize(np.arange(len(rows)))

report = {
    "analysis": "descriptive_heldout_error_analysis",
    "primary_test_evaluation_unchanged": True,
    "source_predictions_sha256": sha256(PREDICTIONS),
    "source_metrics_sha256": sha256(METRICS),
    "overall": overall,
    "potency_quartiles": quartiles,
    "ten_largest_absolute_errors": worst_compounds,
}

# Preserve existing diagnostics rather than replacing them
# with potentially different exploratory results.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
content = json.dumps(report, indent=2) + "\n"

if OUTPUT.exists():
    assert OUTPUT.read_text(encoding="utf-8") == content
else:
    OUTPUT.write_text(content, encoding="utf-8")


print("=== OVERALL ERROR PROFILE ===")
print("Test compounds:", overall["n"])
print(f"MAE: {overall['mae']:.4f}")
print(
    "Mean signed error:",
    f"{overall['mean_signed_error']:+.4f}"
)
print(
    "Absolute error > 1 pIC50:",
    f"{overall['fraction_absolute_error_above_1']:.1%}"
)

print("\n=== ERRORS BY OBSERVED POTENCY ===")
print(
    f"{'Group':<24} {'N':>4} "
    f"{'MAE':>8} {'Bias':>8} {'>1 error':>10}"
)

for q in quartiles:
    print(
        f"{q['group']:<24} "
        f"{q['n']:>4} "
        f"{q['mae']:>8.4f} "
        f"{q['mean_signed_error']:>+8.4f} "
        f"{q['fraction_absolute_error_above_1']:>9.1%}"
    )

print("\n=== FIVE LARGEST ERRORS ===")
for item in worst_compounds[:5]:
    print(
        item["compound_id"],
        f"observed={item['observed_pic50']:.3f}",
        f"predicted={item['predicted_pic50']:.3f}",
        f"error={item['signed_error']:+.3f}",
    )

print("\nPASS: Saved predictions analyzed.")
print("No models retrained or predictions regenerated.")
print("Saved:", OUTPUT)
