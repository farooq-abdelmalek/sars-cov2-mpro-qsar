"""Create scientific figures from archived QSAR results.

No model training, tuning, or new predictions are performed.

Sources:
    results/final_test_predictions.csv
    results/final_test_metrics.json
    results/benchmarks/initial_cv_summary.csv
    results/diagnostics/heldout_error_profile.json

Outputs:
    docs/figures/*.png
    docs/figures/*.svg
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs/figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Consistent, readable style across the three figures.
BLUE = "#246B91"
TEAL = "#168474"
ORANGE = "#C77B37"
GREY = "#AAB5C0"
DARK = "#253445"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#B4BEC8",
    "grid.alpha": 0.18,
    "savefig.facecolor": "white",
    "svg.hashsalt": "mpro-qsar-results-v1",
})


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, name):
    """Save a figure as high-resolution PNG and vector SVG."""
    fig.tight_layout()

    png = OUTPUT / f"{name}.png"
    svg = OUTPUT / f"{name}.svg"

    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Date": None},
    )

    # Matplotlib sometimes adds trailing spaces to SVG lines.
    # Remove them so regenerated figures pass Git checks.
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text(
        "\n".join(line.rstrip(" \t") for line in lines) + "\n",
        encoding="utf-8",
    )

    plt.close(fig)

    print(f"Saved: {png.relative_to(ROOT)}")
    print(f"Saved: {svg.relative_to(ROOT)}")


# -------------------------------------------------------
# Figure 1: Observed vs. predicted pIC50
# -------------------------------------------------------

predictions = read_csv(
    RESULTS / "final_test_predictions.csv"
)
metrics = read_json(
    RESULTS / "final_test_metrics.json"
)

assert len(predictions) == 267
assert len({r["compound_id"] for r in predictions}) == 267
assert metrics["n_test_compounds"] == 267

observed = np.array([
    float(row["true_pic50"])
    for row in predictions
])

predicted = np.array([
    float(row["predicted_pic50"])
    for row in predictions
])

assert np.isfinite(observed).all()
assert np.isfinite(predicted).all()

fig, ax = plt.subplots(figsize=(7.2, 6.5))

ax.scatter(
    observed,
    predicted,
    s=29,
    alpha=0.68,
    color=BLUE,
    edgecolors="white",
    linewidths=0.35,
)

# The dashed diagonal represents perfect prediction.
lower = min(observed.min(), predicted.min()) - 0.2
upper = max(observed.max(), predicted.max()) + 0.2

ax.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    linewidth=1.5,
    color=ORANGE,
    label="Perfect prediction",
)

ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Observed pIC50")
ax.set_ylabel("Predicted pIC50")
ax.set_title("Held-out inhibitor potency prediction")
ax.grid(alpha=0.16)
ax.legend(loc="lower right", frameon=False)

summary = (
    f"N = 267\n"
    f"MAE = {metrics['mae']:.3f}\n"
    f"RMSE = {metrics['rmse']:.3f}\n"
    f"R² = {metrics['r2']:.3f}"
)

ax.text(
    0.04,
    0.96,
    summary,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    color=DARK,
    bbox={
        "facecolor": "white",
        "edgecolor": "#DFE5EA",
        "boxstyle": "round,pad=0.6",
        "alpha": 0.95,
    },
)

save_figure(fig, "01_observed_vs_predicted")


# -------------------------------------------------------
# Figure 2: Initial model benchmarking
# -------------------------------------------------------

benchmark = read_csv(
    RESULTS / "benchmarks/initial_cv_summary.csv"
)

assert len(benchmark) == 13

benchmark.sort(key=lambda row: float(row["mean_mae"]))

model_names = {
    "svr": "SVR",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "tanimoto_knn": "Tanimoto kNN",
}

feature_names = {
    "hybrid": "Hybrid",
    "fingerprints": "Morgan",
    "descriptors": "RDKit",
}

labels = [
    f"{model_names[row['model']]} / "
    f"{feature_names[row['representation']]}"
    for row in benchmark
]

scores = [
    float(row["mean_mae"])
    for row in benchmark
]

selected = [
    row["model"] == "svr"
    and row["representation"] == "hybrid"
    for row in benchmark
]

assert sum(selected) == 1
assert selected[0], "The frozen winner must rank first."

colors = [
    TEAL if is_selected else GREY
    for is_selected in selected
]

fig, ax = plt.subplots(figsize=(10, 7.2))

positions = np.arange(len(labels))

ax.barh(
    positions,
    scores,
    color=colors,
    height=0.72,
)

ax.set_yticks(positions, labels)
ax.invert_yaxis()

ax.set_xlim(0, max(scores) * 1.16)

ax.set_xlabel("Mean five-fold development CV MAE (pIC50)")
ax.set_title("Benchmarking molecular representations and models")

ax.grid(axis="x", alpha=0.16)
ax.set_axisbelow(True)

for i, value in enumerate(scores):
    ax.text(
        value + 0.008,
        i,
        f"{value:.3f}",
        va="center",
        fontsize=9.5,
        color=DARK,
        weight="bold" if selected[i] else "normal",
    )

ax.text(
    0.01,
    -0.10,
    "Lower MAE is better. "
    "Green indicates the frozen selected configuration.",
    transform=ax.transAxes,
    fontsize=9.5,
    color="#586674",
)

save_figure(fig, "02_model_benchmark")


# -------------------------------------------------------
# Figure 3: Prediction errors by observed potency
# -------------------------------------------------------

diagnostics = read_json(
    RESULTS / "diagnostics/heldout_error_profile.json"
)

quartiles = diagnostics["potency_quartiles"]

assert len(quartiles) == 4
assert sum(q["n"] for q in quartiles) == 267
assert diagnostics["primary_test_evaluation_unchanged"]

quartile_labels = [
    "Q1\nWeakest",
    "Q2",
    "Q3",
    "Q4\nMost potent",
]

mae = np.array([
    q["mae"] for q in quartiles
])

bias = np.array([
    q["mean_signed_error"] for q in quartiles
])

counts = [
    q["n"] for q in quartiles
]

fig, ax = plt.subplots(figsize=(9, 5.6))

x = np.arange(4)
width = 0.34

bars_mae = ax.bar(
    x - width / 2,
    mae,
    width,
    label="MAE",
    color=BLUE,
)

bars_bias = ax.bar(
    x + width / 2,
    bias,
    width,
    label="Mean signed error",
    color=ORANGE,
)

ax.axhline(0, color=DARK, linewidth=0.9)

ax.set_xticks(x, quartile_labels)
ax.set_ylabel("Prediction error (pIC50)")
ax.set_title("Prediction errors across potency ranges")

ax.set_ylim(-0.95, 1.05)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.16)

for bar, value in zip(bars_mae, mae):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.035,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

for bar, value in zip(bars_bias, bias):
    offset = 0.035 if value >= 0 else -0.045
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.3f}",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=10,
    )

for i, n in enumerate(counts):
    ax.text(
        i,
        0.94,
        f"n={n}",
        ha="center",
        fontsize=9,
        color="#586674",
    )

ax.legend(frameon=False, loc="lower center", ncol=2)

fig.text(
    0.5,
    0.005,
    "Positive bias: predicted potency exceeds observed potency. "
    "Quartiles are defined by observed pIC50.",
    ha="center",
    fontsize=9,
    color="#586674",
)

save_figure(fig, "03_errors_by_potency")


print("\nPASS: Three scientific figures generated.")
print("Source: archived research results only.")
print("No models trained or predictions regenerated.")
