"""Fit the frozen SVR on all development compounds.

The model was selected using group-aware cross-validation.
This script trains on the 969 development compounds only.

Outputs:
    artifacts/final_svr_hybrid_dev969.joblib
    results/final_training_manifest.json

The held-out test set is not evaluated here.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from benchmark_models import prepare
from sarscov2_qsar.models import make_model


ROOT = Path(__file__).resolve().parents[1]
selection_file = ROOT / "results/final_model_selection.json"
model_file = ROOT / "artifacts/final_svr_hybrid_dev969.joblib"
manifest_file = ROOT / "results/final_training_manifest.json"


def sha256(path):
    """Calculate a file checksum for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Prevent accidental replacement of a trained model.
if model_file.exists() or manifest_file.exists():
    raise RuntimeError(
        "Final training artifacts already exist. "
        "Refusing to overwrite them."
    )

selection = json.loads(selection_file.read_text(encoding="utf-8"))
winner = selection["winning_model"]

assert selection["status"] == "frozen_before_held_out_evaluation"
assert winner["model"] == "svr"
assert winner["representation"] == "hybrid"
assert winner["source"] == "initial"
assert selection["training_compounds"] == 969
assert selection["held_out_test_compounds"] == 267

# Reuse the benchmark's frozen data and checksum checks.
protocol, features, targets, folds = prepare()

X_train = features["hybrid"]
y_train = targets

assert X_train.shape == (969, 2058)
assert y_train.shape == (969,)
assert np.isfinite(X_train).all()
assert np.isfinite(y_train).all()

# Construct exactly the pipeline selected before test evaluation.
model = make_model("svr", "hybrid")

assert repr(model.named_steps["model"]) == selection["estimator"]
assert repr(model.named_steps["preprocess"]) == selection["preprocessing"]

print("=== FINAL MODEL TRAINING ===")
print("Model: SVR + Hybrid")
print("Training compounds:", len(y_train))
print("Input features:", X_train.shape[1])
print("Development CV MAE:", winner["mean_cv_mae"])
print("Held-out test targets: NOT USED")

# Fit the scaler and SVR using all development observations.
model.fit(X_train, y_train)

scaler = model.named_steps["preprocess"].named_transformers_[
    "descriptors"
]

assert int(scaler.n_samples_seen_) == 969

# This checks prediction functionality, NOT generalization.
sample_predictions = model.predict(X_train[:5])
assert sample_predictions.shape == (5,)
assert np.isfinite(sample_predictions).all()

# Save the fitted pipeline, including its descriptor scaler.
model_file.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_file)

# Only reload a model artifact produced by our own script.
reloaded_model = joblib.load(model_file)
reloaded_predictions = reloaded_model.predict(X_train[:5])

assert np.allclose(sample_predictions, reloaded_predictions)

manifest = {
    "model": "SVR + Hybrid",
    "training_compounds": 969,
    "n_features": 2058,
    "fitted_descriptor_scaler_samples": 969,
    "selected_model_cv_mae": winner["mean_cv_mae"],
    "selection_sha256": sha256(selection_file),
    "model_sha256": sha256(model_file),
    "sklearn_version": sklearn.__version__,
    "test_evaluated": False,
}

manifest_file.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("\nPASS: Final model fitted on 969 development compounds.")
print("PASS: Saved model predictions reproduced after loading.")
print("PASS: Descriptor scaler fitted on development data only.")
print("Held-out test evaluation: NOT PERFORMED")
print("Saved:", model_file)
print("Saved:", manifest_file)
