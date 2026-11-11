"""Generate deterministic molecular features without fitting models."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase

from sarscov2_qsar.features import (
    DESCRIPTORS,
    MORGAN_SIZE,
    featurize,
)

ROOT = Path(__file__).resolve().parents[1]

source = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
split_file = ROOT / "data/splits/butina_070_seed42.csv"
output_dir = ROOT / "data/features"
output_dir.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


compounds = sorted(
    read_csv(source),
    key=lambda r: r["compound_id"],
)

splits = read_csv(split_file)

ids = [r["compound_id"] for r in compounds]
split_ids = [r["compound_id"] for r in splits]

assert len(ids) == len(set(ids)) == 1236
assert set(ids) == set(split_ids)

descriptors = []
fingerprints = []

for row in compounds:
    desc, fp = featurize(row["connectivity"])
    descriptors.append(desc)
    fingerprints.append(fp)

X_desc = np.asarray(descriptors, dtype=np.float64)
X_fp = np.asarray(fingerprints, dtype=np.uint8)

assert X_desc.shape == (1236, 10)
assert X_fp.shape == (1236, MORGAN_SIZE)
assert np.isfinite(X_desc).all()
assert np.all((X_fp == 0) | (X_fp == 1))

# Verify that equivalent molecular connectivity produces
# identical representations.
seen = {}

for index, row in enumerate(compounds):
    connectivity = row["connectivity"]

    if connectivity in seen:
        previous = seen[connectivity]
        assert np.array_equal(
            X_desc[index], X_desc[previous]
        )
        assert np.array_equal(
            X_fp[index], X_fp[previous]
        )
    else:
        seen[connectivity] = index

output = output_dir / "moonshot_features.npz"

np.savez_compressed(
    output,
    compound_ids=np.array(ids),
    descriptor_names=np.array(list(DESCRIPTORS)),
    descriptors=X_desc,
    fingerprints=X_fp,
)

# Independently verify the saved artifact.
with np.load(output, allow_pickle=False) as saved:
    assert saved["descriptors"].shape == (1236, 10)
    assert saved["fingerprints"].shape == (1236, 2048)
    assert saved["compound_ids"].tolist() == ids
    assert saved["descriptor_names"].tolist() == list(DESCRIPTORS)

manifest = {
    "source_sha256": sha256(source),
    "split_sha256": sha256(split_file),
    "feature_artifact_sha256": sha256(output),
    "rdkit_version": rdBase.rdkitVersion,
    "n_compounds": len(ids),
    "n_unique_connectivities": len(seen),
    "descriptor_names": list(DESCRIPTORS),
    "fingerprint": {
        "type": "Morgan binary",
        "radius": 2,
        "size": MORGAN_SIZE,
        "include_chirality": False,
    },
    "molecular_input": "connectivity",
    "learned_preprocessing": False,
}

manifest_file = output_dir / "feature_manifest.json"
manifest_file.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("=== FEATURE GENERATION ===")
print("Compounds:", len(ids))
print("Unique connectivities:", len(seen))
print("Descriptors:", X_desc.shape)
print("Morgan fingerprints:", X_fp.shape)
print("All descriptors finite:", np.isfinite(X_desc).all())
print("Binary fingerprint check: PASS")
print("Repeated connectivity consistency: PASS")
print("Saved artifact verification: PASS")
print("RDKit version:", rdBase.rdkitVersion)
print("\nSaved:", output)
print("Saved:", manifest_file)
