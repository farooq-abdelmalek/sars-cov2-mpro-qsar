import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
output = ROOT / "data/splits/butina_070_seed42.csv"
report_path = ROOT / "results/data_qualification/split_summary.json"

with source.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 1236
assert len({r["compound_id"] for r in rows}) == len(rows)

by_connectivity = defaultdict(list)
for row in rows:
    by_connectivity[row["connectivity"]].append(row)

keys = sorted(by_connectivity)
mols = [Chem.MolFromSmiles(s) for s in keys]
assert all(m is not None for m in mols)

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False
)
fps = [generator.GetFingerprint(m) for m in mols]

distances = []
for i in range(1, len(fps)):
    similarities = DataStructs.BulkTanimotoSimilarity(
        fps[i], fps[:i]
    )
    distances.extend(1 - s for s in similarities)

clusters = Butina.ClusterData(
    distances,
    len(keys),
    distThresh=1 - 0.70,
    isDistData=True,
    reordering=True,
)

groups = {}
for cluster_number, cluster in enumerate(clusters):
    name = f"B{cluster_number:04d}"
    groups[name] = [
        row
        for index in cluster
        for row in by_connectivity[keys[index]]
    ]

assert len(groups) == 418, f"Found {len(groups)} groups"

# Reproduce the allocation from the feasibility audit.
# Reproduce the feasibility audit's string ordering.
group_names = sorted(groups, key=lambda x: str(int(x[1:])))

random.Random(42).shuffle(group_names)

test_ids = set()
target = round(len(rows) * 0.20)

for group_name in group_names:
    test_ids.add(group_name)
    if sum(len(groups[g]) for g in test_ids) >= target:
        break

assignments = []
for name, compounds in groups.items():
    split = "test" if name in test_ids else "train"
    for row in compounds:
        assignments.append({
            "compound_id": row["compound_id"],
            "split": split,
            "group_id": name,
            "connectivity": row["connectivity"],
        })

assignments.sort(key=lambda r: r["compound_id"])

assert len(assignments) == 1236
assert len({r["compound_id"] for r in assignments}) == 1236

train = {r["compound_id"] for r in assignments if r["split"] == "train"}
test = {r["compound_id"] for r in assignments if r["split"] == "test"}

assert not train.intersection(test)
assert len(train) + len(test) == len(rows)

# No structural connectivity may appear in both sets.
connectivity_splits = defaultdict(set)
group_splits = defaultdict(set)

for record in assignments:
    connectivity_splits[record["connectivity"]].add(record["split"])
    group_splits[record["group_id"]].add(record["split"])

assert all(len(s) == 1 for s in connectivity_splits.values())
assert all(len(s) == 1 for s in group_splits.values())

def describe(selected_ids):
    selected = [
        r for r in rows if r["compound_id"] in selected_ids
    ]
    y = sorted(float(r["pic50_molar"]) for r in selected)

    def percentile(q):
        return round(y[int((len(y) - 1) * q)], 3)

    return {
        "compounds": len(selected),
        "pic50_min": round(y[0], 3),
        "pic50_q25": percentile(0.25),
        "pic50_median": round(median(y), 3),
        "pic50_q75": percentile(0.75),
        "pic50_max": round(y[-1], 3),
        "replicate_compounds": sum(
            int(r["n_quant"]) > 1 for r in selected
        ),
        "OR_stereo_compounds": sum(
            r["has_stereo_or"] == "True" for r in selected
        ),
        "compounds_with_censored_records": sum(
            int(r["n_weak"]) + int(r["n_strong"]) > 0
            for r in selected
        ),
    }

summary = {
    "status": "provisional",
    "seed": 42,
    "rdkit_version": rdBase.rdkitVersion,
    "grouping": "Butina, Morgan radius 2, 2048 bits, no chirality",
    "similarity_threshold": 0.70,
    "source_sha256": hashlib.sha256(
        source.read_bytes()
    ).hexdigest(),
    "train_groups": len(groups) - len(test_ids),
    "test_groups": len(test_ids),
    "train": describe(train),
    "test": describe(test),
}

output.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=assignments[0].keys())
    writer.writeheader()
    writer.writerows(assignments)

report_path.write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))
print("\nConnectivity/group leakage checks: PASS")
print("Saved:", output)
print("Saved:", report_path)
