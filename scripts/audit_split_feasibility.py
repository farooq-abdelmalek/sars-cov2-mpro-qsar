import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]

input_file = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
output_file = ROOT / "results/data_qualification/split_feasibility.json"

with input_file.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# Each connectivity represents one molecular grouping unit.
by_connectivity = defaultdict(list)
for row in rows:
    by_connectivity[row["connectivity"]].append(row["compound_id"])

keys = sorted(by_connectivity)
mols = [Chem.MolFromSmiles(s) for s in keys]
assert all(m is not None for m in mols)

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False
)
fps = [generator.GetFingerprint(m) for m in mols]

# Calculate the condensed Tanimoto distance matrix.
distances = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(
        fps[i], fps[:i]
    )
    distances.extend(1 - sim for sim in sims)

def butina_labels(threshold):
    clusters = Butina.ClusterData(
        distances,
        len(fps),
        1 - threshold,
        isDistData=True,
        reordering=True,
    )
    labels = [None] * len(fps)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
    assert all(label is not None for label in labels)
    return labels

methods = {
    "murcko": [
        MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        or "[acyclic]"
        for m in mols
    ],
    "butina_0.70": butina_labels(0.70),
    "butina_0.80": butina_labels(0.80),
}

def percentile(values, q):
    values = sorted(values)
    return round(values[int((len(values) - 1) * q)], 3)

results = {}

for name, labels in methods.items():
    groups = defaultdict(list)

    for index, label in enumerate(labels):
        groups[str(label)].append(index)

    # Deterministic group allocation.
    group_names = sorted(groups)
    random.Random(42).shuffle(group_names)

    target = round(0.20 * len(rows))
    test_indices = set()
    test_compound_count = 0

    for group_name in group_names:
        indices = groups[group_name]
        test_indices.update(indices)
        test_compound_count += sum(
            len(by_connectivity[keys[i]])
            for i in indices
        )
        if test_compound_count >= target:
            break

    train_indices = [
        i for i in range(len(keys))
        if i not in test_indices
    ]

    assert train_indices and test_indices

    train_fps = [fps[i] for i in train_indices]

    # Similarity of each held-out structure to its
    # nearest training structure.
    similarities = []
    for i in sorted(test_indices):
        best = max(
            DataStructs.BulkTanimotoSimilarity(
                fps[i], train_fps
            )
        )

        # Report statistics at the compound-ID level.
        similarities.extend(
            [best] * len(by_connectivity[keys[i]])
        )

    report = {
        "train_compounds": len(rows) - test_compound_count,
        "test_compounds": test_compound_count,
        "test_fraction": round(
            test_compound_count / len(rows), 3
        ),
        "test_groups": len(group_names) - (
            len(groups) -
            len({str(labels[i]) for i in test_indices})
        ),
        "train_test_nearest_tanimoto_median":
            round(median(similarities), 3),
        "train_test_nearest_tanimoto_p90":
            percentile(similarities, 0.90),
        "fraction_test_similarity_ge_085": round(
            sum(x >= 0.85 for x in similarities)
            / len(similarities), 3
        ),
    }

    results[name] = report

    print(f"\n=== {name} ===")
    for field, value in report.items():
        print(f"{field}: {value}")

output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(
    json.dumps(results, indent=2) + "\n",
    encoding="utf-8",
)

print("\nSaved:", output_file)
