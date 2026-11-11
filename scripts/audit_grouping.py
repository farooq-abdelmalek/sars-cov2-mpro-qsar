import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]

source = ROOT / "data/processed/moonshot_quantitative_compounds.csv"
output = ROOT / "results/data_qualification/grouping_feasibility.json"

with source.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# Group compound IDs by molecular connectivity.
members = defaultdict(list)
for row in rows:
    members[row["connectivity"]].append(row["compound_id"])

keys = sorted(members)
n = len(keys)

molecules = [Chem.MolFromSmiles(s) for s in keys]
if any(m is None for m in molecules):
    raise ValueError("Invalid connectivity SMILES")

# Chirality-free fingerprints for conservative grouping.
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False,
)
fps = [generator.GetFingerprint(m) for m in molecules]

scaffolds = [
    MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    or "[acyclic]"
    for m in molecules
]

# RDKit Butina expects a condensed lower-triangle matrix.
distances = []
for i in range(1, n):
    similarities = DataStructs.BulkTanimotoSimilarity(
        fps[i], fps[:i]
    )
    distances.extend(1.0 - s for s in similarities)

def make_butina_labels(similarity_threshold):
    clusters = Butina.ClusterData(
        distances,
        n,
        distThresh=1.0 - similarity_threshold,
        isDistData=True,
        reordering=True,
    )

    labels = [-1] * n
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    if any(label < 0 for label in labels):
        raise ValueError("Unassigned Butina molecule")

    return labels

methods = {
    "connectivity": list(range(n)),
    "murcko": scaffolds,
    "butina_0.70": make_butina_labels(0.70),
    "butina_0.80": make_butina_labels(0.80),
}

def percentile(values, fraction):
    values = sorted(values)
    return round(values[int((len(values) - 1) * fraction)], 3)

results = {}

for method, labels in methods.items():
    # Group sizes count compound IDs, not just connectivity keys.
    sizes = Counter()
    for index, label in enumerate(labels):
        sizes[label] += len(members[keys[index]])

    # For every unique connectivity, find its nearest
    # neighbor belonging to a DIFFERENT group.
    nearest_external = [0.0] * n

    position = 0
    for i in range(1, n):
        for j in range(i):
            if labels[i] != labels[j]:
                similarity = 1.0 - distances[position]
                nearest_external[i] = max(
                    nearest_external[i], similarity
                )
                nearest_external[j] = max(
                    nearest_external[j], similarity
                )
            position += 1

    if position != len(distances):
        raise ValueError("Distance matrix indexing error")

    report = {
        "groups": len(sizes),
        "singleton_groups": sum(
            size == 1 for size in sizes.values()
        ),
        "largest_group": max(sizes.values()),
        "median_group_size": median(sizes.values()),
        "cross_group_nearest_tanimoto_median": percentile(
            nearest_external, 0.5
        ),
        "cross_group_nearest_tanimoto_p90": percentile(
            nearest_external, 0.9
        ),
        "fraction_cross_group_similarity_ge_085": round(
            sum(s >= 0.85 for s in nearest_external) / n,
            3,
        ),
    }
    results[method] = report

    print(f"\n=== {method} ===")
    for key, value in report.items():
        print(f"{key}: {value}")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps({
        "rdkit_version": rdBase.rdkitVersion,
        "compound_ids": len(rows),
        "unique_connectivities": n,
        "fingerprint": "Morgan radius=2, 2048 bits, no chirality",
        "methods": results,
    }, indent=2) + "\n",
    encoding="utf-8",
)

print(f"\nSaved: {output}")
