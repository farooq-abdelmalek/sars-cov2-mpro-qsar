from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import csv
import hashlib
import json

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise SystemExit(
        "RDKit is missing in this Python environment. "
        "Share this error before installing anything."
    )

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "cdd_noncovalent_dates_2023_10_18_filt.csv",
    "cdd_achiral_enantiopure_dates_2023_10_18_filt.csv",
]

fp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = int((len(values) - 1) * fraction)
    return round(values[index], 3)


def audit_dataset(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    invalid = []

    for line_number, row in enumerate(rows, start=2):
        smiles = row["suspected_SMILES"]

        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

        if mol is None:
            invalid.append({
                "line": line_number,
                "id": row["Canonical_PostEra_ID"],
            })
            continue

        parsed.append({
            "id": row["Canonical_PostEra_ID"],
            "label": row["Semiquant"],
            "identity": Chem.MolToCXSmiles(mol),
            "plain": Chem.MolToSmiles(
                mol, isomericSmiles=False
            ),
            "scaffold": (
                MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                or "[acyclic]"
            ),
            "has_cx": " |" in smiles,
            "has_stereo_group": len(mol.GetStereoGroups()) > 0,
            "mol": mol,
        })

    # Identify compound IDs mapped to multiple structures.
    structures_by_id = defaultdict(set)

    for record in parsed:
        structures_by_id[record["id"]].add(record["identity"])

    conflicts = {
        compound_id
        for compound_id, structures in structures_by_id.items()
        if len(structures) > 1
    }

    # Quantitative observations only.
    quant_by_id = defaultdict(list)

    for record in parsed:
        if record["label"] == "Quant":
            quant_by_id[record["id"]].append(record)

    # One structure per compound for diversity analysis.
    # Exclude IDs with conflicting molecular structures.
    compounds = [
        observations[0]
        for compound_id, observations in sorted(quant_by_id.items())
        if compound_id not in conflicts
    ]

    identities = Counter(
        record["identity"] for record in compounds
    )

    scaffolds = Counter(
        record["scaffold"] for record in compounds
    )

    stereo_groups = defaultdict(set)

    for record in compounds:
        stereo_groups[record["plain"]].add(
            record["identity"]
        )

    # Calculate nearest-neighbor molecular similarity.
    fingerprints = [
        fp_generator.GetFingerprint(record["mol"])
        for record in compounds
    ]

    nearest = [0.0] * len(fingerprints)

    for i, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints[i + 1:]
        )

        for j, similarity in enumerate(
            similarities, start=i + 1
        ):
            nearest[i] = max(nearest[i], similarity)
            nearest[j] = max(nearest[j], similarity)

    return {
        "file": path.name,
        "sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
        "total_rows": len(rows),
        "invalid_smiles_count": len(invalid),
        "invalid_examples": invalid[:10],
        "rows_with_CX_annotations": sum(
            r["has_cx"] for r in parsed
        ),
        "rows_with_stereo_groups": sum(
            r["has_stereo_group"] for r in parsed
        ),
        "conflicting_compound_IDs": len(conflicts),
        "conflict_examples": sorted(conflicts)[:10],
        "quantitative_rows": sum(
            r["label"] == "Quant" for r in parsed
        ),
        "quantitative_unique_IDs": len(quant_by_id),
        "quantitative_compounds_analyzed": len(compounds),
        "identical_structure_groups_across_IDs": sum(
            count > 1 for count in identities.values()
        ),
        "nonstereo_groups_with_multiple_stereo_identities": sum(
            len(values) > 1
            for values in stereo_groups.values()
        ),
        "scaffold_groups_including_acyclic": len(scaffolds),
        "singleton_scaffolds": sum(
            count == 1 for count in scaffolds.values()
        ),
        "largest_scaffold_group": max(
            scaffolds.values(), default=0
        ),
        "acyclic_compounds": scaffolds.get("[acyclic]", 0),
        "nearest_neighbor_tanimoto": {
            "p25": percentile(nearest, 0.25),
            "median": (
                round(median(nearest), 3)
                if nearest else None
            ),
            "p75": percentile(nearest, 0.75),
            "p95": percentile(nearest, 0.95),
            "fraction_at_least_0.85": (
                round(
                    sum(x >= 0.85 for x in nearest)
                    / len(nearest), 3
                )
                if nearest else None
            ),
        },
    }


def main():
    results = {
        "rdkit_version": rdBase.rdkitVersion,
        "datasets": [],
    }

    for filename in FILES:
        path = ROOT / "data" / "raw" / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        report = audit_dataset(path)
        results["datasets"].append(report)

        print("\n" + json.dumps(report, indent=2))

    output = (
        ROOT / "results" / "data_qualification"
        / "structure_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nAudit saved to: {output}")


if __name__ == "__main__":
    main()
