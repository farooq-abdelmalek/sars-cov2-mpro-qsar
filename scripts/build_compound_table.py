import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]

source = ROOT / (
    "data/raw/"
    "cdd_noncovalent_dates_2023_10_18_filt.csv"
)

output = ROOT / (
    "data/processed/"
    "moonshot_quantitative_compounds.csv"
)

by_id = defaultdict(list)

with source.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        by_id[row["Canonical_PostEra_ID"]].append(row)

compounds = []

for compound_id, measurements in sorted(by_id.items()):
    quant = [
        row for row in measurements
        if row["Semiquant"] == "Quant"
    ]

    if not quant:
        continue

    structures = [
        Chem.MolFromSmiles(row["suspected_SMILES"])
        for row in measurements
    ]

    if any(mol is None for mol in structures):
        raise ValueError(f"Invalid SMILES: {compound_id}")

    connectivities = {
        Chem.MolToSmiles(mol, isomericSmiles=False)
        for mol in structures
    }

    if len(connectivities) != 1:
        raise ValueError(
            f"Conflicting connectivity: {compound_id}"
        )

    mol = structures[0]

    values = [float(row["IC50_(µM)"]) for row in quant]

    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"Invalid IC50: {compound_id}")

    # Median in log space: robust initial aggregation.
    log_values = [math.log10(v) for v in values]
    median_log = median(log_values)

    # Count measurement confidence intervals that
    # cannot be interpreted as valid positive bounds.
    valid_ci = 0
    ci_ratios = []

    for row in quant:
        try:
            lower = float(row["IC50_CI_(Lower)_(µM)"])
            upper = float(row["IC50_CI_(Upper)_(µM)"])

            if (
                math.isfinite(lower)
                and math.isfinite(upper)
                and 0 < lower <= upper
            ):
                valid_ci += 1
                ci_ratios.append(upper / lower)
        except (ValueError, TypeError):
            pass

    compounds.append({
        "compound_id": compound_id,
        "smiles": Chem.MolToSmiles(
            mol, isomericSmiles=True
        ),
        "connectivity": next(iter(connectivities)),
        "has_stereo_or": any(
            str(group.GetGroupType()) == "STEREO_OR"
            for group in mol.GetStereoGroups()
        ),
        "n_quant": len(quant),
        "n_weak": sum(
            row["Semiquant"] == "Weak"
            for row in measurements
        ),
        "n_strong": sum(
            row["Semiquant"] == "Strong"
            for row in measurements
        ),
        "ic50_median_uM": 10 ** median_log,
        "pic50_molar": 6 - median_log,
        "replicate_fold_range": max(values) / min(values),
        "n_valid_ci": valid_ci,
        "max_ci_upper_lower_ratio": (
            max(ci_ratios) if ci_ratios else ""
        ),
    })

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=compounds[0].keys()
    )
    writer.writeheader()
    writer.writerows(compounds)

print("Compound-level curation completed")
print("Compounds:", len(compounds))
print(
    "Multiple quantitative measurements:",
    sum(c["n_quant"] > 1 for c in compounds)
)
print(
    "Replicate IC50 variation >2-fold:",
    sum(c["replicate_fold_range"] > 2 for c in compounds)
)
print(
    "Compounds with additional censored records:",
    sum(
        c["n_weak"] + c["n_strong"] > 0
        for c in compounds
    )
)
print(
    "Compounds with OR stereochemistry:",
    sum(c["has_stereo_or"] for c in compounds)
)
print("Output:", output)
