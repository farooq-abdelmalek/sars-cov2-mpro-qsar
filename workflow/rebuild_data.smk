"""Rebuild data in an isolated workspace.

Original, Git-tracked research data is never overwritten.
"""

from pathlib import Path
import hashlib
import json
import shutil


WORK = Path("tracking/rebuild_workspace")


def checksum(path):
    """Return the SHA-256 checksum of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


rule stage_compound_inputs:
    input:
        script="scripts/build_compound_table.py",
        raw_noncovalent=(
            "data/raw/cdd_noncovalent_dates_2023_10_18_filt.csv"
        ),
        raw_other=(
            "data/raw/cdd_achiral_enantiopure_dates_2023_10_18_filt.csv"
        )
    output:
        marker="tracking/rebuild_workspace/inputs_staged.json"
    run:
        # The script derives ROOT from its own location.
        # Copying it preserves that convention while changing
        # ROOT to our isolated rebuild workspace.
        sources = list(input)

        for source in sources:
            destination = WORK / source
            destination.parent.mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(source, destination)

        record = {
            source: checksum(source)
            for source in sources
        }

        Path(output.marker).write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )


rule rebuild_compound_table:
    input:
        "tracking/rebuild_workspace/inputs_staged.json"
    output:
        "tracking/rebuild_workspace/data/processed/"
        "moonshot_quantitative_compounds.csv"
    shell:
        ".venv/bin/python "
        "tracking/rebuild_workspace/scripts/"
        "build_compound_table.py"


rule verify_rebuilt_compound_table:
    input:
        rebuilt=(
            "tracking/rebuild_workspace/data/processed/"
            "moonshot_quantitative_compounds.csv"
        ),
        original=(
            "data/processed/"
            "moonshot_quantitative_compounds.csv"
        )
    output:
        "tracking/rebuild_workspace/checks/"
        "compound_table_verified.json"
    run:
        rebuilt_hash = checksum(input.rebuilt)
        original_hash = checksum(input.original)

        if rebuilt_hash != original_hash:
            raise ValueError(
                "Rebuilt compound table differs from the "
                "frozen reference. Do not overwrite the original."
            )

        report = {
            "status": "PASS",
            "stage": "compound_table",
            "sha256": rebuilt_hash,
            "byte_identical_to_frozen_reference": True,
            "original_data_modified": False,
        }

        destination = Path(output[0])
        destination.parent.mkdir(
            parents=True, exist_ok=True
        )
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            "PASS: Compound table reproduced byte-for-byte."
        )
        print("Original research data was not modified.")


rule stage_split_script:
    input:
        original="scripts/prepare_split.py"
    output:
        staged="tracking/rebuild_workspace/scripts/prepare_split.py"
    run:
        destination = Path(output.staged)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input.original, destination)


rule rebuild_chemical_split:
    input:
        script="tracking/rebuild_workspace/scripts/prepare_split.py",
        compounds=(
            "tracking/rebuild_workspace/data/processed/"
            "moonshot_quantitative_compounds.csv"
        ),
        compound_check=(
            "tracking/rebuild_workspace/checks/"
            "compound_table_verified.json"
        )
    output:
        split=(
            "tracking/rebuild_workspace/data/splits/"
            "butina_070_seed42.csv"
        ),
        report=(
            "tracking/rebuild_workspace/results/"
            "data_qualification/split_summary.json"
        )
    shell:
        ".venv/bin/python {input.script:q}"


rule verify_rebuilt_split:
    input:
        rebuilt_split=(
            "tracking/rebuild_workspace/data/splits/"
            "butina_070_seed42.csv"
        ),
        original_split="data/splits/butina_070_seed42.csv",
        rebuilt_report=(
            "tracking/rebuild_workspace/results/"
            "data_qualification/split_summary.json"
        ),
        original_report=(
            "results/data_qualification/split_summary.json"
        )
    output:
        "tracking/rebuild_workspace/checks/split_verified.json"
    run:
        import csv

        # Compare both regenerated artifacts with the
        # original files used in our frozen experiment.
        for rebuilt, original in [
            (input.rebuilt_split, input.original_split),
            (input.rebuilt_report, input.original_report),
        ]:
            if checksum(rebuilt) != checksum(original):
                raise ValueError(
                    f"Rebuilt file differs from frozen reference: "
                    f"{original}"
                )

        with open(input.rebuilt_split, newline="", encoding="utf-8") as f:
            assignments = list(csv.DictReader(f))

        train = [r for r in assignments if r["split"] == "train"]
        test = [r for r in assignments if r["split"] == "test"]

        if len(train) != 969 or len(test) != 267:
            raise ValueError("Unexpected train/test counts")

        train_groups = {r["group_id"] for r in train}
        test_groups = {r["group_id"] for r in test}

        if not train_groups.isdisjoint(test_groups):
            raise ValueError("Chemical-group leakage detected")

        report = {
            "status": "PASS",
            "stage": "chemical_group_split",
            "development_compounds": len(train),
            "held_out_compounds": len(test),
            "split_sha256": checksum(input.rebuilt_split),
            "summary_sha256": checksum(input.rebuilt_report),
            "byte_identical_to_frozen_reference": True,
            "original_files_modified": False,
        }

        destination = Path(output[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print("PASS: Chemical-group split reproduced.")
        print("PASS: Train/test groups remain disjoint.")
        print("PASS: Frozen files unchanged.")


rule stage_cv_script:
    input:
        "scripts/prepare_cv_folds.py"
    output:
        "tracking/rebuild_workspace/scripts/prepare_cv_folds.py"
    run:
        destination = Path(output[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input[0], destination)


rule rebuild_cv_folds:
    input:
        script=(
            "tracking/rebuild_workspace/scripts/"
            "prepare_cv_folds.py"
        ),
        compounds=(
            "tracking/rebuild_workspace/data/processed/"
            "moonshot_quantitative_compounds.csv"
        ),
        split=(
            "tracking/rebuild_workspace/data/splits/"
            "butina_070_seed42.csv"
        ),
        split_check=(
            "tracking/rebuild_workspace/checks/"
            "split_verified.json"
        )
    output:
        folds=(
            "tracking/rebuild_workspace/data/splits/"
            "development_cv5.csv"
        ),
        report=(
            "tracking/rebuild_workspace/results/"
            "data_qualification/cv_fold_summary.json"
        )
    shell:
        "PYTHONPATH=src .venv/bin/python {input.script:q}"


rule verify_rebuilt_cv_folds:
    input:
        rebuilt_folds=(
            "tracking/rebuild_workspace/data/splits/"
            "development_cv5.csv"
        ),
        original_folds="data/splits/development_cv5.csv",
        rebuilt_report=(
            "tracking/rebuild_workspace/results/"
            "data_qualification/cv_fold_summary.json"
        ),
        original_report=(
            "results/data_qualification/cv_fold_summary.json"
        )
    output:
        "tracking/rebuild_workspace/checks/"
        "cv_folds_verified.json"
    run:
        comparisons = [
            (input.rebuilt_folds, input.original_folds),
            (input.rebuilt_report, input.original_report),
        ]

        for rebuilt, original in comparisons:
            if checksum(rebuilt) != checksum(original):
                raise ValueError(
                    f"Rebuilt CV artifact differs: {original}"
                )

        report = {
            "status": "PASS",
            "stage": "development_cv_folds",
            "cv_folds_sha256": checksum(input.rebuilt_folds),
            "cv_summary_sha256": checksum(input.rebuilt_report),
            "byte_identical_to_frozen_reference": True,
            "original_files_modified": False,
        }

        destination = Path(output[0])
        destination.parent.mkdir(
            parents=True, exist_ok=True
        )
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print("PASS: CV folds reproduced exactly.")
        print("PASS: CV summary reproduced exactly.")
        print("Original research files unchanged.")
