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
