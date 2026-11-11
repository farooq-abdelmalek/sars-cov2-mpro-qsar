"""SARS-CoV-2 Mpro QSAR reproducibility workflow.

First milestone: verify existing frozen research results.
Later milestones will add reproducible upstream generation.
"""

PYTHON = ".venv/bin/python"


rule all:
    input:
        "tracking/workflow/frozen_snapshot_verified.json"


rule verify_frozen_snapshot:
    input:
        checker="scripts/verify_frozen_snapshot.py",
        selection="results/final_model_selection.json",
        manifest="results/final_training_manifest.json",
        metrics="results/final_test_metrics.json",
        predictions="results/final_test_predictions.csv",
        diagnostics="results/diagnostics/heldout_error_profile.json",
        split="data/splits/butina_070_seed42.csv",
    output:
        "tracking/workflow/frozen_snapshot_verified.json"
    shell:
        "{PYTHON} scripts/verify_frozen_snapshot.py "
        "--output {output:q}"
