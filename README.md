# SARS-CoV-2 Mpro Inhibitor Potency Prediction

**A Reproducible QSAR Study with Chemical-Group Validation**

[![QSAR CI](https://github.com/farooq-abdelmalek/sars-cov2-mpro-qsar/actions/workflows/ci.yml/badge.svg)](https://github.com/farooq-abdelmalek/sars-cov2-mpro-qsar/actions/workflows/ci.yml)

**Reproducible molecular machine learning for predicting SARS-CoV-2 main protease inhibitor potency.**

This project investigates how well conventional machine-learning models predict experimental pIC50 values from molecular structure, particularly when evaluated on held-out chemical groups.

The focus is not only predictive performance, but also **data quality, leakage-aware validation, reproducible experiments, and honest reporting of generalization limits**.

## Key results

**Selected model:** Support Vector Regression (RBF kernel) using Morgan fingerprints and RDKit descriptors.

| Metric | Development CV | Held-out test |
|---|---:|---:|
| MAE (pIC50) | 0.4630 | **0.5014** |
| RMSE (pIC50) | 0.6029 | **0.6861** |
| R² | 0.5197 | **0.3579** |

The development values summarize five-fold grouped cross-validation; R² is averaged across folds. The test values come from one evaluation on 267 reserved compounds.

**Main finding:** The model learns useful structure–activity relationships, but its generalization to held-out chemical groups is limited. Errors are especially pronounced for weak and highly potent compounds.

The reported metrics are not evidence that the model is ready to prioritize compounds experimentally without further validation.

## Project overview

| Component | Implementation |
|---|---|
| Dataset | COVID Moonshot noncovalent Mpro IC50 measurements |
| Regression target | Experimental pIC50 |
| Quantitative compounds | 1,236 |
| Development / held-out test | 969 / 267 |
| Molecular representations | 10 RDKit descriptors, 2,048-bit Morgan fingerprints, hybrid |
| Model families | Ridge, SVR, Random Forest, XGBoost, Tanimoto kNN |
| Initial benchmark | 13 model–representation configurations |
| Model selection | Five-fold group-aware CV, minimizing mean MAE |
| Hyperparameter tuning | Optuna TPE, 20 trials per selected finalist |
| Experiment tracking | MLflow and Optuna |
| Workflow orchestration | Snakemake |
| Reproducible environment | Docker |
| Automated checks | pytest and GitHub Actions |

## 1. Dataset and preprocessing

The data originate from the
[COVID Moonshot cleaned dataset](https://github.com/asapdiscovery/COVID_moonshot_data_clean).

The upstream source was pinned at commit
`bb7317c5f8c122de5190fbaf7fc9e92d1077841b`.

The source contains 2,146 experimental records:
- 1,450 quantitative IC50 measurements;
- 670 measurements classified as weak and censored;
- 26 measurements classified as strong and censored.

Quantitative measurements were aggregated within compound IDs using the median of log10(IC50 in micromolar), resulting in **1,236 compound-level regression targets**.

The transformation used is:

`pIC50 = 6 - median(log10(IC50 in µM))`

Censored measurements are recorded but are not used as exact regression targets.

The data audit also addresses repeated measurements, chemical identity, and stereochemical uncertainty.

Full source information and attribution:
[Data provenance](docs/data_provenance.md).

## 2. Chemical representations

Three molecular feature sets were evaluated:

| Representation | Features |
|---|---|
| RDKit descriptors | 10 physicochemical descriptors |
| Morgan fingerprints | 2,048 bits, radius 2 |
| Hybrid | 2,058 features combining both |

Descriptors include molecular weight, logP, TPSA, donor/acceptor counts, rotatable bonds, and ring-related properties.

Descriptor scaling is fitted within the relevant training folds for models that require it. Fingerprint bits remain binary.

These are **ligand-only, primarily two-dimensional representations**. They do not explicitly include protein–ligand binding poses, molecular dynamics, or assay-specific experimental conditions.

## 3. Validation design

A random molecule-level split can give misleadingly optimistic estimates when related structures appear in both training and evaluation data.

This project uses a fixed chemical-group split constructed with Butina clustering and Morgan-fingerprint Tanimoto similarity.

- Clustering similarity threshold: 0.70
- Split seed: 42
- Development compounds: 969
- Held-out test compounds: 267
- Held-out chemical groups: 136

The five development CV folds preserve chemical-group separation.

Model selection, feature comparisons, and tuning use **development data only**.

The test set is used once for the primary evaluation of the frozen selected model.

The exact split assignments, CV folds, and experimental protocols are included in the repository.

See [model-selection plan](docs/model_selection_plan.md).

## 4. Model benchmarking and optimization

The initial experiment compared 13 model–representation configurations across five model families.

The best-performing configurations from the initial benchmark and subsequent optimization were:

| Configuration | Mean development CV MAE |
|---|---:|
| **SVR + Hybrid, initial** | **0.4630** |
| SVR + Hybrid, tuned | 0.4675 |
| XGBoost + Morgan, tuned | 0.4793 |
| XGBoost + Morgan, initial | 0.4962 |

For context, the development median-prediction baseline achieved MAE 0.7889.

Two distinct finalist families were selected before tuning:
SVR with hybrid features and XGBoost with Morgan fingerprints.

Each underwent 20 Optuna TPE trials using the same five grouped CV folds.

**The initial SVR configuration remained the overall winner under the frozen MAE-based selection rule.**

Final hyperparameters:

`SVR(C=10, epsilon=0.1, gamma="scale", kernel="rbf")`

The selected pipeline was then refitted on all 969 development compounds before the held-out evaluation.

Full experimental outputs:
- [Initial benchmark summary](results/benchmarks/initial_cv_summary.csv)
- [Initial fold-level results](results/benchmarks/initial_cv_folds.csv)
- [Tuning verification](results/tuning/tuning_verification.json)
- [Final model selection](results/final_model_selection.json)

## 5. Held-out evaluation and limitations

The primary held-out results are:

| Metric | Value |
|---|---:|
| MAE | 0.5014 pIC50 |
| RMSE | 0.6861 pIC50 |
| R² | 0.3579 |
| Median absolute error | 0.3637 pIC50 |
| 90th-percentile absolute error | 1.1168 pIC50 |

An MAE of approximately 0.50 log10 potency units corresponds to an approximately 3.2-fold IC50 difference at that error magnitude.

The main diagnostic finding is **regression toward the central potency range**:

- Weak compounds tend to have their potency overpredicted.
- Highly potent compounds tend to have their potency underpredicted.
- Approximately 13.9% of held-out predictions have absolute error greater than 1 pIC50 unit.

Important limitations include the relatively small quantitative dataset, excluded censored activity measurements, experimental measurement uncertainty, stereochemical ambiguity, and limited ligand-only descriptors.

The chemical-group split makes evaluation more demanding than a simple random split, but it does not establish generalization to every new chemical scaffold, experimental assay, or biological setting.

The project reports these limitations without modifying the model after its primary held-out evaluation.

Archived evidence:
- [Test metrics](results/final_test_metrics.json)
- [Compound-level test predictions](results/final_test_predictions.csv)
- [Error analysis](results/diagnostics/heldout_error_profile.json)

## 6. Reproducing the verified workflow

**Recommended: Docker**

From the repository root:

```bash
docker build -t mpro-qsar:local .
docker run --rm mpro-qsar:local
```

The container runs Snakemake, reconstructs data-preparation artifacts in an isolated workspace, and checks them against the archived reference files.

It verifies:
1. Compound-table reconstruction
2. Chemical-group splitting
3. Five-fold CV assignments
4. Molecular feature generation
5. Frozen evaluation record integrity

Run the automated scientific-integrity tests with:

```bash
docker run --rm --network none mpro-qsar:local \
  .venv/bin/python -m pytest tests/ -v
```

The test suite checks group separation, CV assignments, feature integrity, model-selection consistency, saved test metrics, and provenance.

**Reproducibility boundary:** The default Snakemake workflow reproduces data preparation and verifies archived research results. It does **not** automatically retrain all benchmark models, rerun Optuna, or repeat the primary held-out evaluation.

The corresponding experiment scripts and frozen outputs are provided separately. The trained model binary and local MLflow/Optuna databases are not committed to Git. Exported trial results, metrics, protocols, and prediction records are preserved.

## 7. Repository structure

```text
data/
  raw/                Original Moonshot source files
  processed/          Aggregated quantitative compounds
  splits/             Frozen chemical split and CV folds
  features/           RDKit descriptors and Morgan fingerprints
src/sarscov2_qsar/     Reusable molecular ML code
scripts/              Curation, experiments, evaluation, verification
workflow/             Snakemake data-reproduction rules
tests/                Automated scientific-integrity tests
results/              Archived protocols, metrics, and predictions
docs/                 Methodology, provenance, upstream attribution
Snakefile             Workflow entry point
Dockerfile            Container environment
.github/workflows/    Continuous-integration configuration
```

### Engineering stack

Python · RDKit · NumPy · scikit-learn · XGBoost · Optuna · MLflow · Snakemake · Docker · pytest · Git · GitHub Actions

The CI configuration builds the Docker environment, runs the integrity tests, and executes the Snakemake verification workflow. GitHub-hosted CI execution will be confirmed after publication.

## 8. Data attribution and licensing

Experimental data originate from the COVID Moonshot project.

Original repository:
[COVID_moonshot_data_clean](https://github.com/asapdiscovery/COVID_moonshot_data_clean).

The upstream MIT license is retained in
[docs/third_party/COVID_moonshot_data_clean_LICENSE](docs/third_party/COVID_moonshot_data_clean_LICENSE).

Dataset attribution does not imply endorsement by the original contributors. The original code in this repository is released under the [MIT License](LICENSE). Third-party data and materials retain their respective copyright and attribution notices.

## References

- [COVID Moonshot cleaned experimental data](https://github.com/asapdiscovery/COVID_moonshot_data_clean)
- [RDKit documentation](https://www.rdkit.org/docs/)
- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [MLflow documentation](https://mlflow.org/docs/latest/)
- [Snakemake documentation](https://snakemake.readthedocs.io/)
