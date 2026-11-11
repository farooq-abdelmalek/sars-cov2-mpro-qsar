# Mpro QSAR — Model Selection Plan

## Scientific objective

Predict quantitative SARS-CoV-2 Mpro inhibitory potency
(pIC50), with primary emphasis on generalization to
chemically separated compounds.

## Dataset and validation

- 1,236 curated quantitative compounds.
- 969 development compounds.
- 267 held-out test compounds.
- Butina 0.70 molecular grouping.
- Five fixed, group-aware development CV folds.
- No test-set evaluation during model selection.

## Reference baseline

- Training-fold median predictor.
- Observed mean development CV MAE: 0.7889 pIC50.

## Primary model families

1. Ridge Regression — regularized linear model.
2. RBF-SVR — nonlinear kernel method.
3. Random Forest — bagged decision-tree ensemble.
4. XGBoost — gradient-boosted decision trees.
5. Tanimoto kNN — chemical analog-based prediction.

## Feature representations

- Ten RDKit physicochemical descriptors.
- Binary Morgan fingerprints, radius 2, 2048 bits.
- Hybrid descriptors plus fingerprints.

Ridge, SVR, RF, and XGBoost use all three
representations. Tanimoto kNN uses fingerprints only.

Total: 13 primary model-feature configurations.

## Preprocessing

- Scale continuous descriptors for Ridge and SVR.
- Preserve binary fingerprints without standardization.
- Use unscaled features for tree-based models.
- Fit learned preprocessing within each training fold.

## Evaluation

Primary: mean five-fold validation MAE in pIC50 units.

Secondary:
- RMSE
- R-squared
- Fold-level metric variability
- Performance relative to the median baseline

## Model selection safeguards

- Reuse identical CV folds for every candidate.
- No data-dependent preprocessing outside training folds.
- Select and tune models without accessing test targets.
- Freeze model and hyperparameters before final testing.

## Optional secondary analyses

- LightGBM comparison with XGBoost.
- Expanded 15-descriptor panel.
- Stereochemistry-sensitive fingerprints.

These are not part of the primary benchmark.

## Remaining design decisions

- Initial hyperparameters for each model.
- Tanimoto kNN neighbor weighting and neighbor count.
- Equal hyperparameter optimization budget.
- Experiment tracking configuration.
