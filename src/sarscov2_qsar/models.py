"""Unfitted regression pipelines for the Mpro QSAR benchmark."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sarscov2_qsar.preprocessing import make_preprocessor


MODEL_NAMES = ("ridge", "svr", "random_forest", "xgboost")
REPRESENTATIONS = ("descriptors", "fingerprints", "hybrid")


def make_model(name, representation, random_state=42):
    """Create a fresh regression pipeline."""

    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")

    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")

    if name == "ridge":
        estimator = Ridge(alpha=1.0)

    elif name == "svr":
        estimator = SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale",
        )

    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=2,
        )

    else:
        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=2,
            verbosity=0,
        )

    scale_descriptors = name in ("ridge", "svr")

    return Pipeline([
        (
            "preprocess",
            make_preprocessor(
                representation,
                scale_descriptors=scale_descriptors,
            ),
        ),
        ("model", estimator),
    ])
