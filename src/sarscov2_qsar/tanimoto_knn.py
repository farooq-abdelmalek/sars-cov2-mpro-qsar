"""Tanimoto nearest-neighbor regression for binary fingerprints."""

import numpy as np
from rdkit import DataStructs
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def to_bitvectors(X):
    """Convert a binary feature matrix to RDKit bit vectors."""
    X = np.asarray(X)

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Expected a 2D fingerprint matrix.")

    if not np.isin(X, [0, 1]).all():
        raise ValueError("Fingerprints must contain only 0 and 1.")

    return [
        DataStructs.CreateFromBitString(
            "".join(str(int(bit)) for bit in row)
        )
        for row in X
    ]


class TanimotoKNNRegressor(RegressorMixin, BaseEstimator):
    """Predict from the most similar training fingerprints."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y have incompatible shapes.")

        if not np.isfinite(y).all():
            raise ValueError("Targets must be finite.")

        if not isinstance(self.n_neighbors, int) or not (
            1 <= self.n_neighbors <= len(y)
        ):
            raise ValueError("Invalid number of neighbors.")

        self.n_features_in_ = X.shape[1]
        self.train_fps_ = to_bitvectors(X)
        self.train_targets_ = y.copy()
        self.fallback_ = float(np.median(y))

        return self

    def predict(self, X):
        check_is_fitted(self, "train_fps_")

        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected fingerprint dimensions.")

        query_fps = to_bitvectors(X)
        predictions = []

        for query in query_fps:
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    query, self.train_fps_
                )
            )

            nearest = np.argsort(
                -similarities, kind="stable"
            )[:self.n_neighbors]

            weights = similarities[nearest]

            if weights.sum() == 0:
                prediction = self.fallback_
            else:
                prediction = np.average(
                    self.train_targets_[nearest],
                    weights=weights,
                )

            predictions.append(float(prediction))

        return np.asarray(predictions)
