"""Preprocessing for molecular feature representations."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

N_DESCRIPTORS = 10


def make_preprocessor(representation, scale_descriptors):
    """Return a fresh, unfitted feature preprocessor."""

    if representation == "descriptors":
        if scale_descriptors:
            return StandardScaler()
        return FunctionTransformer()

    if representation == "fingerprints":
        return FunctionTransformer()

    if representation == "hybrid":
        descriptor_transform = (
            StandardScaler() if scale_descriptors else "passthrough"
        )

        return ColumnTransformer(
            transformers=[
                ("descriptors", descriptor_transform, slice(0, N_DESCRIPTORS)),
                ("fingerprints", "passthrough", slice(N_DESCRIPTORS, None)),
            ],
            remainder="drop",
        )

    raise ValueError(f"Unknown representation: {representation}")
