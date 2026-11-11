"""Molecular features for the Mpro QSAR study."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


# Small, interpretable set of 2D molecular descriptors.
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}

MORGAN_SIZE = 2048

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=MORGAN_SIZE,
    includeChirality=False,
)


def featurize(smiles):
    """Return descriptor and fingerprint arrays for a SMILES."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descriptors = np.array(
        [fn(mol) for fn in DESCRIPTORS.values()],
        dtype=np.float64,
    )

    fingerprint = np.zeros(MORGAN_SIZE, dtype=np.uint8)
    fp = morgan_generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, fingerprint)

    if not np.isfinite(descriptors).all():
        raise ValueError("Non-finite molecular descriptor")

    return descriptors, fingerprint
