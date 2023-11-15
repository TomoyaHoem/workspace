import numpy as np
from indices.baroni_urbani_buser import BaroniUrbaniBuser


def internal_similarity(
    self, molecules, c_threshold=None, w_factor="fraction"
) -> float:
    """
    Returns the extended Faith similarity for the list of molecules
    """
    fingerprints = np.array([molecule.fingerprint for molecule in molecules])
    multiple_comparisons_index = BaroniUrbaniBuser(
        fingerprints=fingerprints, c_threshold=c_threshold, w_factor=w_factor
    )
    return multiple_comparisons_index.__dict__["Fai_1sim_wdis"]


def main() -> None:
    pass


if __name__ == "__main__":
    main()
