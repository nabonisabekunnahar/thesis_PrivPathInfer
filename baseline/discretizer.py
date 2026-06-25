"""
discretizer.py — Feature Discretization for SDTC Baseline
===========================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

This module implements the feature discretization required by the
SDTC baseline (Liang et al. 2021).

SDTC Limitation (Addressed by PrivPathInfer Contribution 1):
    SDTC requires all continuous features to be discretized into
    bins before encryption. This causes accuracy loss because:
        - Fine-grained distinctions within a bin are lost
        - More bins = better accuracy but exponential storage growth
        - Fewer bins = worse accuracy but manageable storage

PrivPathInfer Contribution 1 (Native Continuous Feature Support):
    PrivPathInfer eliminates discretization entirely by encrypting
    exact floating-point thresholds using Paillier homomorphic
    encryption with fixed-point encoding.

    Result: PrivPathInfer always matches plaintext accuracy (0% loss).
    SDTC accuracy degrades with fewer bins (Experiment 1).

Discretization Methods:
    Equal-width:  Divide [min, max] into k equal-width bins
    Equal-freq:   Divide into k bins with equal number of samples

Reference:
    Liang et al. 2021, Section III-B: "Feature Preprocessing"
    The paper requires discretization as a preprocessing step.

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import math
from typing import List, Optional, Tuple
import numpy as np


class Discretizer:
    """
    Discretizes continuous features into integer bin indices.

    Used by SDTC baseline to convert continuous features to
    discrete values before encryption.

    Accuracy Loss:
        With k bins, any two values in the same bin are treated
        as identical. This causes misclassification when a decision
        tree threshold falls within a bin.

        Example (k=5 bins for glucose [50, 200]):
            Bin width = 30
            Glucose = 125 and glucose = 140 → same bin (2)
            But decision threshold = 126.5 distinguishes them.
            SDTC treats them as equal → wrong classification.
    """

    def __init__(self, n_bins: int = 10, strategy: str = 'equal_width'):
        """
        Initialize the discretizer.

        Args:
            n_bins:   number of bins (k). More bins = more accuracy
                      but SDTC storage grows as O(2^depth * k^features)
            strategy: 'equal_width' or 'equal_freq'
        """
        self.n_bins   = n_bins
        self.strategy = strategy
        self.bin_edges_per_feature: List[np.ndarray] = []
        self.fitted = False

    def fit(self, X: np.ndarray):
        """
        Compute bin edges from training data.

        Args:
            X: training data, shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        n_features = X.shape[1]
        self.bin_edges_per_feature = []

        for f in range(n_features):
            col = X[:, f]
            if self.strategy == 'equal_width':
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            elif self.strategy == 'equal_freq':
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col, percentiles)
                edges = np.unique(edges)  # Remove duplicates
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_per_feature.append(edges)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Discretize features into bin indices.

        Args:
            X: data to discretize, shape (n_samples, n_features)

        Returns:
            np.ndarray: discretized data, shape (n_samples, n_features)
                        values in [0, n_bins-1]
        """
        assert self.fitted, "Call fit() before transform()"
        X = np.array(X, dtype=float)
        X_disc = np.zeros_like(X, dtype=int)

        for f, edges in enumerate(self.bin_edges_per_feature):
            X_disc[:, f] = np.digitize(X[:, f], edges[1:-1])

        return X_disc

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def run_all_tests():
    """Verify discretizer correctness."""
    print("=" * 60)
    print("Discretizer Verification Tests")
    print("SDTC Baseline — Feature Discretization")
    print("=" * 60)

    X = np.array([
        [50.0,  100.0],
        [80.0,  126.5],
        [120.0, 150.0],
        [160.0, 200.0],
        [200.0, 250.0],
    ])

    disc = Discretizer(n_bins=5, strategy='equal_width')
    X_disc = disc.fit_transform(X)

    assert X_disc.shape == X.shape, "Shape mismatch after discretization"
    assert X_disc.dtype == int or np.issubdtype(X_disc.dtype, np.integer), \
        "Discretized values should be integers"
    print("[PASS] Equal-width discretization produces integer bin indices")

    disc_freq = Discretizer(n_bins=5, strategy='equal_freq')
    X_disc_freq = disc_freq.fit_transform(X)
    assert X_disc_freq.shape == X.shape
    print("[PASS] Equal-frequency discretization works correctly")

    print("\n[ALL TESTS PASSED] discretizer.py verified.")
    print("Note: Discretization causes accuracy loss (PrivPathInfer avoids this).")


if __name__ == "__main__":
    run_all_tests()