"""
test_tunable_dedup.py - Regression tests for Contribution 4 (Pi_c).
Paste these into tests/test_all.py, or run standalone with:
    python -m tests.test_tunable_dedup

Guards the storage-leakage scheme against silent breakage:
  - endpoint c=1   : S(1) == T   (no dedup; baseline)
  - endpoint c=inf : S(inf) == U  (full dedup)
  - leakage monotonicity in c
  - real-crypto correctness and replica unlinkability
"""

import os
import sys
import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from system.tunable_dedup import profile, TunableDedupEncryptor
from crypto.paillier import keygen


def _toy_paths(depth=6, n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 8) * 100.0
    y = (X[:, 1] > 50).astype(int)          # depends on a continuous feature
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
    return PathExtractor(from_sklearn_tree(clf)).extract_paths()


def test_endpoint_c1_equals_T():
    paths = _toy_paths()
    p = profile(paths, 1)
    assert p['ciphertexts_S'] == p['total_conditions_T'], \
        "c=1 must store one ciphertext per condition (S = T)"


def test_endpoint_cinf_equals_U():
    paths = _toy_paths()
    p = profile(paths, math.inf)
    assert p['ciphertexts_S'] == p['unique_values_U'], \
        "c=inf must store one ciphertext per unique value (S = U)"


def test_storage_monotone_nonincreasing():
    paths = _toy_paths()
    cs = [1, 2, 4, 8, 16, math.inf]
    S = [profile(paths, c)['ciphertexts_S'] for c in cs]
    assert all(S[i] >= S[i + 1] for i in range(len(S) - 1)), \
        "storage must be non-increasing as c grows"


def test_leakage_monotone_nondecreasing():
    paths = _toy_paths()
    cs = [1, 2, 4, 8, 16, math.inf]
    L = [profile(paths, c)['max_linkability'] for c in cs]
    assert all(L[i] <= L[i + 1] for i in range(len(L) - 1)), \
        "max-linkability must be non-decreasing as c grows"


def test_real_crypto_correctness_and_unlinkability():
    paths = _toy_paths(depth=6)
    enc = TunableDedupEncryptor(512, _keys=keygen(512))   # 512-bit: fast for CI
    model = enc.build(paths, c=4)
    assert enc.verify_correctness(model, paths), "decrypt must match original"
    assert enc.check_unlinkability(model), "replicas must be distinct ciphertexts"


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tunable-dedup tests passed")
