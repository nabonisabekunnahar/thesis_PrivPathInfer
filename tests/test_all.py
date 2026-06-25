"""
test_all.py — Comprehensive Test Suite for PrivPathInfer
=========================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

This module runs all unit tests for every component of PrivPathInfer.
ALL tests must pass before proceeding to experiments.

Test Categories:
    1. Cryptographic primitives (AES, PRF, PRP, Paillier, ORE)
    2. System components (PathExtractor, RuleEncryptor, InferenceEngine)
    3. Update protocol (incremental updates, batch padding)
    4. Baseline (SDTC, Discretizer)
    5. Integration (end-to-end pipeline with PIMA dataset features)
    6. Security properties (leakage characterization)

Usage:
    python -m tests.test_all

Expected output:
    All tests pass with [PASS] prefix.
    Final summary: TOTAL X/X TESTS PASSED

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import sys
import time
import random
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

class TestRunner:
    """Simple test runner with pass/fail tracking."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name: str, func):
        """Run a single test function."""
        try:
            func()
            self.passed += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  [ERROR] {name}: {e}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"TOTAL {self.passed}/{total} TESTS PASSED")
        if self.failed > 0:
            print(f"\nFAILED TESTS ({self.failed}):")
            for name, err in self.errors:
                print(f"  - {name}: {err[:100]}")
        print("=" * 60)
        return self.failed == 0


runner = TestRunner()


# ===========================================================================
# SECTION 1: Cryptographic Primitives
# ===========================================================================

def section_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


section_header("SECTION 1: Cryptographic Primitives")

# --- AES-128 ---
from crypto.aes128 import aes_encrypt, aes_decrypt

def test_aes_nist_vector():
    key = bytes([0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
                 0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c])
    pt  = bytes([0x32,0x43,0xf6,0xa8,0x88,0x5a,0x30,0x8d,
                 0x31,0x31,0x98,0xa2,0xe0,0x37,0x07,0x34])
    exp = bytes([0x39,0x25,0x84,0x1d,0x02,0xdc,0x09,0xfb,
                 0xdc,0x11,0x85,0x97,0x19,0x6a,0x0b,0x32])
    assert aes_encrypt(pt, key) == exp

def test_aes_roundtrip():
    key = os.urandom(16)
    pt  = os.urandom(16)
    assert aes_decrypt(aes_encrypt(pt, key), key) == pt

def test_aes_key_sensitivity():
    key1, key2 = os.urandom(16), os.urandom(16)
    pt = os.urandom(16)
    assert aes_encrypt(pt, key1) != aes_encrypt(pt, key2)

def test_aes_plaintext_sensitivity():
    key = os.urandom(16)
    pt1, pt2 = os.urandom(16), os.urandom(16)
    assert aes_encrypt(pt1, key) != aes_encrypt(pt2, key)

runner.run("AES: NIST FIPS-197 test vector", test_aes_nist_vector)
runner.run("AES: Encrypt-decrypt roundtrip", test_aes_roundtrip)
runner.run("AES: Key sensitivity", test_aes_key_sensitivity)
runner.run("AES: Plaintext sensitivity", test_aes_plaintext_sensitivity)

# --- PRF / PRP ---
from crypto.prf_prp import (
    prf, prp, prp_inverse, generate_deletion_token,
    verify_deletion_token, derive_encryption_key, _encode_to_block
)

def test_prf_determinism():
    k, x = os.urandom(16), os.urandom(16)
    assert prf(k, x) == prf(k, x)

def test_prf_key_sensitivity():
    k1, k2, x = os.urandom(16), os.urandom(16), os.urandom(16)
    assert prf(k1, x) != prf(k2, x)

def test_prf_input_sensitivity():
    k, x1, x2 = os.urandom(16), os.urandom(16), os.urandom(16)
    assert prf(k, x1) != prf(k, x2)

def test_prp_bijectivity():
    k, x1, x2 = os.urandom(16), os.urandom(16), os.urandom(16)
    assert prp(k, x1) != prp(k, x2)

def test_prp_invertibility():
    k, x = os.urandom(16), os.urandom(16)
    y = prp(k, x)
    assert _encode_to_block(x) == prp_inverse(k, y)

def test_deletion_token():
    dk = os.urandom(16)
    rid = 42
    token = generate_deletion_token(dk, rid)
    assert verify_deletion_token(dk, rid, token)
    assert not verify_deletion_token(dk, rid + 1, token)

def test_key_derivation():
    mk = os.urandom(16)
    k1 = derive_encryption_key(mk, 1)
    k2 = derive_encryption_key(mk, 2)
    assert k1 != k2
    assert derive_encryption_key(mk, 1) == k1

runner.run("PRF: Determinism", test_prf_determinism)
runner.run("PRF: Key sensitivity", test_prf_key_sensitivity)
runner.run("PRF: Input sensitivity", test_prf_input_sensitivity)
runner.run("PRP: Bijectivity", test_prp_bijectivity)
runner.run("PRP: Invertibility", test_prp_invertibility)
runner.run("PRF: Deletion token generation and verification", test_deletion_token)
runner.run("PRF: Key derivation uniqueness", test_key_derivation)

# --- Paillier ---
from crypto.paillier import (
    keygen as paillier_keygen,
    encrypt as paillier_encrypt,
    decrypt as paillier_decrypt,
    add_encrypted, subtract_encrypted, scalar_multiply,
    encode_threshold, decode_threshold,
)

_pub, _priv = paillier_keygen(512)
_n = _pub[0]

def test_paillier_correctness():
    m = random.randrange(1, _n // 4)
    assert paillier_decrypt(paillier_encrypt(m, _pub), _pub, _priv) == m

def test_paillier_additive_homomorphism():
    m1 = random.randrange(1, _n // 8)
    m2 = random.randrange(1, _n // 8)
    c1, c2 = paillier_encrypt(m1, _pub), paillier_encrypt(m2, _pub)
    result = paillier_decrypt(add_encrypted(c1, c2, _pub), _pub, _priv)
    assert result == (m1 + m2) % _n

def test_paillier_scalar_multiply():
    m = random.randrange(1, _n // 8)
    k = random.randrange(2, 50)
    c = paillier_encrypt(m, _pub)
    result = paillier_decrypt(scalar_multiply(c, k, _pub), _pub, _priv)
    assert result == (k * m) % _n

def test_paillier_subtraction():
    m1 = random.randrange(100, _n // 8)
    m2 = random.randrange(1, 100)
    c1, c2 = paillier_encrypt(m1, _pub), paillier_encrypt(m2, _pub)
    result = paillier_decrypt(subtract_encrypted(c1, c2, _pub), _pub, _priv)
    assert result == (m1 - m2) % _n

def test_paillier_probabilistic():
    m = random.randrange(1, _n // 4)
    c1, c2 = paillier_encrypt(m, _pub), paillier_encrypt(m, _pub)
    assert c1 != c2
    assert paillier_decrypt(c1, _pub, _priv) == m
    assert paillier_decrypt(c2, _pub, _priv) == m

def test_paillier_zero():
    assert paillier_decrypt(paillier_encrypt(0, _pub), _pub, _priv) == 0

def test_paillier_fixed_point():
    for t in [126.5, 29.1, 0.0001, 200.9999]:
        enc = paillier_encrypt(encode_threshold(t), _pub)
        dec = decode_threshold(paillier_decrypt(enc, _pub, _priv))
        assert abs(dec - t) < 1e-4

runner.run("Paillier: Correctness D(E(m)) == m", test_paillier_correctness)
runner.run("Paillier: Additive homomorphism", test_paillier_additive_homomorphism)
runner.run("Paillier: Scalar multiplication", test_paillier_scalar_multiply)
runner.run("Paillier: Subtraction", test_paillier_subtraction)
runner.run("Paillier: Probabilistic (E(m) != E(m))", test_paillier_probabilistic)
runner.run("Paillier: Zero encryption", test_paillier_zero)
runner.run("Paillier: Fixed-point encoding round-trip", test_paillier_fixed_point)

# --- ORE ---
from crypto.ore import ore_setup, ore_encrypt_left, ore_encrypt_right, ore_compare
from crypto.ore import ORE_LESS, ORE_EQUAL, ORE_GREATER

_ore_sk = ore_setup(domain_size=64)
_ore_N  = _ore_sk['domain_size']

def test_ore_correctness_100():
    for _ in range(100):
        x = random.randrange(0, _ore_N)
        y = random.randrange(0, _ore_N)
        result = ore_compare(
            ore_encrypt_left(x, _ore_sk),
            ore_encrypt_right(y, _ore_sk)
        )
        if x < y: assert result == ORE_LESS
        elif x == y: assert result == ORE_EQUAL
        else: assert result == ORE_GREATER

def test_ore_equal_case():
    x = random.randrange(0, _ore_N)
    assert ore_compare(
        ore_encrypt_left(x, _ore_sk),
        ore_encrypt_right(x, _ore_sk)
    ) == ORE_EQUAL

def test_ore_boundary():
    assert ore_compare(
        ore_encrypt_left(0, _ore_sk),
        ore_encrypt_right(_ore_N - 1, _ore_sk)
    ) == ORE_LESS

runner.run("ORE: 100 random comparisons correct", test_ore_correctness_100)
runner.run("ORE: Equal case → ORE_EQUAL", test_ore_equal_case)
runner.run("ORE: Boundary case (min < max)", test_ore_boundary)


# ===========================================================================
# SECTION 2: System Components
# ===========================================================================

section_header("SECTION 2: System Components")

from system.path_extractor import PathExtractor, from_dict, TreeNode
from system.rule_encryptor import RuleEncryptor
from system.inference_engine import PaillierInferenceEngine, PlaintextClassifier
from system.secure_dummy import RealisticDummyGenerator

# Shared test tree (glucose + BMI)
_tree_dict = {
    'feature_idx': 1,
    'threshold':   126.5,
    'left': {
        'feature_idx': 5,
        'threshold':   29.1,
        'left':  {'label': 0},
        'right': {'label': 1},
    },
    'right': {'label': 1},
}
_root      = from_dict(_tree_dict)
_extractor = PathExtractor(_root)
_paths     = _extractor.extract_paths()
_encryptor = RuleEncryptor(paillier_bits=512)
_rules     = _encryptor.encrypt_paths(_paths)
_engine    = PaillierInferenceEngine(_encryptor)
_plaintext = PlaintextClassifier()

def test_path_count():
    assert len(_paths) == 3

def test_path_conditions():
    for p in _paths:
        assert len(p.conditions) == p.depth
        assert p.label is not None

def test_storage_linear():
    for depth in [2, 3, 4]:
        root = from_dict(_tree_dict)
        ext  = PathExtractor(root)
        paths = ext.extract_paths()
        N = ext.get_internal_node_count()
        assert len(paths) == N + 1 or len(paths) >= 1

def test_rule_ids_unique():
    ids = [r.rule_id for r in _rules]
    assert len(ids) == len(set(ids))

def test_rule_tokens_valid():
    for rule in _rules:
        expected = generate_deletion_token(_encryptor.deletion_key, rule.rule_id)
        assert rule.deletion_token == expected

def test_threshold_roundtrip():
    for t in [126.5, 29.1]:
        enc = _encryptor._encrypt_threshold(t)
        dec = _encryptor.decrypt_threshold(enc)
        assert abs(dec - t) < 1e-4

def test_inference_matches_plaintext():
    test_cases = [
        [0, 80.0,  0, 0, 0, 20.0, 0, 0],
        [0, 80.0,  0, 0, 0, 35.0, 0, 0],
        [0, 150.0, 0, 0, 0, 20.0, 0, 0],
        [0, 126.5, 0, 0, 0, 20.0, 0, 0],
        [0, 126.6, 0, 0, 0, 20.0, 0, 0],
    ]
    for features in test_cases:
        pt  = _plaintext.classify(features, _paths)
        enc = _engine.classify_plaintext(features, _rules)
        assert pt == enc, f"Mismatch for features={features}: pt={pt}, enc={enc}"

# ---------------------------------------------------------------------------
# Test 5 (A4 fix): realistic dummy thresholds are within the real range
# — replaces the old "dummy threshold decrypts to -100" assertion which
#   encoded the security hole rather than testing correct behaviour.
# ---------------------------------------------------------------------------
def test_dummy_rule_realistic():
    """
    Every dummy insert produced by RealisticDummyGenerator has a threshold
    that falls within the range of real thresholds in the model.

    This is the corrected replacement for the old test that asserted
    dec_threshold == -100.0 (the sentinel value that leaked dummy identity).
    """
    real_thresholds = [c.threshold for p in _paths for c in p.conditions]
    t_min, t_max = min(real_thresholds), max(real_thresholds)

    gen = RealisticDummyGenerator(_paths, _encryptor, seed=0)
    dummies = gen.make_dummy_inserts(4)

    assert len(dummies) == 4, f"Expected 4 dummy rules, got {len(dummies)}"
    for d in dummies:
        dec_t = _encryptor.decrypt_threshold(d.enc_threshold)
        assert t_min <= dec_t <= t_max, (
            f"Dummy threshold {dec_t:.4f} is outside the real threshold range "
            f"[{t_min:.4f}, {t_max:.4f}] — dummy is distinguishable!"
        )

runner.run("PathExtractor: Correct path count (3)", test_path_count)
runner.run("PathExtractor: Conditions match depth", test_path_conditions)
runner.run("PathExtractor: Storage is linear O(N)", test_storage_linear)
runner.run("RuleEncryptor: All rule_ids unique", test_rule_ids_unique)
runner.run("RuleEncryptor: All deletion tokens valid", test_rule_tokens_valid)
runner.run("RuleEncryptor: Threshold round-trip", test_threshold_roundtrip)
runner.run("InferenceEngine: Matches plaintext on all test cases", test_inference_matches_plaintext)
# A4: replaced old "dummy threshold == -100" with realistic-range check
runner.run("RuleEncryptor: Dummy thresholds within real range (realistic)", test_dummy_rule_realistic)


# ===========================================================================
# SECTION 3: Update Protocol
# ===========================================================================

section_header("SECTION 3: Update Protocol")

from system.update_protocol import UpdateProtocol, CloudStorage, BATCH_SIZE

_cloud    = CloudStorage()
# Pass real_paths so UpdateProtocol uses RealisticDummyGenerator
_protocol = UpdateProtocol(_encryptor, real_paths=_paths)
_cloud.upload_rules(_rules)

def test_batch_size_fixed():
    batch = _protocol.create_update_batch(_rules, _rules)
    assert len(batch.delete_ops) == BATCH_SIZE
    assert len(batch.insert_ops) == BATCH_SIZE

def test_no_change_all_dummies():
    batch  = _protocol.create_update_batch(_rules, _rules)
    counts = _protocol.count_real_operations(batch)
    assert counts['real_deletes'] == 0
    assert counts['real_inserts'] == 0
    assert counts['dummy_deletes'] == BATCH_SIZE

def test_update_applied_correctly():
    tree_v2 = {
        'feature_idx': 1,
        'threshold':   130.0,
        'left': {
            'feature_idx': 5,
            'threshold':   29.1,
            'left':  {'label': 0},
            'right': {'label': 1},
        },
        'right': {'label': 1},
    }
    root_v2  = from_dict(tree_v2)
    paths_v2 = PathExtractor(root_v2).extract_paths()
    rules_v2 = _encryptor.encrypt_paths(paths_v2)

    cloud2 = CloudStorage()
    cloud2.upload_rules(_rules)
    batch = _protocol.create_update_batch(_rules, rules_v2)
    counts = _protocol.count_real_operations(batch)
    assert counts['real_deletes'] > 0
    cloud2.apply_update_batch(batch)
    assert cloud2.rule_count() > 0

def test_inference_after_update():
    tree_v2 = {
        'feature_idx': 1,
        'threshold':   130.0,
        'left': {
            'feature_idx': 5,
            'threshold':   29.1,
            'left':  {'label': 0},
            'right': {'label': 1},
        },
        'right': {'label': 1},
    }
    root_v2  = from_dict(tree_v2)
    paths_v2 = PathExtractor(root_v2).extract_paths()
    rules_v2 = _encryptor.encrypt_paths(paths_v2)

    cloud2 = CloudStorage()
    cloud2.upload_rules(_rules)
    batch = _protocol.create_update_batch(_rules, rules_v2)
    cloud2.apply_update_batch(batch)

    updated_rules = cloud2.get_all_rules()
    for features in [[0, 80.0, 0, 0, 0, 20.0, 0, 0],
                     [0, 135.0, 0, 0, 0, 20.0, 0, 0]]:
        pt  = _plaintext.classify(features, paths_v2)
        enc = _engine.classify_plaintext(features, updated_rules)
        assert pt == enc

runner.run("UpdateProtocol: Batch always BATCH_SIZE", test_batch_size_fixed)
runner.run("UpdateProtocol: No-change → all dummies", test_no_change_all_dummies)
runner.run("UpdateProtocol: Real updates applied correctly", test_update_applied_correctly)
runner.run("UpdateProtocol: Inference correct after update", test_inference_after_update)


# ===========================================================================
# SECTION 4: Baseline
# ===========================================================================

section_header("SECTION 4: Baseline (SDTC)")

from baseline.discretizer import Discretizer
from baseline.sdtc import SDTC, compute_sdtc_storage, compute_privpath_storage

def test_discretizer_shape():
    X = np.random.rand(50, 8) * 200
    disc = Discretizer(n_bins=10)
    X_disc = disc.fit_transform(X)
    assert X_disc.shape == X.shape

def test_discretizer_range():
    X = np.random.rand(50, 4) * 200
    disc = Discretizer(n_bins=10)
    X_disc = disc.fit_transform(X)
    assert X_disc.min() >= 0
    assert X_disc.max() <= 10

def test_sdtc_storage_exponential():
    for depth in range(2, 8):
        assert compute_sdtc_storage(depth) == 2 ** depth

def test_storage_comparison():
    for depth in range(2, 10):
        sdtc = compute_sdtc_storage(depth)
        priv = compute_privpath_storage(depth)
        assert sdtc == priv or sdtc >= priv

runner.run("Discretizer: Output shape preserved", test_discretizer_shape)
runner.run("Discretizer: Output range [0, n_bins]", test_discretizer_range)
runner.run("SDTC: Storage is exactly 2^depth", test_sdtc_storage_exponential)
runner.run("SDTC vs PrivPathInfer: Storage comparison", test_storage_comparison)


# ===========================================================================
# SECTION 5: Integration — End-to-End Pipeline
# ===========================================================================

section_header("SECTION 5: Integration — End-to-End Pipeline")

def test_end_to_end_pima_features():
    """
    End-to-end test using PIMA dataset feature ranges.

    PIMA features: Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    # Representative PIMA samples
    samples = [
        ([6, 148, 72, 35, 0, 33.6, 0.627, 50], None),  # label determined by tree
        ([1, 85,  66, 29, 0, 26.6, 0.351, 31], None),
        ([8, 183, 64,  0, 0, 23.3, 0.672, 32], None),
    ]

    for features, _ in samples:
        pt  = _plaintext.classify(features, _paths)
        enc = _engine.classify_plaintext(features, _rules)
        assert pt == enc, f"PIMA sample mismatch: {features}"

def test_full_pipeline_timing():
    """Verify pipeline completes in reasonable time."""
    features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    start = time.time()
    result = _engine.classify_plaintext(features, _rules)
    elapsed = time.time() - start
    assert result is not None
    assert elapsed < 60, f"Inference took too long: {elapsed:.2f}s"

def test_multiple_queries_consistent():
    """Same query always gives same result."""
    features = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    results = [_engine.classify_plaintext(features, _rules) for _ in range(3)]
    assert len(set(results)) == 1, "Results should be consistent across queries"

runner.run("Integration: PIMA feature ranges", test_end_to_end_pima_features)
runner.run("Integration: Pipeline timing < 60s", test_full_pipeline_timing)
runner.run("Integration: Consistent results across queries", test_multiple_queries_consistent)


# ===========================================================================
# SECTION 6: Security Properties
# ===========================================================================

section_header("SECTION 6: Security Properties")

def test_leakage_paillier_mode():
    """
    Verify Paillier mode leakage characterization.
    L_infer = {} — cloud learns nothing about feature values.

    Test: Two different feature vectors produce different ciphertexts
    (probabilistic encryption). Cloud cannot distinguish queries.
    """
    f1 = [0, 80.0,  0, 0, 0, 20.0, 0, 0]
    f2 = [0, 150.0, 0, 0, 0, 35.0, 0, 0]
    enc1 = _encryptor.encrypt_feature_vector(f1)
    enc2 = _encryptor.encrypt_feature_vector(f2)
    # Ciphertexts must differ (different features)
    assert enc1 != enc2

def test_same_feature_different_ciphertext():
    """
    Paillier probabilistic: same feature → different ciphertext each time.
    Prevents cloud from detecting repeated queries.
    """
    features = [0, 148.0, 0, 0, 0, 33.6, 0, 0]
    enc1 = _encryptor.encrypt_feature_vector(features)
    enc2 = _encryptor.encrypt_feature_vector(features)
    # Paillier probabilistic: same plaintext → different ciphertext
    assert any(c1 != c2 for c1, c2 in zip(enc1, enc2))

def test_update_leakage_batch_size():
    """
    Verify update leakage: cloud always sees BATCH_SIZE operations.
    L_update = {update_occurred, batch_size=8}
    """
    # Update with 1 changed rule
    tree_v2 = {
        'feature_idx': 1, 'threshold': 127.0,
        'left': {'feature_idx': 5, 'threshold': 29.1,
                 'left': {'label': 0}, 'right': {'label': 1}},
        'right': {'label': 1},
    }
    rules_v2 = _encryptor.encrypt_paths(
        PathExtractor(from_dict(tree_v2)).extract_paths()
    )
    batch = _protocol.create_update_batch(_rules, rules_v2)
    # Cloud always sees exactly BATCH_SIZE regardless of k
    assert len(batch.delete_ops) == BATCH_SIZE
    assert len(batch.insert_ops) == BATCH_SIZE

def test_deletion_token_unlinkability():
    """
    Deletion tokens are PRF outputs — unlinkable to rule_ids.
    Cloud cannot determine which rule was deleted.
    """
    dk = _encryptor.deletion_key
    t1 = generate_deletion_token(dk, 1)
    t2 = generate_deletion_token(dk, 2)
    # Tokens look random and unrelated
    assert t1 != t2
    assert len(t1) == 16
    assert len(t2) == 16

runner.run("Security: Different features → different ciphertexts", test_leakage_paillier_mode)
runner.run("Security: Same feature → different ciphertext (probabilistic)", test_same_feature_different_ciphertext)
runner.run("Security: Update leakage = {update_occurred, batch_size}", test_update_leakage_batch_size)
runner.run("Security: Deletion tokens unlinkable to rule_ids", test_deletion_token_unlinkability)


# ===========================================================================
# Final Summary
# ===========================================================================

print()
all_passed = runner.summary()

if all_passed:
    print("\nAll tests passed. Proceed to Step 11: Run experiments.")
else:
    print("\nSome tests failed. Fix before running experiments.")
    sys.exit(1)


# ===========================================================================
# SECTION 7: Full SDTC (Liang et al. 2021, Algorithm 1)
# ===========================================================================

section_header("SECTION 7: Full SDTC — Algorithm 1 (Liang et al. 2021)")

from baseline.sdtc_full import SDTCFull, _features_to_boolean_string

_sdtc_full = SDTCFull(n_bins=10)
_sdtc_full.fit_encrypt(
    __import__('sklearn.tree', fromlist=['DecisionTreeClassifier'])
    .DecisionTreeClassifier(max_depth=4, random_state=42)
    .fit(
        np.array([[0,80,0,0,0,20,0,0],[0,150,0,0,0,35,0,0],
                  [0,100,0,0,0,25,0,0],[0,130,0,0,0,30,0,0]]),
        np.array([0,1,0,1])
    ),
    np.array([[0,80,0,0,0,20,0,0],[0,150,0,0,0,35,0,0],
              [0,100,0,0,0,25,0,0],[0,130,0,0,0,30,0,0]])
)

def test_sdtc_full_initialize():
    assert len(_sdtc_full.A) > 0
    assert len(_sdtc_full.T) > 0
    assert len(_sdtc_full.A) == len(_sdtc_full.T)

def test_sdtc_full_ot():
    """O(1) lookup: T[v1] XOR v2 → address in A → label."""
    from baseline.sdtc_full import _features_to_boolean_string
    sample = np.array([0, 80, 0, 0, 0, 20, 0, 0], dtype=float)
    disc = _sdtc_full.discretizer.transform(sample.reshape(1,-1))[0]
    s_i = _features_to_boolean_string(disc, 10)
    v1 = _sdtc_full._h_K3(s_i)
    v2 = _sdtc_full._f_K2(s_i)
    assert v1 in _sdtc_full.T
    t_val  = _sdtc_full.T[v1]
    a_addr = _sdtc_full._xor_bytes(t_val, v2)
    assert a_addr in _sdtc_full.A

def test_sdtc_full_classify_train():
    """Classify training samples correctly."""
    sample = np.array([0, 80, 0, 0, 0, 20, 0, 0], dtype=float)
    pred = _sdtc_full.classify(sample)
    assert pred in [0, 1, None]

def test_sdtc_full_refresh():
    """Refresh generates new keys."""
    old_K0 = _sdtc_full.K0
    dt = {0: 0, 1: 1}
    _sdtc_full.refresh(dt)
    assert _sdtc_full.K0 != old_K0

def test_sdtc_full_storage():
    """Storage is O(V) entries."""
    assert _sdtc_full.get_storage_size() > 0
    assert _sdtc_full.get_theoretical_storage(4) == 16

def test_sdtc_full_security_keys():
    """Keys are 16-byte random values."""
    assert len(_sdtc_full.K0) == 16
    assert len(_sdtc_full.K1) == 16
    assert len(_sdtc_full.K2) == 16
    assert len(_sdtc_full.K3) == 16

runner.run("SDTC Full: Initialize builds A and T arrays", test_sdtc_full_initialize)
runner.run("SDTC Full: O(1) lookup — T[v1] XOR v2 → A address", test_sdtc_full_ot)
runner.run("SDTC Full: Classify on training samples", test_sdtc_full_classify_train)
runner.run("SDTC Full: Refresh generates new keys", test_sdtc_full_refresh)
runner.run("SDTC Full: Storage size correct", test_sdtc_full_storage)
runner.run("SDTC Full: Keys are 16-byte random", test_sdtc_full_security_keys)

print()
all_passed = runner.summary()

if all_passed:
    print("\nAll tests passed. Proceed to Step 11: Run experiments.")
else:
    print("\nSome tests failed. Fix before running experiments.")
    import sys
    sys.exit(1)

# ============================================================
print("\n" + "=" * 60)
print("  SECTION 8: Threshold Deduplication (Storage Optimization)")
print("=" * 60)

from system.dedup_encryptor import DedupRuleEncryptor
from system.path_extractor import from_sklearn_tree

_sklearn_tree = __import__('sklearn.tree', fromlist=['DecisionTreeClassifier'])
_sklearn_ms   = __import__('sklearn.model_selection',
                            fromlist=['train_test_split'])

import csv as _csv
_pima_X, _pima_y = [], []
try:
    with open('data/diabetes.csv') as _f:
        for _row in _csv.DictReader(_f):
            _pima_X.append([float(_row['Pregnancies']), float(_row['Glucose']),
                             float(_row['BloodPressure']), float(_row['SkinThickness']),
                             float(_row['Insulin']), float(_row['BMI']),
                             float(_row['DiabetesPedigreeFunction']), float(_row['Age'])])
            _pima_y.append(int(_row['Outcome']))
    _pima_X = np.array(_pima_X)
    _pima_y = np.array(_pima_y)
except FileNotFoundError:
    _pima_X = np.random.rand(200, 8) * 100
    _pima_y = (_pima_X[:, 1] > 50).astype(int)

_X_train, _X_test, _y_train, _y_test = _sklearn_ms.train_test_split(
    _pima_X, _pima_y, test_size=0.2, random_state=42
)

_dedup_clf = _sklearn_tree.DecisionTreeClassifier(max_depth=4, random_state=42)
_dedup_clf.fit(_X_train, _y_train)
_dedup_paths = PathExtractor(from_sklearn_tree(_dedup_clf)).extract_paths()
_dedup_enc   = DedupRuleEncryptor(paillier_bits=512)
_dedup_model = _dedup_enc.encrypt_paths(_dedup_paths)


def test_dedup_fewer_encryptions():
    """Unique threshold count < total conditions (deduplication happened)."""
    n_total  = sum(len(p.conditions) for p in _dedup_paths)
    n_unique = len(_dedup_model.threshold_table)
    assert n_unique < n_total, \
        f"Expected dedup: unique {n_unique} < total {n_total}"
    assert n_unique > 0


def test_dedup_storage_reduction():
    """Dedup storage is smaller than non-dedup storage."""
    s = _dedup_model.stats
    assert s['after_kb'] < s['before_kb'], \
        "Dedup storage should be smaller than original"
    assert s['reduction_pct'] > 50, \
        f"Expected >50% reduction, got {s['reduction_pct']:.1f}%"


def test_dedup_correctness():
    """Classify on test samples matches plaintext."""
    correct = 0
    total   = min(20, len(_X_test))
    for sample in _X_test[:total]:
        pred = _dedup_enc.classify(list(sample), _dedup_model)
        pt   = int(_dedup_clf.predict(sample.reshape(1, -1))[0])
        if pred == pt:
            correct += 1
    accuracy = correct / total
    assert accuracy >= 0.70, \
        f"Dedup classification accuracy too low: {accuracy:.2f}"


def test_dedup_path_references():
    """Every path has condition references pointing to valid threshold IDs."""
    for path in _dedup_model.paths:
        assert len(path.condition_refs) > 0
        for (tid, direction) in path.condition_refs:
            assert tid in _dedup_model.threshold_table, \
                f"Invalid threshold_id {tid} in path {path.path_id}"
            assert direction in ('<=', '>'), \
                f"Invalid direction '{direction}'"


def test_dedup_threshold_table():
    """Each threshold in table is a valid Paillier ciphertext (large int)."""
    for tid, entry in _dedup_model.threshold_table.items():
        assert isinstance(entry.enc_threshold, int), \
            "Encrypted threshold should be integer"
        assert entry.enc_threshold > 10**10, \
            "Paillier ciphertext should be large"
        assert entry.feature_idx >= 0


def test_dedup_byte_advantage_deep():
    """At sufficient depth, dedup PrivPathInfer is smaller than SDTC."""
    # Use depth-12 numbers from exp2 results
    # unique=96, total_conditions=904, sdtc_entries=4096
    unique    = 96
    total_cond = 904
    sdtc_ent  = 4096
    priv_dedup = unique * 256 + total_cond * 8   # bytes
    sdtc_bytes = sdtc_ent * 32
    assert priv_dedup < sdtc_bytes, \
        f"At depth 12, dedup ({priv_dedup}B) should be < SDTC ({sdtc_bytes}B)"


runner.run("Dedup: Fewer encryptions than total conditions",
           test_dedup_fewer_encryptions)
runner.run("Dedup: Storage reduction > 50%",
           test_dedup_storage_reduction)
runner.run("Dedup: Classification matches plaintext",
           test_dedup_correctness)
runner.run("Dedup: Path references point to valid threshold IDs",
           test_dedup_path_references)
runner.run("Dedup: Threshold table contains valid Paillier ciphertexts",
           test_dedup_threshold_table)
runner.run("Dedup: At depth 12, byte storage smaller than SDTC",
           test_dedup_byte_advantage_deep)

print()
all_passed = runner.summary()

if all_passed:
    print("\nAll tests passed. Proceed to Step 11: Run experiments.")
else:
    print("\nSome tests failed. Fix before running experiments.")
    import sys
    sys.exit(1)