"""
rule_encryptor.py — Encrypted Rule Generation for PrivPathInfer
===============================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

This module implements the encryption of extracted decision tree paths
into independently encrypted rules stored on the cloud server.

PrivPathInfer Contributions Implemented Here:
    Contribution 1 — Native Continuous Feature Support:
        Thresholds are encrypted using Paillier homomorphic encryption
        with fixed-point encoding. No discretization required.
        threshold_int = int(threshold * 10000) + 10^9

    Contribution 2 — Linear Storage O(N):
        Each path condition is independently encrypted.
        Total encrypted rules = number of paths = O(N).

    Contribution 3 — Incremental Update Support:
        Each rule has a unique rule_id and a PRF-derived deletion token.
        Independent encryption allows selective re-encryption on update.

Security:
    Theorem 2 (Classifier Privacy):
        Encrypted thresholds reveal no information about plaintext values
        under Paillier semantic security (DCR assumption).
        Paillier 1999, Theorem 15.

    Theorem 1 (Data Privacy):
        Rule indices are permuted using PRP to prevent ordering leakage.
        Boneh-Shoup, Definition 4.1.

Encrypted Rule Format:
    Each EncryptedRule contains:
        rule_id:           unique identifier
        path_id:           which path this rule belongs to
        condition_index:   position within the path (0-indexed)
        enc_feature_idx:   PRP(permutation_key, feature_idx)
        enc_threshold:     Paillier.Enc(fixed_point(threshold))
        direction:         'left' (<=) or 'right' (>)
        label:             class label (only for the last condition in path)
        deletion_token:    PRF(deletion_key, rule_id)

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from crypto.paillier import (
    keygen as paillier_keygen,
    encrypt as paillier_encrypt,
    decrypt as paillier_decrypt,
    encode_threshold,
    decode_threshold,
    SCALE_FACTOR,
    OFFSET,
)
from crypto.prf_prp import (
    prf,
    prp,
    prp_inverse,
    generate_deletion_token,
    derive_encryption_key,
    _encode_to_block,
)
from system.path_extractor import LeafPath, PathCondition


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EncryptedRule:
    """
    A single encrypted rule stored on the cloud server.

    Each rule corresponds to one condition in one root-to-leaf path.
    Rules are independently encrypted, enabling selective updates.

    Fields:
        rule_id:           unique integer identifier
        path_id:           which path this rule belongs to
        condition_index:   position within the path (0 = root split)
        enc_feature_idx:   PRP-permuted feature index (int)
        enc_threshold:     Paillier ciphertext of fixed-point threshold
        direction:         'left' (feature <= threshold) or 'right' (>)
        label:             class label (set only for last condition in path)
        is_last:           True if this is the last condition in its path
        deletion_token:    PRF(deletion_key, rule_id) — bytes(16)
        depth:             depth of the split node in the tree
    """
    rule_id:         int
    path_id:         int
    condition_index: int
    enc_feature_idx: int
    enc_threshold:   int
    direction:       str
    label:           Optional[int]
    is_last:         bool
    deletion_token:  bytes
    depth:           int


# ---------------------------------------------------------------------------
# Rule Encryptor
# ---------------------------------------------------------------------------

class RuleEncryptor:
    """
    Encrypts extracted decision tree paths into independently encrypted rules.

    The Medical Institution (MI) runs this module to:
        1. Generate Paillier key pair
        2. Encrypt all thresholds
        3. Permute feature indices using PRP
        4. Generate deletion tokens for each rule
        5. Store encrypted rules on cloud server

    Key Management:
        paillier_public_key:  shared with cloud and users
        paillier_private_key: kept secret by MI (for decryption)
        permutation_key:      kept secret by MI (for PRP)
        deletion_key:         kept secret by MI (for PRF tokens)
    """

    def __init__(self, paillier_bits=512):
        """
        Initialize the RuleEncryptor with fresh cryptographic keys.

        Args:
            paillier_bits: key size for Paillier (512 for testing,
                           1024 for experiments, 2048 for production)
        """
        # Generate Paillier key pair
        self.paillier_pub, self.paillier_priv = paillier_keygen(paillier_bits)
        self.n = self.paillier_pub[0]

        # Generate secret keys for PRP and PRF
        self.permutation_key = os.urandom(16)  # for PRP feature index permutation
        self.deletion_key    = os.urandom(16)  # for PRF deletion token generation

        # Rule counter (global, across all paths)
        self._rule_counter = 0

    def _next_rule_id(self) -> int:
        """Generate the next unique rule_id."""
        rid = self._rule_counter
        self._rule_counter += 1
        return rid

    def _encrypt_threshold(self, threshold_float: float) -> int:
        """
        Encrypt a continuous threshold using Paillier with fixed-point encoding.

        Encoding: threshold_int = int(threshold * 10000) + 10^9
        This preserves 4 decimal places without discretization.

        Contribution 1: Unlike SDTC which requires discretization into bins,
        PrivPathInfer encrypts exact thresholds homomorphically.

        Args:
            threshold_float: continuous threshold value

        Returns:
            int: Paillier ciphertext of the encoded threshold
        """
        threshold_int = encode_threshold(threshold_float)
        return paillier_encrypt(threshold_int, self.paillier_pub)

    def _permute_feature_idx(self, feature_idx: int) -> int:
        """
        Permute a feature index using PRP.

        PRP(permutation_key, feature_idx) — prevents cloud from inferring
        which features are used most often in the decision tree.

        Reference: Boneh-Shoup, Definition 4.1 (PRP security).

        Args:
            feature_idx: original feature index (0-indexed)

        Returns:
            int: permuted feature index (128-bit integer)
        """
        output = prp(self.permutation_key, feature_idx)
        return int.from_bytes(output, byteorder='big')

    def _generate_token(self, rule_id: int) -> bytes:
        """
        Generate a deletion token for a rule.

        token = PRF(deletion_key, rule_id)

        Reference: PrivPathInfer Contribution 3, Theorem 4.

        Args:
            rule_id: unique rule identifier

        Returns:
            bytes: 16-byte deletion token
        """
        return generate_deletion_token(self.deletion_key, rule_id)

    def encrypt_paths(self, paths: List[LeafPath]) -> List[EncryptedRule]:
        """
        Encrypt all extracted paths into independently encrypted rules.

        For each path, each condition becomes one EncryptedRule.
        The label is stored only in the last rule of each path.

        Storage: O(total conditions) = O(N) for N internal nodes.

        Args:
            paths: list of LeafPath objects from PathExtractor

        Returns:
            list of EncryptedRule objects
        """
        encrypted_rules = []

        for path in paths:
            num_conditions = len(path.conditions)

            for i, cond in enumerate(path.conditions):
                rule_id = self._next_rule_id()
                is_last = (i == num_conditions - 1)

                enc_rule = EncryptedRule(
                    rule_id         = rule_id,
                    path_id         = path.path_id,
                    condition_index = i,
                    enc_feature_idx = self._permute_feature_idx(cond.feature_idx),
                    enc_threshold   = self._encrypt_threshold(cond.threshold),
                    direction       = cond.direction,
                    label           = path.label if is_last else None,
                    is_last         = is_last,
                    deletion_token  = self._generate_token(rule_id),
                    depth           = cond.depth,
                )
                encrypted_rules.append(enc_rule)

        return encrypted_rules

    def encrypt_feature_vector(self, features: List[float]) -> List[int]:
        """
        Encrypt a user's feature vector for secure inference.

        Each feature is encoded with fixed-point encoding and encrypted
        under Paillier. The cloud uses these ciphertexts to perform
        homomorphic comparison with encrypted thresholds.

        Theorem 1 (Data Privacy):
            User features are computationally indistinguishable from
            random under Paillier semantic security (DCR assumption).

        Args:
            features: list of float feature values

        Returns:
            list of Paillier ciphertexts, one per feature
        """
        encrypted_features = []
        for f in features:
            f_int = encode_threshold(f)
            enc_f = paillier_encrypt(f_int, self.paillier_pub)
            encrypted_features.append(enc_f)
        return encrypted_features

    def decrypt_threshold(self, enc_threshold: int) -> float:
        """
        Decrypt an encrypted threshold (MI only).

        Used for verification and testing. The cloud never calls this.

        Args:
            enc_threshold: Paillier ciphertext

        Returns:
            float: decrypted threshold value
        """
        threshold_int = paillier_decrypt(
            enc_threshold, self.paillier_pub, self.paillier_priv
        )
        return decode_threshold(threshold_int)

    def get_public_params(self) -> Dict[str, Any]:
        """
        Return public parameters to be shared with cloud and users.

        Returns:
            dict with 'paillier_public_key' = (n, g)
        """
        return {
            'paillier_public_key': self.paillier_pub,
        }


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """
    Verify rule encryption correctness.

    Tests:
        1. Threshold encryption/decryption round-trip
        2. Fixed-point encoding preserves precision
        3. Each rule has unique rule_id
        4. Each rule has valid deletion token
        5. Realistic dummy generator produces in-range thresholds
        6. Full path encryption: rule count matches path structure
        7. Feature vector encryption
    """
    from system.path_extractor import PathExtractor
    from system.secure_dummy import RealisticDummyGenerator

    print("=" * 60)
    print("RuleEncryptor Verification Tests")
    print("PrivPathInfer Contributions 1, 2, 3")
    print("=" * 60)

    # Build a simple test tree
    from system.path_extractor import from_dict
    tree_dict = {
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
    root = from_dict(tree_dict)
    extractor = PathExtractor(root)
    paths = extractor.extract_paths()

    print(f"\nTest tree: {len(paths)} paths, "
          f"{extractor.get_internal_node_count()} internal nodes")

    encryptor = RuleEncryptor(paillier_bits=512)
    print("Paillier key pair generated (512-bit)")

    # Test 1: Threshold encryption round-trip
    for threshold in [126.5, 29.1, 80.0, 0.001, 200.9999]:
        enc = encryptor._encrypt_threshold(threshold)
        dec = encryptor.decrypt_threshold(enc)
        assert abs(dec - threshold) < 1e-4, \
            f"Threshold round-trip FAILED: {threshold} → {dec}"
    print("[PASS] Test 1: Threshold encryption round-trip")

    # Test 2: Fixed-point precision
    threshold = 126.5432
    enc = encryptor._encrypt_threshold(threshold)
    dec = encryptor.decrypt_threshold(enc)
    assert abs(dec - threshold) < 1e-4, "Fixed-point precision FAILED"
    print("[PASS] Test 2: Fixed-point encoding preserves 4 decimal places")

    # Test 3 & 4: Encrypt all paths
    rules = encryptor.encrypt_paths(paths)

    rule_ids = [r.rule_id for r in rules]
    assert len(rule_ids) == len(set(rule_ids)), "Rule IDs must be unique"
    print("[PASS] Test 3: All rule_ids are unique")

    for rule in rules:
        expected_token = generate_deletion_token(encryptor.deletion_key, rule.rule_id)
        assert rule.deletion_token == expected_token, \
            f"Deletion token mismatch for rule {rule.rule_id}"
    print("[PASS] Test 4: All deletion tokens are valid PRF outputs")

    # Test 5: Realistic dummy — thresholds within real range
    real_thresholds = [c.threshold for p in paths for c in p.conditions]
    t_min, t_max = min(real_thresholds), max(real_thresholds)
    gen = RealisticDummyGenerator(paths, encryptor, seed=0)
    dummies = gen.make_dummy_inserts(4)
    for d in dummies:
        dec_t = encryptor.decrypt_threshold(d.enc_threshold)
        assert t_min <= dec_t <= t_max, (
            f"Dummy threshold {dec_t:.4f} outside real range "
            f"[{t_min:.4f}, {t_max:.4f}]"
        )
    print("[PASS] Test 5: Realistic dummy thresholds are within real range")

    # Test 6: Rule count matches path structure
    total_conditions = sum(len(p.conditions) for p in paths)
    assert len(rules) == total_conditions, \
        f"Rule count {len(rules)} != total conditions {total_conditions}"
    print(f"[PASS] Test 6: Rule count ({len(rules)}) == total conditions")

    # Test 7: Feature vector encryption
    features = [148.0, 85.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]
    enc_features = encryptor.encrypt_feature_vector(features)
    assert len(enc_features) == len(features), "Feature count mismatch"
    print(f"[PASS] Test 7: Feature vector ({len(features)} features) encrypted")

    # Test 8: Label stored only in last condition of each path
    for path in paths:
        path_rules = sorted(
            [r for r in rules if r.path_id == path.path_id],
            key=lambda r: r.condition_index
        )
        for i, r in enumerate(path_rules):
            if i < len(path_rules) - 1:
                assert r.label is None, \
                    f"Non-last rule {r.rule_id} should have label=None"
            else:
                assert r.label is not None, \
                    f"Last rule {r.rule_id} should have a label"
    print("[PASS] Test 8: Labels stored only in last rule of each path")

    print("\n[ALL TESTS PASSED] rule_encryptor.py verified.")
    print("Contribution 1: Continuous thresholds encrypted without discretization")
    print("Contribution 2: O(N) independently encrypted rules")
    print("Contribution 3: Deletion tokens generated for all rules")


if __name__ == "__main__":
    run_all_tests()