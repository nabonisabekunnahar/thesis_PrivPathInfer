"""
inference_engine.py — Secure Inference Engine for PrivPathInfer
================================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

This module implements the core secure inference protocol in which a
semi-honest Cloud holds the encrypted model and a User holds the secret
key, and the two interact to classify the User's encrypted feature vector
without the Cloud learning the feature values, the thresholds, or the
diagnosis.

-----------------------------------------------------------------------
CORRECTION (this revision) — unblinded difference and the L_infer claim
-----------------------------------------------------------------------
An earlier revision of this module computed
    enc_diff = Enc(feature) (.) Enc(threshold)^{-1} = Enc(feature - threshold)
and handed the RAW (unblinded) ciphertext to whichever party performs the
decryption. Decrypting an unblinded difference reveals the EXACT integer
(feature - threshold) to the decrypting party. Anyone who also knows the
plaintext feature value can therefore recover the threshold exactly:
    threshold = feature - (feature - threshold).
This is fatal to Theorem 2 (Classifier Privacy) when the User is the party
who decrypts (the User trivially knows their own feature), and it directly
contradicts a bare "L_infer = {}" claim against that party.

The fix implemented here is multiplicative blinding before any value
derived from (feature - threshold) is exposed for decryption:
    enc_blinded = Enc(feature - threshold)^r = Enc(r * (feature - threshold))
for a fresh random positive r sampled by the CLOUD per condition per query.
Decrypting enc_blinded reveals only the SIGN of (feature - threshold) (since
r > 0 preserves sign under the centered representation of Z_n), not its
magnitude. This restores the comparison-bit-only leakage in a single,
non-adaptive query.

Honest residual leakage (stated precisely, not hidden):
    1. The User who decrypts learns the per-node comparison BIT for their
       own query (which the protocol must reveal for classification to
       work at all) and the final label. This is intentional, not a flaw.
    2. The User does NOT learn the threshold magnitude from one query
       (blinding hides it). An ADAPTIVE user issuing many queries that
       vary one feature could binary-search a threshold across queries —
       this is a multi-query leakage channel that blinding alone does not
       close; it requires query rate-limiting (the same caveat Tai et al.
       2017 note for their own DGK-based comparison).
    3. The CLOUD, which never holds the secret key in the two-party
       protocol (see CloudParty below), learns nothing about plaintext
       feature values, thresholds, or the comparison result: its view is
       exclusively semantically-secure ciphertexts. L_infer = {} is
       defensible FOR THE CLOUD under this role separation.

Two Inference Modes:
    1. Paillier Mode (PRIMARY — fully implemented):
       - User encrypts features under Paillier
       - Cloud performs homomorphic subtraction AND blinding
       - 2 communication rounds
       - Leakage:
           against the Cloud:        L_infer^Cloud = {}
           against the decrypting
           User (single query):      L_infer^User  = {comparison_bits, label}
       - Formally proven secure (Theorem 1, Theorem 2, Theorem 3 below)

    2. ORE Mode (THEORETICAL — discussion only):
       - User encrypts features under ORE left encryption
       - Cloud compares directly using ORE.Compare
       - 1 communication round
       - Leakage: L_infer = {comparison_results}
       - NOT fully implemented (see ore.py)

Security Theorems (restated precisely; see thesis Chapter III, Sec. 3.3 for
full proofs):
    Theorem 1 (Data Privacy):
        User features are computationally indistinguishable from random
        under the DCR assumption (Paillier IND-CPA). Holds against the
        Cloud unconditionally; the User is the data owner so this theorem
        is not about hiding data from the User.

    Theorem 2 (Classifier Privacy):
        Stored Enc(threshold) is IND-CPA-secure, so the CLOUD learns
        nothing about thresholds from storage or from the protocol
        transcript (it never decrypts). Against the decrypting USER,
        classifier privacy holds PROVIDED the difference is blinded before
        decryption (see CORRECTION above); it is a single-query,
        non-adaptive guarantee, not an unconditional one.

    Theorem 3 (Leakage Characterization):
        Paillier mode leakage against the Cloud:  L_infer = {}
        Paillier mode leakage against the User:   L_infer = {comparison
            bits for the queried record, final label} (irreducible — the
            User must learn these to receive a classification)
        ORE mode leakage:                          L_infer = {comparison_results}

Inference Protocol (Paillier Mode, 2 rounds, roles separated):
    Round 1 (User → Cloud):
        User sends encrypted features: {Enc(x_i)} for i = 1..n
        User holds the Paillier secret key; the Cloud holds only pk.

    Cloud Processing (CloudParty, no secret key):
        For each path p:
            For each condition c in path p:
                Compute Enc(x_{c.feature} - c.threshold) homomorphically
                BLIND it with a fresh random positive scalar r (per query,
                per condition) before sending: Enc(r * (x - threshold))

    Round 2 (Cloud → User):
        Cloud sends the blinded ciphertexts (and, for inference only,
        which path/condition each corresponds to — an access-pattern
        leakage explicitly out of scope per the thesis threat model).

    User (UserParty, holds sk):
        User decrypts each blinded value, reads off the SIGN only,
        checks all conditions for each path, returns the matching label.

A single-process PaillierInferenceEngine (below) is retained for
correctness/timing experiments (Experiments 1, 3, 5) where one process
plays both roles for measurement convenience; it now also applies blinding,
so its measured behaviour matches the deployed two-party protocol. The
explicit CloudParty / UserParty split makes the role separation that the
security claim depends on impossible to silently violate in new code.

Reference:
    Liang et al. 2021 (SDTC): O(1) inference but exponential storage
    PrivPathInfer: O(d·N) inference with O(N) storage
    Tai et al. 2017: notes the same adaptive-query caveat for DGK-style
        comparison protocols (rate-limiting required against an adaptive
        querier even though a single query leaks only the comparison bit).

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import random
from typing import List, Optional, Dict, Tuple, Any

from crypto.paillier import (
    encrypt as paillier_encrypt,
    decrypt as paillier_decrypt,
    add_encrypted,
    subtract_encrypted,
    scalar_multiply,
    negate_encrypted,
    encode_threshold,
    decode_threshold,
    OFFSET,
)
from crypto.prf_prp import prp_inverse, _encode_to_block
from system.rule_encryptor import EncryptedRule, RuleEncryptor


# ---------------------------------------------------------------------------
# Paillier Mode Inference
# ---------------------------------------------------------------------------

class PaillierInferenceEngine:
    """
    Secure inference engine using Paillier homomorphic encryption.

    Implements the 2-round secure inference protocol for PrivPathInfer.

    Protocol Overview:
        The cloud holds encrypted rules {EncryptedRule}.
        The user submits encrypted features.
        The cloud performs homomorphic subtraction to compute
        Enc(feature - threshold) for each condition.
        The sign of the decrypted result determines the comparison.

    Leakage Analysis (Theorem 3):
        The cloud sees only:
            - Encrypted features (semantically secure under DCR)
            - Encrypted thresholds (semantically secure under DCR)
            - Homomorphic intermediate values (also encrypted)
        The cloud learns nothing about plaintext values.
        L_infer = {} (empty leakage set)

    Note on Access Pattern:
        Access pattern leakage (which rules are accessed) is outside
        the security model. ORAM integration would address this and
        is stated as future work.
    """

    def __init__(self, encryptor: RuleEncryptor):
        """
        Initialize the inference engine with a RuleEncryptor.

        Args:
            encryptor: RuleEncryptor holding keys and encrypted rules
        """
        self.encryptor  = encryptor
        self.pub        = encryptor.paillier_pub
        self.priv       = encryptor.paillier_priv
        self.perm_key   = encryptor.permutation_key
        self.n          = encryptor.n

    def _recover_feature_idx(self, enc_feature_idx: int) -> int:
        """
        Recover original feature index from PRP-permuted value.

        Computes PRP^{-1}(permutation_key, enc_feature_idx).
        Only the MI can do this (holds permutation_key).

        Args:
            enc_feature_idx: PRP-permuted feature index (128-bit int)

        Returns:
            int: original feature index (0-indexed)
        """
        enc_bytes = enc_feature_idx.to_bytes(16, byteorder='big')
        original  = prp_inverse(self.perm_key, enc_bytes)
        return int.from_bytes(original, byteorder='big') % 256

    def _homomorphic_compare(
        self,
        enc_feature: int,
        enc_threshold: int,
        direction: str,
    ) -> int:
        """
        Homomorphic comparison: compute a BLINDED Enc(feature - threshold).

        SECURITY-CRITICAL (see module docstring "CORRECTION"): the raw
        difference Enc(feature - threshold) must never be handed to a party
        for decryption, because decrypting it reveals the exact integer
        (feature - threshold). Anyone who also knows `feature` then learns
        `threshold` exactly. This method therefore blinds the difference
        with a fresh random positive scalar r before returning it:

            Enc(feature - threshold)         [raw, INTERNAL only]
              -> scalar_multiply by r > 0
            Enc(r * (feature - threshold))   [returned — safe to decrypt]

        Multiplying by a positive r preserves the SIGN of the centered
        representative of (feature - threshold) mod n while scrambling its
        magnitude, so a single decryption yields only the comparison bit
        the protocol must reveal, not the threshold value. r is freshly
        sampled per call (per condition, per query), so repeated queries
        against the same threshold do not reuse the same blinding factor.

        Using Paillier subtraction + scalar multiplication:
            Enc(feature - threshold) = Enc(feature) * Enc(threshold)^{-1} mod n^2
            Enc(r * (feature - threshold)) = Enc(feature - threshold)^r mod n^2

        Args:
            enc_feature:   Paillier ciphertext of encoded feature
            enc_threshold: Paillier ciphertext of encoded threshold
            direction:     'left' (feature <= threshold) or 'right' (>)
                           (unused here; kept for interface stability —
                           direction is applied in _check_condition)

        Returns:
            int: Paillier ciphertext of r * (feature - threshold), r > 0
                 freshly random. NEVER the raw, unblinded difference.
        """
        enc_diff = subtract_encrypted(enc_feature, enc_threshold, self.pub)
        # Fresh positive blinding factor r, resampled on every call.
        #
        # Bound: the encoded values are produced by encode_threshold(), i.e.
        # int(v * SCALE_FACTOR) + OFFSET, so an unblinded difference
        # (feature_int - threshold_int) has magnitude at most ~2*OFFSET
        # (both terms are non-negative and offset-centered near OFFSET).
        # Multiplying by r must not let r * diff exceed n/2 in absolute
        # value, or the centered (sign) representation wraps and the sign
        # we are trying to preserve is corrupted. We therefore cap r well
        # below n / (4 * OFFSET), leaving comfortable margin.
        max_diff_bound = 2 * OFFSET
        r_max = max(2, self.n // (8 * max_diff_bound))
        r = random.randrange(2, r_max)
        return scalar_multiply(enc_diff, r, self.pub)

    def _check_condition(
        self,
        enc_diff: int,
        direction: str,
    ) -> bool:
        """
        Check if a condition is satisfied by decrypting the BLINDED difference.

        Decrypts Enc(r * (feature - threshold)) — see _homomorphic_compare —
        and checks its sign. Multiplying by r > 0 preserves the sign of the
        centered representative under Z_n (and maps 0 to 0), so this
        decryption recovers the comparison bit exactly while the magnitude
        of (feature - threshold), and therefore the threshold value, stays
        hidden from whoever performs this decryption.

        For Paillier with modulus n:
            Values in [0, n//2] represent positive numbers
            Values in (n//2, n) represent negative numbers (mod n)

        Args:
            enc_diff:  Paillier ciphertext of r * (feature - threshold)
                       (blinded; r > 0, freshly random per call)
            direction: 'left' (<=) or 'right' (>)

        Returns:
            bool: True if condition is satisfied
        """
        diff = paillier_decrypt(enc_diff, self.pub, self.priv)

        # Interpret as signed value mod n. Because r > 0, this sign is
        # identical to the sign of the unblinded (feature - threshold);
        # only the magnitude has been scrambled.
        # diff in [0, n//2) → positive → feature >= threshold
        # diff in (n//2, n) → negative (mod n) → feature < threshold
        half_n = self.n // 2

        if direction == 'left':
            # Condition: feature <= threshold  ⟺  feature - threshold <= 0
            # diff <= 0 means diff == 0 or diff is "negative" (> n//2)
            return diff == 0 or diff > half_n
        else:
            # Condition: feature > threshold  ⟺  feature - threshold > 0
            return 0 < diff <= half_n

    def classify(
        self,
        encrypted_features: List[int],
        encrypted_rules: List[EncryptedRule],
    ) -> Optional[int]:
        """
        Perform secure classification on encrypted features.

        Algorithm:
            For each path (grouped by path_id):
                For each condition in the path:
                    Recover feature index from PRP-permuted value
                    Compute Enc(feature - threshold) homomorphically
                    Decrypt and check condition
                If ALL conditions satisfied:
                    Return path label

        Complexity:
            O(d · N) where d = average path depth, N = number of paths
            vs SDTC O(1) with O(2^N) storage

        Leakage: L_infer = {} under Paillier mode (Theorem 3)

        Args:
            encrypted_features: list of Paillier ciphertexts, one per feature
            encrypted_rules:    list of EncryptedRule from RuleEncryptor

        Returns:
            int: predicted class label, or None if no path matches
        """
        # Group rules by path_id, sorted by condition_index
        paths_dict: Dict[int, List[EncryptedRule]] = {}
        for rule in encrypted_rules:
            if rule.path_id not in paths_dict:
                paths_dict[rule.path_id] = []
            paths_dict[rule.path_id].append(rule)

        for path_id in sorted(paths_dict.keys()):
            path_rules = sorted(
                paths_dict[path_id],
                key=lambda r: r.condition_index
            )

            path_satisfied = True
            label = None

            for rule in path_rules:
                # Recover original feature index
                feature_idx = self._recover_feature_idx(rule.enc_feature_idx)

                # Clamp to valid range
                feature_idx = feature_idx % len(encrypted_features)

                # Get encrypted feature
                enc_feature = encrypted_features[feature_idx]

                # Homomorphic comparison
                enc_diff = self._homomorphic_compare(
                    enc_feature,
                    rule.enc_threshold,
                    rule.direction,
                )

                # Check condition
                condition_ok = self._check_condition(enc_diff, rule.direction)

                if not condition_ok:
                    path_satisfied = False
                    break

                if rule.is_last:
                    label = rule.label

            if path_satisfied and label is not None:
                return label

        return None  # No path matched (should not happen with valid tree)

    def classify_plaintext(
        self,
        features: List[float],
        encrypted_rules: List[EncryptedRule],
    ) -> Optional[int]:
        """
        Convenience: encrypt features then classify.

        Args:
            features:        list of float feature values
            encrypted_rules: list of EncryptedRule

        Returns:
            int: predicted class label
        """
        enc_features = self.encryptor.encrypt_feature_vector(features)
        return self.classify(enc_features, encrypted_rules)


# ---------------------------------------------------------------------------
# Two-party role separation (CloudParty / UserParty)
# ---------------------------------------------------------------------------
#
# PaillierInferenceEngine above plays both roles in one process, which is
# convenient for correctness/timing experiments but does not, by itself,
# demonstrate that the Cloud's L_infer = {} claim holds: that claim is a
# statement about what a party WITHOUT the secret key can learn, and a
# single merged class cannot show the secret key is absent from the Cloud's
# view. CloudParty below has no `priv` attribute and no decrypt capability
# at all — by construction, not by convention — so any future code built on
# it cannot silently reintroduce cloud-side decryption.

class CloudParty:
    """
    The Cloud's role in the two-round Paillier inference protocol.

    Holds: the public key, the encrypted rule set.
    Does NOT hold: the Paillier secret key, the permutation key, or the
    deletion key. There is no `priv` attribute on this class; any attempt
    to decrypt must go through code that does not exist here.

    Per Theorem 3, the Cloud's view under this role separation is exactly:
        - Enc(feature) ciphertexts received from the User (IND-CPA)
        - Enc(threshold) ciphertexts already stored (IND-CPA)
        - The blinded difference ciphertexts it computes (IND-CPA; the
          blinding factor r is never revealed to the Cloud's counterpart,
          and in any case the Cloud cannot decrypt anything)
    so L_infer^Cloud = {} (access-pattern leakage aside; see Sec. 1.4/3.3).
    """

    def __init__(self, pub_key: Tuple[int, int]):
        self.pub = pub_key

    def compute_blinded_diffs(
        self,
        encrypted_features: List[int],
        encrypted_rules: List[EncryptedRule],
        recover_feature_idx,
    ) -> List[Tuple[EncryptedRule, int]]:
        """
        For each rule, homomorphically compute Enc(feature - threshold) and
        blind it with a fresh random positive scalar before returning it.
        The Cloud never decrypts; this method only produces ciphertexts.

        Args:
            encrypted_features: the User's Enc(x_i) for i = 1..n
            encrypted_rules:    the stored EncryptedRule set
            recover_feature_idx: callable(enc_feature_idx) -> int, supplied
                by the caller because recovering the PRP-permuted feature
                index requires the permutation key, which the Cloud does
                not hold either — in the real protocol the Cloud sends the
                permuted index and the User (who holds the permutation
                key) resolves it; this parameter keeps that asymmetry
                explicit instead of quietly giving the Cloud the key.

        Returns:
            list of (rule, blinded_enc_diff) pairs, one per rule.
        """
        out = []
        for rule in encrypted_rules:
            feature_idx = recover_feature_idx(rule.enc_feature_idx)
            feature_idx = feature_idx % len(encrypted_features)
            enc_feature = encrypted_features[feature_idx]

            enc_diff = subtract_encrypted(enc_feature, rule.enc_threshold, self.pub)
            n = self.pub[0]
            r_max = max(2, n // (8 * 2 * OFFSET))
            r = random.randrange(2, r_max)
            blinded = scalar_multiply(enc_diff, r, self.pub)
            out.append((rule, blinded))
        return out


class UserParty:
    """
    The User's role in the two-round Paillier inference protocol.

    Holds: the Paillier secret key (sk), the permutation key (needed to
    resolve which feature each rule's PRP-permuted index refers to), and
    their own plaintext feature vector.

    Per Theorem 3, the User's view includes the comparison bit for each
    condition evaluated against their own query, and the final label —
    both of which the protocol must reveal for classification to succeed.
    It does NOT include the threshold magnitude (the blinding factor
    applied by the Cloud prevents recovering it from a single query); an
    adaptive User issuing many queries could still binary-search a
    threshold, which is why query rate-limiting is stated as a deployment
    requirement rather than folded into a stronger single-query theorem.
    """

    def __init__(self, encryptor: RuleEncryptor):
        self.encryptor = encryptor
        self.pub = encryptor.paillier_pub
        self.priv = encryptor.paillier_priv
        self.perm_key = encryptor.permutation_key
        self.n = encryptor.n

    def recover_feature_idx(self, enc_feature_idx: int) -> int:
        """Resolve a PRP-permuted feature index using the permutation key."""
        enc_bytes = enc_feature_idx.to_bytes(16, byteorder='big')
        original = prp_inverse(self.perm_key, enc_bytes)
        return int.from_bytes(original, byteorder='big') % 256

    def decrypt_sign(self, blinded_enc_diff: int, direction: str) -> bool:
        """Decrypt a blinded difference and read off the comparison bit only."""
        diff = paillier_decrypt(blinded_enc_diff, self.pub, self.priv)
        half_n = self.n // 2
        if direction == 'left':
            return diff == 0 or diff > half_n
        return 0 < diff <= half_n

    def classify_with_cloud(
        self,
        features: List[float],
        encrypted_rules: List[EncryptedRule],
        cloud: 'CloudParty',
    ) -> Optional[int]:
        """
        Run the full two-round protocol against a separate CloudParty
        instance: encrypt features, send to Cloud, receive blinded diffs,
        decrypt only the sign of each, and evaluate paths exactly as
        PaillierInferenceEngine.classify does.
        """
        enc_features = self.encryptor.encrypt_feature_vector(features)
        pairs = cloud.compute_blinded_diffs(
            enc_features, encrypted_rules, self.recover_feature_idx
        )

        paths_dict: Dict[int, List[Tuple[EncryptedRule, int]]] = {}
        for rule, blinded in pairs:
            paths_dict.setdefault(rule.path_id, []).append((rule, blinded))

        for path_id in sorted(paths_dict.keys()):
            path_pairs = sorted(paths_dict[path_id], key=lambda rb: rb[0].condition_index)
            path_satisfied = True
            label = None
            for rule, blinded in path_pairs:
                if not self.decrypt_sign(blinded, rule.direction):
                    path_satisfied = False
                    break
                if rule.is_last:
                    label = rule.label
            if path_satisfied and label is not None:
                return label
        return None


# ---------------------------------------------------------------------------
# Plaintext Reference Classifier (for accuracy comparison)
# ---------------------------------------------------------------------------

class PlaintextClassifier:
    """
    Reference plaintext decision tree classifier.

    Used in Experiment 1 to verify that PrivPathInfer achieves
    identical accuracy to plaintext classification.

    This proves Contribution 1: no accuracy loss from continuous
    feature support (unlike SDTC discretization).
    """

    def classify(
        self,
        features: List[float],
        paths,  # List[LeafPath]
    ) -> Optional[int]:
        """
        Classify using plaintext decision tree paths.

        Args:
            features: list of float feature values
            paths:    list of LeafPath from PathExtractor

        Returns:
            int: class label, or None if no path matches
        """
        for path in paths:
            satisfied = True
            for cond in path.conditions:
                val = features[cond.feature_idx]
                if cond.direction == 'left':
                    if not (val <= cond.threshold):
                        satisfied = False
                        break
                else:
                    if not (val > cond.threshold):
                        satisfied = False
                        break
            if satisfied:
                return path.label
        return None


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """
    Verify that PrivPathInfer secure inference matches plaintext classification,
    AND verify the blinding fix: (a) correctness is preserved after blinding,
    (b) the blinded difference does not reveal the unblinded magnitude, and
    (c) the explicit two-party CloudParty/UserParty split matches both the
    single-process engine and the plaintext reference exactly.

    Tests:
        1. Single sample: encrypted result matches plaintext result
        2. Multiple samples: all match
        3. All class labels correctly returned
        4. Edge cases: boundary threshold values
        5. Blinding correctness: classification still exact after the fix
        6. Blinding hides magnitude: decrypted blinded diff != raw diff
           whenever raw diff != 0 (the security property this fix adds)
        7. Two-party split (CloudParty/UserParty) matches the single-process
           engine and the plaintext classifier on every test case
    """
    from system.path_extractor import PathExtractor, from_dict

    print("=" * 60)
    print("InferenceEngine Verification Tests")
    print("Paillier Mode — L_infer^Cloud = {}, L_infer^User = {comparison")
    print("bits, label} (single-query; see module docstring, Theorem 3)")
    print("=" * 60)

    # Build test tree
    # Tree: glucose > 126.5 → diabetic (1), else check BMI
    tree_dict = {
        'feature_idx': 1,   # glucose (index 1)
        'threshold':   126.5,
        'left': {
            'feature_idx': 5,  # BMI (index 5)
            'threshold':   29.1,
            'left':  {'label': 0},  # low glucose, low BMI → not diabetic
            'right': {'label': 1},  # low glucose, high BMI → diabetic
        },
        'right': {'label': 1},  # high glucose → diabetic
    }

    root      = from_dict(tree_dict)
    extractor = PathExtractor(root)
    paths     = extractor.extract_paths()

    print(f"\nTest tree: {len(paths)} paths")
    print("Features: [f0, glucose, f2, f3, f4, BMI, f6, f7]")

    # Setup encryptor and engine
    encryptor = RuleEncryptor(paillier_bits=512)
    rules     = encryptor.encrypt_paths(paths)
    engine    = PaillierInferenceEngine(encryptor)
    plaintext = PlaintextClassifier()

    # Test cases: [feature_vector, expected_label, description]
    test_cases = [
        ([0]*8,                                    0, "all zeros → not diabetic"),
        ([0, 80.0,  0, 0, 0, 20.0, 0, 0],         0, "low glucose, low BMI → 0"),
        ([0, 80.0,  0, 0, 0, 35.0, 0, 0],         1, "low glucose, high BMI → 1"),
        ([0, 150.0, 0, 0, 0, 20.0, 0, 0],         1, "high glucose → 1"),
        ([0, 126.5, 0, 0, 0, 20.0, 0, 0],         0, "exactly at boundary (<=) → 0"),
        ([0, 126.6, 0, 0, 0, 20.0, 0, 0],         1, "just above boundary → 1"),
        ([0, 29.0,  0, 0, 0, 29.1, 0, 0],         0, "BMI exactly at boundary → 0"),
    ]

    passed = 0
    for features, expected, description in test_cases:
        # Plaintext classification
        pt_result = plaintext.classify(features, paths)

        # Secure classification (now blinded internally — Test 5 coverage)
        enc_result = engine.classify_plaintext(features, rules)

        assert pt_result  == expected,  \
            f"Plaintext FAILED [{description}]: got {pt_result}, expected {expected}"
        assert enc_result == expected,  \
            f"Encrypted FAILED [{description}]: got {enc_result}, expected {expected}"
        assert pt_result  == enc_result, \
            f"Mismatch [{description}]: plaintext={pt_result}, encrypted={enc_result}"

        print(f"[PASS] {description}")
        passed += 1

    print(f"\n[ALL {passed} TESTS PASSED] correctness preserved after blinding fix.")

    # ------------------------------------------------------------------
    # Test 6: blinding actually hides the magnitude (the security property)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Test 6: blinded decryption does not reveal the raw difference")
    print("-" * 60)
    enc_feat = encryptor.encrypt_feature_vector([0, 150.0, 0, 0, 0, 20.0, 0, 0])
    rule0 = rules[0]
    from crypto.paillier import subtract_encrypted as _sub, decrypt as _dec

    feature_idx0 = engine._recover_feature_idx(rule0.enc_feature_idx) % len(enc_feat)
    raw_enc_diff = _sub(enc_feat[feature_idx0], rule0.enc_threshold, encryptor.paillier_pub)
    raw_diff_val = _dec(raw_enc_diff, encryptor.paillier_pub, encryptor.paillier_priv)

    blinded_enc_diff = engine._homomorphic_compare(
        enc_feat[feature_idx0], rule0.enc_threshold, rule0.direction
    )
    blinded_diff_val = _dec(blinded_enc_diff, engine.pub, engine.priv)
    if raw_diff_val != 0:
        assert blinded_diff_val != raw_diff_val, (
            "Blinding FAILED: decrypted blinded value equals the raw "
            "unblinded difference — the magnitude is not hidden."
        )
        print(f"[PASS] raw diff = {raw_diff_val}, blinded diff = "
              f"{blinded_diff_val} (different — magnitude is hidden)")
    else:
        print("[INFO] raw diff happened to be 0 for this sample/rule; "
              "magnitude-hiding check is vacuous here by construction "
              "(0 * r = 0), retrying is unnecessary since sign (the only "
              "thing the protocol must reveal) is still correctly 0.")

    # ------------------------------------------------------------------
    # Test 7: explicit two-party split matches plaintext and the
    # single-process engine on every test case
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Test 7: CloudParty/UserParty two-party split matches engine")
    print("-" * 60)
    cloud = CloudParty(encryptor.paillier_pub)
    user  = UserParty(encryptor)
    assert not hasattr(cloud, 'priv'), \
        "CloudParty must never hold a secret-key attribute"

    for features, expected, description in test_cases:
        two_party_result = user.classify_with_cloud(features, rules, cloud)
        assert two_party_result == expected, (
            f"Two-party split FAILED [{description}]: "
            f"got {two_party_result}, expected {expected}"
        )
    print(f"[PASS] Two-party split matches plaintext on all "
          f"{len(test_cases)} test cases; CloudParty holds no secret key.")

    print(f"\n[ALL TESTS PASSED] inference_engine.py verified "
          f"(post-blinding-fix).")
    print("Secure inference matches plaintext classification exactly.")
    print("Leakage (corrected): L_infer^Cloud = {} ; "
          "L_infer^User = {comparison bits, label} (single query).")


if __name__ == "__main__":
    run_all_tests()