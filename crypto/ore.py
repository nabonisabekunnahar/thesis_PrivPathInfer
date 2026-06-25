"""
ore.py — Simplified Order-Revealing Encryption (ORE) — Theoretical Demonstration
==================================================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

IMPORTANT SCOPING NOTE:
    This ORE implementation is for THEORETICAL DISCUSSION ONLY.
    It is NOT the primary cryptographic contribution of PrivPathInfer.
    The fully implemented and cryptographically rigorous mode is
    Paillier mode (see paillier.py).

    ORE mode is discussed in the paper to show that an alternative
    inference protocol exists with different leakage characteristics.
    It is NOT used in the security proofs or experiments.

Implementation Reference:
    Lewi, K. and Wu, D. "Order-Revealing Encryption: New Constructions,
    Applications, and Lower Bounds." ACM CCS 2016, pp. 1167-1178.
    ACM, New York, 2016.

ORE Security Definition (Lewi-Wu 2016, Definition 2.2):
    An ORE scheme Π = (ORE.Setup, ORE.Encrypt, ORE.Compare) is secure
    with leakage function L if for all polynomial-size adversaries A,
    there exists a simulator S such that:
        REAL^{ORE}_A(λ) ≈_c SIM^{ORE}_{A,S,L}(λ)

    For the small-domain construction used here:
        L_CMP = {(i,j, cmp(mᵢ, mⱼ)) : 1 ≤ i < j ≤ t}
    The leakage is the pairwise comparison results of all queried values.

Leakage Comparison (PrivPathInfer, Theorem 3):
    Paillier mode: L_infer = {}          (empty — cloud learns nothing)
    ORE mode:      L_infer = {comparison_results}

    This is why Paillier mode is the primary contribution: it achieves
    strictly less leakage than ORE mode.

Lewi-Wu Small-Domain ORE Construction (Section 3):
    Message space: [N] where N = poly(λ)
    Setup: Sample PRF key k, random permutation π : [N] → [N]
    Left encrypt(x):  ct_L = (F(k, π(x)), π(x))
    Right encrypt(y): ct_R = (r, {cmp(π⁻¹(i), y) + H(F(k,i), r) mod 3}_{i∈[N]})
    Compare(ct_L, ct_R): result = v_{π(x)} - H(k', r) mod 3

    For our simplified demonstration, we implement a conceptually
    equivalent construction using AES as the PRF and a truncated
    domain for practical testing.

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import random
import hashlib
from crypto.aes128 import aes_encrypt
from crypto.prf_prp import prf, _encode_to_block


# ---------------------------------------------------------------------------
# ORE Constants and Configuration
# ---------------------------------------------------------------------------

# Domain size for the small-domain ORE construction
# In Lewi-Wu 2016, N = poly(λ). For demonstration, we use a fixed domain.
# For the full construction over large domains, see Section 4 of Lewi-Wu 2016.
_ORE_DOMAIN_SIZE = 64  # Small domain for theoretical demonstration
# Note: Lewi-Wu 2016 right encryption is O(N) per ciphertext.
# For large domains, the full construction in Section 4 of Lewi-Wu 2016
# (block-level ORE) is required. This small domain suffices for
# theoretical demonstration purposes in PrivPathInfer.

# Comparison output values (Lewi-Wu 2016, Remark 2.3)
ORE_LESS    = -1   # x < y
ORE_EQUAL   =  0   # x == y
ORE_GREATER =  1   # x > y


# ---------------------------------------------------------------------------
# ORE Setup
# Reference: Lewi-Wu 2016, Section 3.1 (ORE.Setup)
# ---------------------------------------------------------------------------

def ore_setup(domain_size=_ORE_DOMAIN_SIZE):
    """
    ORE Key Generation.

    Generates a PRF key k and a random permutation π over the domain [N].

    Reference: Lewi-Wu 2016, Section 3.1, ORE.Setup(1^λ).

    Algorithm:
        1. Sample PRF key k ←$ {0,1}^λ
        2. Sample random permutation π : [N] → [N]
        3. Secret key sk = (k, π)

    Args:
        domain_size: size of the plaintext domain N

    Returns:
        dict: secret key containing:
            'prf_key':    PRF key k (bytes, 16)
            'permutation': π as a list, π[i] = π(i)
            'inv_perm':   π⁻¹ as a list
            'domain_size': N
    """
    # Step 1: Sample PRF key
    prf_key = os.urandom(16)

    # Step 2: Sample random permutation π : [N] → [N]
    perm = list(range(domain_size))
    random.shuffle(perm)

    # Compute inverse permutation π⁻¹
    inv_perm = [0] * domain_size
    for i, p in enumerate(perm):
        inv_perm[p] = i

    return {
        'prf_key':     prf_key,
        'permutation': perm,
        'inv_perm':    inv_perm,
        'domain_size': domain_size,
    }


# ---------------------------------------------------------------------------
# ORE Encryption
# Reference: Lewi-Wu 2016, Section 3.1
# ---------------------------------------------------------------------------

def ore_encrypt_left(x, sk):
    """
    ORE Left Encryption.

    ct_L = (F(k, π(x)), π(x))

    The left ciphertext consists of:
        - A PRF evaluation at the permuted position
        - The permuted position itself

    Reference: Lewi-Wu 2016, Section 3.1, ORE.EncryptL(sk, x).

    Args:
        x:  plaintext value in [0, N-1]
        sk: secret key from ore_setup()

    Returns:
        tuple: (prf_val, perm_x) — the left ciphertext
    """
    k    = sk['prf_key']
    perm = sk['permutation']
    N    = sk['domain_size']

    assert 0 <= x < N, f"Plaintext x={x} must be in [0, {N-1}]"

    perm_x  = perm[x]
    prf_val = prf(k, perm_x)

    return (prf_val, perm_x)


def ore_encrypt_right(y, sk):
    """
    ORE Right Encryption.

    For each i ∈ [N]:
        vᵢ = cmp(π⁻¹(i), y) + H(F(k, i), r) mod 3

    ct_R = (r, v₀, v₁, ..., v_{N-1})

    The right ciphertext encodes comparison results with all domain
    elements, masked by PRF-derived values.

    Reference: Lewi-Wu 2016, Section 3.1, ORE.EncryptR(sk, y).

    Args:
        y:  plaintext value in [0, N-1]
        sk: secret key from ore_setup()

    Returns:
        tuple: (nonce, values_list) — the right ciphertext
    """
    k        = sk['prf_key']
    inv_perm = sk['inv_perm']
    N        = sk['domain_size']

    assert 0 <= y < N, f"Plaintext y={y} must be in [0, {N-1}]"

    # Sample random nonce r
    r = os.urandom(16)

    # For each i in [N], compute vᵢ
    values = []
    for i in range(N):
        # cmp(π⁻¹(i), y): comparison of original value at permuted position i with y
        orig_i = inv_perm[i]
        if orig_i < y:
            cmp_val = -1  # less
        elif orig_i == y:
            cmp_val = 0   # equal
        else:
            cmp_val = 1   # greater

        # H(F(k, i), r): hash of PRF output at i, with nonce r
        prf_i     = prf(k, i)
        hash_val  = _hash_mod3(prf_i, r)

        # vᵢ = cmp(π⁻¹(i), y) + H(F(k,i), r) mod 3
        # Map cmp_val from {-1, 0, 1} to {0, 1, 2} for mod 3 arithmetic
        cmp_mod3 = cmp_val % 3  # -1 → 2, 0 → 0, 1 → 1
        v_i      = (cmp_mod3 + hash_val) % 3
        values.append(v_i)

    return (r, values)


def ore_compare(ct_left, ct_right):
    """
    ORE Comparison: determine ordering from left and right ciphertexts.

    Algorithm:
        Parse ct_L = (k', h) = (prf_val, perm_x)
        Parse ct_R = (r, v₀, ..., v_{N-1})
        result = v_{perm_x} - H(k', r) mod 3

    The result maps:
        0 → equal   (x == y)
        1 → greater (x > y)
        2 → less    (x < y)

    Reference: Lewi-Wu 2016, Section 3.1, ORE.Compare(ct_L, ct_R).

    Leakage Note (PrivPathInfer, Theorem 3):
        The cloud learns the comparison result (x < y, x == y, x > y).
        Formal leakage: L_infer = {comparison_results}
        This is strictly more than Paillier mode (L_infer = {}).

    Args:
        ct_left:  left ciphertext (prf_val, perm_x) from ore_encrypt_left()
        ct_right: right ciphertext (r, values) from ore_encrypt_right()

    Returns:
        int: ORE_LESS (-1), ORE_EQUAL (0), or ORE_GREATER (1)
    """
    prf_val, perm_x = ct_left
    r, values       = ct_right

    # Compute H(k', r) = hash of PRF value with nonce
    hash_val = _hash_mod3(prf_val, r)

    # v_{π(x)} - H(k', r) mod 3
    v_perm_x = values[perm_x]
    result   = (v_perm_x - hash_val) % 3

    # Map result: 0 → equal, 1 → greater, 2 → less
    if result == 0:
        return ORE_EQUAL
    elif result == 1:
        return ORE_GREATER
    else:  # result == 2
        return ORE_LESS


# ---------------------------------------------------------------------------
# Helper: Hash function mod 3
# ---------------------------------------------------------------------------

def _hash_mod3(prf_output, nonce):
    """
    Hash function H : {0,1}^λ × {0,1}^λ → Z₃

    Modeled as a random oracle in the security proof.
    Instantiated with SHA-256 for the theoretical demonstration.

    H(prf_output, nonce) = SHA256(prf_output || nonce) mod 3

    Reference: Lewi-Wu 2016, Section 3.1.

    Args:
        prf_output: bytes (16)
        nonce:      bytes (16)

    Returns:
        int: value in {0, 1, 2}
    """
    if isinstance(prf_output, bytes):
        data = prf_output + nonce
    else:
        data = _encode_to_block(prf_output) + nonce

    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], byteorder='big') % 3


# ---------------------------------------------------------------------------
# Simplified ORE for Fixed-Point Encoded Thresholds
# Used in PrivPathInfer ORE mode inference
# ---------------------------------------------------------------------------

def ore_encrypt_threshold(threshold_int, sk):
    """
    Encrypt a fixed-point encoded threshold for ORE-based inference.

    The threshold_int is produced by paillier.encode_threshold().
    Since the ORE domain must be [0, N-1], we normalize the threshold
    to fit within the demonstration domain.

    NOTE: This is a conceptual demonstration. In a full implementation,
    the ORE construction of Lewi-Wu 2016 Section 4 (large-domain ORE)
    would be used to handle the full fixed-point encoding range.

    Args:
        threshold_int: encoded threshold from encode_threshold()
        sk:           ORE secret key from ore_setup()

    Returns:
        tuple: right ciphertext for the threshold
    """
    N = sk['domain_size']
    # Normalize to domain: use modular reduction for demonstration
    y_normalized = threshold_int % N
    return ore_encrypt_right(y_normalized, sk)


def ore_encrypt_feature(feature_int, sk):
    """
    Encrypt a fixed-point encoded feature value for ORE comparison.

    Args:
        feature_int: encoded feature from encode_threshold()
        sk:         ORE secret key from ore_setup()

    Returns:
        tuple: left ciphertext for the feature value
    """
    N = sk['domain_size']
    x_normalized = feature_int % N
    return ore_encrypt_left(x_normalized, sk)


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests(num_tests=1000):
    """
    Verify ORE correctness: all comparison results must match plaintext.

    Tests 1000 random pairs (x, y) and verifies that:
        ore_compare(E_L(x), E_R(y)) == cmp(x, y)

    Reference: Lewi-Wu 2016, Section 3.1 (Correctness).

    Args:
        num_tests: number of random pairs to test (default 1000)
    """
    print("=" * 60)
    print("Simplified ORE Verification Tests")
    print("Reference: Lewi-Wu 2016, ACM CCS")
    print("SCOPE: Theoretical demonstration only")
    print("=" * 60)

    sk = ore_setup()
    N  = sk['domain_size']

    print(f"\nDomain size N = {N}")
    print(f"Running {num_tests} random comparison tests...")

    passed = 0
    failed = 0

    for _ in range(num_tests):
        x = random.randrange(0, N)
        y = random.randrange(0, N)

        ct_left  = ore_encrypt_left(x, sk)
        ct_right = ore_encrypt_right(y, sk)
        result   = ore_compare(ct_left, ct_right)

        # Expected comparison result
        if x < y:
            expected = ORE_LESS
        elif x == y:
            expected = ORE_EQUAL
        else:
            expected = ORE_GREATER

        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: x={x}, y={y}, expected={expected}, got={result}")

    print(f"\nResults: {passed}/{num_tests} passed, {failed}/{num_tests} failed")

    assert failed == 0, f"ORE correctness FAILED: {failed} test cases incorrect"
    print(f"\n[PASS] All {num_tests} comparison tests correct.")

    # Test edge cases
    x = 0
    ct_l = ore_encrypt_left(x, sk)
    ct_r = ore_encrypt_right(x, sk)
    assert ore_compare(ct_l, ct_r) == ORE_EQUAL, "Equal case FAILED"
    print("[PASS] Edge case: x == y → ORE_EQUAL")

    x, y = 0, N-1
    ct_l = ore_encrypt_left(x, sk)
    ct_r = ore_encrypt_right(y, sk)
    assert ore_compare(ct_l, ct_r) == ORE_LESS, "Min < Max case FAILED"
    print("[PASS] Edge case: min < max → ORE_LESS")

    print("\n[ALL TESTS PASSED] ore.py verified.")
    print("Leakage: L_infer = {comparison_results} (Theorem 3)")
    print("Note: Paillier mode achieves L_infer = {} — strictly less leakage.")


if __name__ == "__main__":
    run_all_tests(num_tests=1000)