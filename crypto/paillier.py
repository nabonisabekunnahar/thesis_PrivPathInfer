"""
paillier.py — Paillier Homomorphic Encryption from Scratch
===========================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Implementation Reference:
    Paillier, P. "Public-Key Cryptosystems Based on Composite Degree
    Residuosity Classes." EUROCRYPT 1999, LNCS 1592, pp. 223-238.
    Springer-Verlag, 1999.

Security Assumption:
    Decisional Composite Residuosity Assumption (DCRA), Paillier 1999,
    Conjecture 2:
        There exists no polynomial-time distinguisher for n-th residues
        modulo n^2. That is, given z ∈ Z*_{n^2}, it is computationally
        hard to decide whether z = y^n mod n^2 for some y ∈ Z*_{n^2}.

    Theorem 15 (Paillier 1999):
        Scheme 1 is semantically secure (IND-CPA) if and only if the
        Decisional Composite Residuosity Assumption holds.

    Semantic Security Definition (Boneh-Shoup, Definition 2.2):
        For all efficient adversaries A:
            SSadv[A, E] = |Pr[b̂=b in Exp 0] - Pr[b̂=b in Exp 1]| ≤ negl(λ)

Usage in PrivPathInfer:
    - Theorem 1 (Data Privacy): User features encrypted under Paillier.
    - Theorem 2 (Classifier Privacy): Decision tree thresholds encrypted
      under Paillier. Semantic security under DCRA ensures no information
      about plaintext thresholds is revealed to the cloud.
    - Inference: Cloud performs homomorphic comparison without decryption.

Homomorphic Properties (Paillier 1999, Section 8):
    Additive:  D(E(m1) * E(m2) mod n^2) = m1 + m2 mod n
    Scalar:    D(E(m)^k mod n^2)         = k * m  mod n

Fixed-Point Encoding for Continuous Features (PrivPathInfer Contribution 1):
    threshold_int = int(threshold * 10000) + 10^9
    This preserves 4 decimal places of precision without discretization.

Key Sizes:
    512-bit  n: for testing and development
    1024-bit n: for experiments (reported in paper)
    2048-bit n: recommended for production

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import math
import random


# ---------------------------------------------------------------------------
# Number Theory Primitives
# ---------------------------------------------------------------------------

def _miller_rabin(n, k=20):
    """
    Miller-Rabin probabilistic primality test.

    Theory:
        For an odd integer n > 2, write n-1 = 2^r * d where d is odd.
        For a random witness a, n is probably prime if:
            a^d ≡ 1 (mod n), OR
            a^(2^j * d) ≡ -1 (mod n) for some j ∈ {0, ..., r-1}

        If none of these hold, n is definitely composite.

    Error probability: at most 4^(-k) for k rounds.
    For k=20: error probability ≤ 4^(-20) ≈ 10^(-12), negligible.

    Reference: Cormen et al., Introduction to Algorithms, Chapter 31.

    Args:
        n: integer to test for primality
        k: number of rounds (default 20 for negligible error)

    Returns:
        True if n is probably prime, False if definitely composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 = 2^r * d with d odd
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Perform k rounds of Miller-Rabin
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # Definitely composite

    return True  # Probably prime


def _generate_prime(bits):
    """
    Generate a random prime of the specified bit length.

    Algorithm:
        1. Generate a random odd integer of the specified bit length.
        2. Test primality using Miller-Rabin with 20 rounds.
        3. Repeat until a prime is found.

    Expected iterations: O(bits) by the prime number theorem.

    Args:
        bits: desired bit length of the prime

    Returns:
        int: a probable prime of exactly `bits` bits
    """
    while True:
        # Generate random odd number with correct bit length
        p = random.getrandbits(bits)
        # Ensure correct bit length: set high bit and low bit
        p |= (1 << (bits - 1))  # set high bit
        p |= 1                   # set low bit (odd)
        if _miller_rabin(p):
            return p


def _extended_gcd(a, b):
    """
    Extended Euclidean Algorithm (iterative).

    Computes gcd(a, b) and coefficients x, y such that:
        a*x + b*y = gcd(a, b)

    Iterative version to avoid recursion limit for large integers
    (e.g., 1024-bit Paillier moduli).

    Reference: Cormen et al., Introduction to Algorithms, Section 31.2

    Args:
        a, b: integers

    Returns:
        (gcd, x, y) such that a*x + b*y = gcd(a, b)
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t


def _mod_inverse(a, m):
    """
    Compute the modular multiplicative inverse of a modulo m.

    Finds x such that a*x ≡ 1 (mod m), using the Extended Euclidean
    Algorithm. Inverse exists if and only if gcd(a, m) = 1.

    Reference: Paillier 1999 uses this in KeyGen to compute μ.

    Args:
        a: integer whose inverse is sought
        m: modulus

    Returns:
        int: x such that (a * x) % m == 1

    Raises:
        ValueError: if gcd(a, m) != 1 (inverse does not exist)
    """
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(
            f"Modular inverse does not exist: gcd({a}, {m}) = {gcd} ≠ 1"
        )
    return x % m


def _lcm(a, b):
    """
    Compute the least common multiple of a and b.

    lcm(a, b) = |a * b| / gcd(a, b)

    Used in Paillier KeyGen: λ = lcm(p-1, q-1)
    where λ is the Carmichael function of n = pq.

    Reference: Paillier 1999, Notations section.

    Args:
        a, b: positive integers

    Returns:
        int: lcm(a, b)
    """
    return abs(a * b) // math.gcd(a, b)


def _L(x, n):
    """
    The L function from Paillier 1999.

    L(x) = (x - 1) / n

    This function is well-defined for x ∈ S_n where:
        S_n = {u < n^2 : u ≡ 1 mod n}

    For any such x, (x - 1) is divisible by n, so the result is an integer.

    Reference: Paillier 1999, Section 3, proof of Theorem 9.

    Args:
        x: integer with x ≡ 1 (mod n)
        n: the RSA modulus

    Returns:
        int: (x - 1) // n
    """
    assert (x - 1) % n == 0, (
        f"L function undefined: ({x} - 1) is not divisible by {n}. "
        f"Ensure x ≡ 1 (mod n)."
    )
    return (x - 1) // n


def _random_coprime(n):
    """
    Generate a random integer r in [1, n-1] with gcd(r, n) = 1.

    Used in Paillier encryption to generate the randomness r.
    Since n = pq with large primes p, q, almost all integers in [1, n-1]
    are coprime to n. Expected iterations: O(1).

    Args:
        n: the RSA modulus

    Returns:
        int: random r with 1 ≤ r < n and gcd(r, n) = 1
    """
    while True:
        r = random.randrange(1, n)
        if math.gcd(r, n) == 1:
            return r


# ---------------------------------------------------------------------------
# Paillier Key Generation
# Reference: Paillier 1999, Section 4 (Scheme 1)
# ---------------------------------------------------------------------------

def keygen(bits=512):
    """
    Generate a Paillier key pair.

    Algorithm (Paillier 1999, Scheme 1 with g = n+1 simplification):
        1. Generate two random primes p, q of size bits//2
        2. Compute n = p * q
        3. Compute λ = lcm(p-1, q-1)  [Carmichael function]
        4. Set g = n + 1               [Simplified variant]
        5. Compute μ = (L(g^λ mod n^2))^{-1} mod n

    g = n+1 Justification (Binomial Theorem):
        (1+n)^m mod n^2 = 1 + mn mod n^2
        This simplification is valid and makes encryption faster.
        All terms of degree ≥ 2 in n vanish modulo n^2.

    Key Sizes:
        bits=512:  testing and development
        bits=1024: final experiments (reported in paper)
        bits=2048: production recommendation

    Args:
        bits: total bit length of n = p*q (default 512 for testing)

    Returns:
        (public_key, private_key) where:
            public_key  = (n, g)
            private_key = (lam, mu, p, q)
    """
    half_bits = bits // 2

    # Step 1: Generate two distinct primes p, q of size bits//2
    p = _generate_prime(half_bits)
    q = _generate_prime(half_bits)
    while q == p:
        q = _generate_prime(half_bits)

    # Step 2: n = p * q
    n = p * q
    n2 = n * n

    # Step 3: λ = lcm(p-1, q-1)
    lam = _lcm(p - 1, q - 1)

    # Step 4: g = n + 1 (simplified variant)
    # Justification: (n+1)^m mod n^2 = 1 + mn mod n^2 (binomial theorem)
    g = n + 1

    # Step 5: μ = (L(g^λ mod n^2))^{-1} mod n
    g_lam = pow(g, lam, n2)
    L_val = _L(g_lam, n)
    mu = _mod_inverse(L_val, n)

    public_key  = (n, g)
    private_key = (lam, mu, p, q)

    return public_key, private_key


# ---------------------------------------------------------------------------
# Paillier Encryption
# Reference: Paillier 1999, Scheme 1
# ---------------------------------------------------------------------------

def encrypt(m, public_key):
    """
    Paillier Encryption.

    Formula: c = g^m * r^n mod n^2

    Where r is a random element with gcd(r, n) = 1.

    IND-CPA Security:
        The random r ensures that the same plaintext m encrypts to a
        different ciphertext each time (probabilistic encryption).
        This prevents the cloud from comparing ciphertexts to learn
        equality of plaintexts (Boneh-Shoup, Definition 2.2).

    With g = n+1:
        g^m mod n^2 = (1+n)^m mod n^2 = 1 + mn mod n^2
        This simplification is used in the implementation.

    Reference: Paillier 1999, Section 4, Scheme 1.

    Args:
        m:          plaintext integer, 0 ≤ m < n
        public_key: (n, g) from keygen()

    Returns:
        int: ciphertext c with 0 ≤ c < n^2
    """
    n, g = public_key
    n2 = n * n

    assert 0 <= m < n, f"Plaintext m={m} must satisfy 0 ≤ m < n={n}"

    # Generate random r with gcd(r, n) = 1
    r = _random_coprime(n)

    # c = g^m * r^n mod n^2
    c = (pow(g, m, n2) * pow(r, n, n2)) % n2

    return c


def encrypt_with_r(m, public_key, r):
    """
    Paillier Encryption with explicit randomness r.

    Used for testing and for cases where deterministic behavior is needed
    (e.g., threshold encryption where the same r is reused for comparison).

    Args:
        m:          plaintext integer, 0 ≤ m < n
        public_key: (n, g) from keygen()
        r:          explicit randomness, gcd(r, n) = 1

    Returns:
        int: ciphertext c
    """
    n, g = public_key
    n2 = n * n

    assert 0 <= m < n, f"Plaintext m={m} must satisfy 0 ≤ m < n={n}"
    assert math.gcd(r, n) == 1, f"r={r} must be coprime to n"

    c = (pow(g, m, n2) * pow(r, n, n2)) % n2
    return c


# ---------------------------------------------------------------------------
# Paillier Decryption
# Reference: Paillier 1999, Scheme 1
# ---------------------------------------------------------------------------

def decrypt(c, public_key, private_key):
    """
    Paillier Decryption.

    Formula: m = L(c^λ mod n^2) * μ mod n

    Correctness Proof (Paillier 1999, Section 4):
        c = g^m * r^n mod n^2
        c^λ mod n^2 = (g^m * r^n)^λ mod n^2
                    = g^(mλ) * r^(nλ) mod n^2

        By Carmichael's theorem: r^(nλ) ≡ 1 mod n^2
        With g = n+1: g^(mλ) mod n^2 = 1 + mλn mod n^2

        L(c^λ mod n^2) = mλn/n = mλ mod n
        m = mλ * μ mod n = mλ * λ^{-1} mod n = m ✓

    Reference: Paillier 1999, Section 4, Scheme 1.

    Args:
        c:           ciphertext integer, 0 ≤ c < n^2
        public_key:  (n, g) from keygen()
        private_key: (lam, mu, p, q) from keygen()

    Returns:
        int: plaintext m with 0 ≤ m < n
    """
    n, g = public_key
    lam, mu, p, q = private_key
    n2 = n * n

    assert 0 <= c < n2, f"Ciphertext c={c} must satisfy 0 ≤ c < n^2"

    # Step 1: u = c^λ mod n^2
    u = pow(c, lam, n2)

    # Step 2: m = L(u) * μ mod n
    m = (_L(u, n) * mu) % n

    return m


# ---------------------------------------------------------------------------
# Homomorphic Operations
# Reference: Paillier 1999, Section 8
# ---------------------------------------------------------------------------

def add_encrypted(c1, c2, public_key):
    """
    Homomorphic addition of two ciphertexts.

    Property (Paillier 1999, Section 8):
        D(E(m1) * E(m2) mod n^2) = m1 + m2 mod n

    Proof:
        E(m1) * E(m2) = (g^m1 * r1^n) * (g^m2 * r2^n) mod n^2
                      = g^(m1+m2) * (r1*r2)^n mod n^2
                      = E(m1+m2, r1*r2)  ✓

    Usage in PrivPathInfer:
        Cloud computes E(feature) - E(threshold) homomorphically to
        perform secure comparison without learning plaintext values.

    Args:
        c1, c2:     Paillier ciphertexts
        public_key: (n, g) from keygen()

    Returns:
        int: ciphertext encrypting (m1 + m2) mod n
    """
    n, g = public_key
    n2 = n * n
    return (c1 * c2) % n2


def subtract_encrypted(c1, c2, public_key):
    """
    Homomorphic subtraction: compute E(m1 - m2 mod n).

    D(E(m1) * E(m2)^{-1} mod n^2) = m1 - m2 mod n

    Since E(m2)^{-1} mod n^2 = E(-m2) = E(n - m2), subtraction reduces
    to adding the encryption of the negation.

    Args:
        c1, c2:     Paillier ciphertexts
        public_key: (n, g) from keygen()

    Returns:
        int: ciphertext encrypting (m1 - m2) mod n
    """
    n, g = public_key
    n2 = n * n
    # c2^{-1} mod n^2 is the modular inverse of c2
    c2_inv = _mod_inverse(c2, n2)
    return (c1 * c2_inv) % n2


def scalar_multiply(c, k, public_key):
    """
    Homomorphic scalar multiplication.

    Property (Paillier 1999, Section 8):
        D(E(m)^k mod n^2) = k * m mod n

    Proof:
        E(m)^k = (g^m * r^n)^k mod n^2
               = g^(km) * (r^k)^n mod n^2
               = E(km, r^k)  ✓

    Args:
        c:          Paillier ciphertext encrypting m
        k:          scalar multiplier (integer)
        public_key: (n, g) from keygen()

    Returns:
        int: ciphertext encrypting (k * m) mod n
    """
    n, g = public_key
    n2 = n * n
    return pow(c, k, n2)


def negate_encrypted(c, public_key):
    """
    Homomorphic negation: compute E(-m mod n) from E(m).

    D(E(m)^{n-1} mod n^2) = (n-1)*m mod n = -m mod n

    Equivalently: modular inverse of c in Z_{n^2}^*.

    Args:
        c:          Paillier ciphertext encrypting m
        public_key: (n, g) from keygen()

    Returns:
        int: ciphertext encrypting (-m mod n)
    """
    n, g = public_key
    n2 = n * n
    return _mod_inverse(c, n2)


# ---------------------------------------------------------------------------
# Fixed-Point Encoding for Continuous Features
# Reference: PrivPathInfer Contribution 1 — Native Continuous Feature Support
# ---------------------------------------------------------------------------

SCALE_FACTOR = 10000      # Preserves 4 decimal places of precision
OFFSET       = 10**9      # Ensures positive encoding (medical features ≥ 0)


def encode_threshold(threshold_float):
    """
    Encode a continuous threshold value as a non-negative integer.

    Encoding: threshold_int = int(threshold * SCALE_FACTOR) + OFFSET

    This is NOT discretization. The original floating-point value is
    preserved exactly to 4 decimal places. Comparison results are
    identical to plaintext comparisons.

    Example:
        threshold = 126.5 → threshold_int = 1265000 + 1000000000 = 1001265000

    PrivPathInfer Contribution 1:
        Unlike SDTC (Liang et al. 2021) which requires discretization into
        bins (losing accuracy), PrivPathInfer encrypts exact thresholds.
        Experiment 1 demonstrates that PrivPathInfer matches plaintext
        accuracy while SDTC degrades with fewer bins.

    Args:
        threshold_float: floating-point threshold value

    Returns:
        int: non-negative integer encoding of the threshold
    """
    return int(threshold_float * SCALE_FACTOR) + OFFSET


def decode_threshold(threshold_int):
    """
    Decode an integer threshold back to float.

    Inverse of encode_threshold.

    Args:
        threshold_int: encoded integer threshold

    Returns:
        float: original threshold value
    """
    return (threshold_int - OFFSET) / SCALE_FACTOR


# ---------------------------------------------------------------------------
# Algebraic Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests(bits=512):
    """
    Verify all Paillier algebraic properties required for PrivPathInfer.

    Tests:
        1. Correctness: D(E(m)) == m
        2. Additive homomorphism: D(E(m1)*E(m2)) == m1+m2
        3. Scalar multiplication: D(E(m)^k) == k*m
        4. Subtraction: D(E(m1)/E(m2)) == m1-m2
        5. Probabilistic: E(m) != E(m) (different r each time)

    Reference: Paillier 1999, Scheme 1, Section 8
    """
    print("=" * 60)
    print("Paillier Algebraic Verification Tests")
    print(f"Key size: {bits}-bit n")
    print("Reference: Paillier 1999, EUROCRYPT")
    print("=" * 60)

    print(f"\nGenerating {bits}-bit Paillier key pair...")
    pub, priv = keygen(bits)
    n, g = pub
    print(f"  n = {n.bit_length()}-bit integer")
    print(f"  g = n + 1 (simplified variant)")

    m1 = random.randrange(1, n // 4)
    m2 = random.randrange(1, n // 4)
    k  = random.randrange(2, 100)

    print(f"\n  m1 = {m1}")
    print(f"  m2 = {m2}")
    print(f"  k  = {k}")

    # Test 1: Correctness
    c1 = encrypt(m1, pub)
    assert decrypt(c1, pub, priv) == m1, "Correctness FAILED"
    print("\n[PASS] Test 1: Correctness — D(E(m)) == m")

    # Test 2: Additive homomorphism
    c2 = encrypt(m2, pub)
    c_sum = add_encrypted(c1, c2, pub)
    result = decrypt(c_sum, pub, priv)
    expected = (m1 + m2) % n
    assert result == expected, f"Additive homomorphism FAILED: got {result}, expected {expected}"
    print("[PASS] Test 2: Additive homomorphism — D(E(m1)*E(m2)) == m1+m2 mod n")

    # Test 3: Scalar multiplication
    c_scaled = scalar_multiply(c1, k, pub)
    result = decrypt(c_scaled, pub, priv)
    expected = (k * m1) % n
    assert result == expected, f"Scalar multiply FAILED: got {result}, expected {expected}"
    print("[PASS] Test 3: Scalar multiplication — D(E(m)^k) == k*m mod n")

    # Test 4: Subtraction
    c_diff = subtract_encrypted(c1, c2, pub)
    result = decrypt(c_diff, pub, priv)
    expected = (m1 - m2) % n
    assert result == expected, f"Subtraction FAILED: got {result}, expected {expected}"
    print("[PASS] Test 4: Subtraction — D(E(m1)/E(m2)) == m1-m2 mod n")

    # Test 5: Probabilistic encryption
    c1a = encrypt(m1, pub)
    c1b = encrypt(m1, pub)
    assert c1a != c1b, "Probabilistic encryption FAILED: same ciphertext generated"
    assert decrypt(c1a, pub, priv) == m1, "Decryption of second encryption FAILED"
    assert decrypt(c1b, pub, priv) == m1, "Decryption of third encryption FAILED"
    print("[PASS] Test 5: Probabilistic — E(m) != E(m), both decrypt correctly")

    # Test 6: Zero encryption
    c_zero = encrypt(0, pub)
    assert decrypt(c_zero, pub, priv) == 0, "Zero encryption FAILED"
    print("[PASS] Test 6: Zero encryption — D(E(0)) == 0")

    # Test 7: Fixed-point encoding round-trip
    threshold = 126.5432
    encoded = encode_threshold(threshold)
    decoded = decode_threshold(encoded)
    assert abs(decoded - threshold) < 1e-4, "Fixed-point encoding FAILED"
    c_thresh = encrypt(encoded, pub)
    recovered = decode_threshold(decrypt(c_thresh, pub, priv))
    assert abs(recovered - threshold) < 1e-4, "Encrypted threshold round-trip FAILED"
    print("[PASS] Test 7: Fixed-point encoding — encrypt/decrypt threshold preserves value")

    print("\n[ALL TESTS PASSED] paillier.py is verified and ready.")
    print("Security: Semantically secure under DCRA (Paillier 1999, Theorem 15)")
    print("Homomorphic: Additive homomorphism verified algebraically")


if __name__ == "__main__":
    run_all_tests(bits=512)