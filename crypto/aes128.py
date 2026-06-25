"""
aes128.py — AES-128 Block Cipher Implementation from Scratch
=============================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Implementation Reference:
    FIPS 197: Advanced Encryption Standard (AES)
    National Institute of Standards and Technology, 2001.
    https://csrc.nist.gov/publications/detail/fips/197/final

Security Assumption:
    AES-128 is modeled as a secure pseudorandom permutation (PRP) under
    the standard PRP assumption. By the PRF/PRP switching lemma
    (Boneh-Shoup, Section 4.4), AES-128 is computationally
    indistinguishable from a PRF over domain {0,1}^128, with
    distinguishing advantage bounded by Q^2 / 2^128, which is negligible
    for any practical number of queries Q.

Components Implemented:
    - GF(2^8) arithmetic (xtime and polynomial multiplication)
    - SubBytes transformation (S-Box)
    - ShiftRows transformation
    - MixColumns transformation
    - AddRoundKey transformation
    - Key Expansion (11 round keys for AES-128)
    - AES-128 Encryption and Decryption

Verification:
    Run against FIPS-197 Appendix B test vector:
        Key:   2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c
        Input: 32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34
        Output:39 25 84 1d 02 dc 09 fb dc 11 85 97 19 6a 0b 32

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os


# ---------------------------------------------------------------------------
# GF(2^8) Arithmetic
# Reference: FIPS 197, Section 4.2
# ---------------------------------------------------------------------------

def _xtime(a):
    """
    Multiply element a by x (i.e., 0x02) in GF(2^8).

    GF(2^8) uses the irreducible polynomial:
        m(x) = x^8 + x^4 + x^3 + x + 1  (0x11b in hex)

    If the high bit of a is set, we shift left and XOR with 0x1b
    (the low 8 bits of 0x11b) to reduce modulo m(x).

    Reference: FIPS 197, Section 4.2.1
    """
    if a & 0x80:
        return ((a << 1) ^ 0x1b) & 0xFF
    return (a << 1) & 0xFF


def _gf_mul(a, b):
    """
    Multiply two elements a, b in GF(2^8) using the Russian peasant algorithm.

    This implements polynomial multiplication modulo the AES irreducible
    polynomial m(x) = x^8 + x^4 + x^3 + x + 1.

    Algorithm: For each bit of b, if the bit is set, XOR the current
    value of a into the result, then double a using xtime.

    Reference: FIPS 197, Section 4.2.1
    """
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = _xtime(a)
        b >>= 1
    return result


# ---------------------------------------------------------------------------
# AES S-Box and Inverse S-Box
# Reference: FIPS 197, Section 5.1.1 (SubBytes) and 5.3.2 (InvSubBytes)
# ---------------------------------------------------------------------------

# AES S-Box: computed from GF(2^8) multiplicative inverse + affine transform
# Source: FIPS 197, Figure 7
_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

# AES Inverse S-Box
# Source: FIPS 197, Figure 14
_INV_SBOX = [0] * 256
for _i, _v in enumerate(_SBOX):
    _INV_SBOX[_v] = _i


# ---------------------------------------------------------------------------
# AES Round Constants (Rcon)
# Reference: FIPS 197, Section 5.2 (Key Expansion)
# Rcon[i] = x^(i-1) in GF(2^8), represented as a 4-byte word
# ---------------------------------------------------------------------------

_RCON = [
    0x00000000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b000000, 0x36000000,
]


# ---------------------------------------------------------------------------
# AES Transformations
# Reference: FIPS 197, Section 5.1
# ---------------------------------------------------------------------------

def _sub_bytes(state):
    """
    SubBytes: Apply S-Box substitution to each byte of the state.

    Each byte s[r][c] of the state is replaced by S-Box[s[r][c]].
    Provides non-linearity to the cipher.

    Reference: FIPS 197, Section 5.1.1
    """
    return [[_SBOX[state[r][c]] for c in range(4)] for r in range(4)]


def _inv_sub_bytes(state):
    """
    InvSubBytes: Apply inverse S-Box substitution.

    Reference: FIPS 197, Section 5.3.2
    """
    return [[_INV_SBOX[state[r][c]] for c in range(4)] for r in range(4)]


def _shift_rows(state):
    """
    ShiftRows: Cyclically shift each row of the state by its row index.

    Row 0: no shift
    Row 1: shift left by 1
    Row 2: shift left by 2
    Row 3: shift left by 3

    Provides diffusion across columns.

    Reference: FIPS 197, Section 5.1.2
    """
    return [
        [state[0][0], state[0][1], state[0][2], state[0][3]],
        [state[1][1], state[1][2], state[1][3], state[1][0]],
        [state[2][2], state[2][3], state[2][0], state[2][1]],
        [state[3][3], state[3][0], state[3][1], state[3][2]],
    ]


def _inv_shift_rows(state):
    """
    InvShiftRows: Cyclically shift each row right by its row index.

    Reference: FIPS 197, Section 5.3.1
    """
    return [
        [state[0][0], state[0][1], state[0][2], state[0][3]],
        [state[1][3], state[1][0], state[1][1], state[1][2]],
        [state[2][2], state[2][3], state[2][0], state[2][1]],
        [state[3][1], state[3][2], state[3][3], state[3][0]],
    ]


def _mix_columns(state):
    """
    MixColumns: Multiply each column by the fixed matrix in GF(2^8).

    The fixed matrix is:
        [ 2  3  1  1 ]
        [ 1  2  3  1 ]
        [ 1  1  2  3 ]
        [ 3  1  1  2 ]

    Each column is treated as a polynomial over GF(2^8) and multiplied
    by a(x) = {03}x^3 + {01}x^2 + {01}x + {02} modulo x^4 + 1.

    Reference: FIPS 197, Section 5.1.3
    """
    new_state = [[0]*4 for _ in range(4)]
    for c in range(4):
        s0, s1, s2, s3 = state[0][c], state[1][c], state[2][c], state[3][c]
        new_state[0][c] = (_gf_mul(0x02, s0) ^ _gf_mul(0x03, s1) ^ s2 ^ s3)
        new_state[1][c] = (s0 ^ _gf_mul(0x02, s1) ^ _gf_mul(0x03, s2) ^ s3)
        new_state[2][c] = (s0 ^ s1 ^ _gf_mul(0x02, s2) ^ _gf_mul(0x03, s3))
        new_state[3][c] = (_gf_mul(0x03, s0) ^ s1 ^ s2 ^ _gf_mul(0x02, s3))
    return new_state


def _inv_mix_columns(state):
    """
    InvMixColumns: Multiply each column by the inverse matrix in GF(2^8).

    The inverse matrix is:
        [ 14  11  13   9 ]
        [  9  14  11  13 ]
        [ 13   9  14  11 ]
        [ 11  13   9  14 ]

    Reference: FIPS 197, Section 5.3.3
    """
    new_state = [[0]*4 for _ in range(4)]
    for c in range(4):
        s0, s1, s2, s3 = state[0][c], state[1][c], state[2][c], state[3][c]
        new_state[0][c] = (_gf_mul(0x0e, s0) ^ _gf_mul(0x0b, s1) ^
                           _gf_mul(0x0d, s2) ^ _gf_mul(0x09, s3))
        new_state[1][c] = (_gf_mul(0x09, s0) ^ _gf_mul(0x0e, s1) ^
                           _gf_mul(0x0b, s2) ^ _gf_mul(0x0d, s3))
        new_state[2][c] = (_gf_mul(0x0d, s0) ^ _gf_mul(0x09, s1) ^
                           _gf_mul(0x0e, s2) ^ _gf_mul(0x0b, s3))
        new_state[3][c] = (_gf_mul(0x0b, s0) ^ _gf_mul(0x0d, s1) ^
                           _gf_mul(0x09, s2) ^ _gf_mul(0x0e, s3))
    return new_state


def _add_round_key(state, round_key):
    """
    AddRoundKey: XOR the state with the round key.

    Each byte of the state is XORed with the corresponding byte of the
    round key. This is the only step that uses the key material directly.

    Reference: FIPS 197, Section 5.1.4
    """
    return [[state[r][c] ^ round_key[r][c] for c in range(4)]
            for r in range(4)]


# ---------------------------------------------------------------------------
# Key Expansion
# Reference: FIPS 197, Section 5.2
# ---------------------------------------------------------------------------

def _key_expansion(key_bytes):
    """
    Expand a 16-byte AES-128 key into 11 round keys (176 bytes total).

    AES-128 uses 10 rounds, requiring 11 round keys of 16 bytes each.
    The key schedule expands the original key into a key schedule array
    W[0..43] of 4-byte words.

    Key schedule operations:
        - SubWord: Apply S-Box to each byte of a word
        - RotWord: Cyclic left shift of a word by 1 byte
        - Rcon[i]: Round constant for round i

    Reference: FIPS 197, Section 5.2, Figure 11

    Args:
        key_bytes: list of 16 integers (0-255), the AES-128 key

    Returns:
        List of 11 round keys, each a 4x4 matrix of bytes
    """
    assert len(key_bytes) == 16, "AES-128 requires a 16-byte key"

    def sub_word(word):
        return ((_SBOX[(word >> 24) & 0xFF] << 24) |
                (_SBOX[(word >> 16) & 0xFF] << 16) |
                (_SBOX[(word >>  8) & 0xFF] <<  8) |
                (_SBOX[(word      ) & 0xFF]      ))

    def rot_word(word):
        return ((word << 8) | (word >> 24)) & 0xFFFFFFFF

    # Initialize W[0..3] from the original key
    W = []
    for i in range(4):
        W.append((key_bytes[4*i]   << 24) |
                 (key_bytes[4*i+1] << 16) |
                 (key_bytes[4*i+2] <<  8) |
                 (key_bytes[4*i+3]      ))

    # Expand to 44 words
    for i in range(4, 44):
        temp = W[i-1]
        if i % 4 == 0:
            temp = sub_word(rot_word(temp)) ^ _RCON[i // 4]
        W.append(W[i-4] ^ temp)

    # Convert W into 11 round keys, each a 4x4 state matrix
    round_keys = []
    for rnd in range(11):
        rk = [[0]*4 for _ in range(4)]
        for c in range(4):
            word = W[rnd*4 + c]
            for r in range(4):
                rk[r][c] = (word >> (24 - 8*r)) & 0xFF
        round_keys.append(rk)

    return round_keys


# ---------------------------------------------------------------------------
# AES-128 Encrypt and Decrypt
# Reference: FIPS 197, Section 5.1 (Encrypt), Section 5.3 (Decrypt)
# ---------------------------------------------------------------------------

def _bytes_to_state(block):
    """
    Convert a 16-byte block into a 4x4 AES state matrix.

    AES state is column-major: bytes fill columns from top to bottom.
    state[row][col] = block[row + 4*col]

    Reference: FIPS 197, Section 3.4
    """
    state = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            state[r][c] = block[r + 4*c]
    return state


def _state_to_bytes(state):
    """
    Convert a 4x4 AES state matrix back to a 16-byte list.

    Reference: FIPS 197, Section 3.4
    """
    block = []
    for c in range(4):
        for r in range(4):
            block.append(state[r][c])
    return block


def aes_encrypt_block(plaintext_bytes, key_bytes):
    """
    Encrypt a single 16-byte block using AES-128.

    AES-128 performs 10 rounds:
        - 1 initial AddRoundKey
        - 9 full rounds: SubBytes, ShiftRows, MixColumns, AddRoundKey
        - 1 final round: SubBytes, ShiftRows, AddRoundKey (no MixColumns)

    Security: AES-128 is modeled as a secure PRP (Boneh-Shoup, Def 4.1).
    By the PRF/PRP switching lemma, it is also a secure PRF for use in
    the PrivPathInfer token generation protocol.

    Reference: FIPS 197, Section 5.1, Figure 5

    Args:
        plaintext_bytes:  list of 16 integers (0-255)
        key_bytes:        list of 16 integers (0-255)

    Returns:
        list of 16 integers (0-255): the ciphertext block
    """
    assert len(plaintext_bytes) == 16, "AES block must be 16 bytes"
    assert len(key_bytes) == 16, "AES-128 key must be 16 bytes"

    round_keys = _key_expansion(key_bytes)
    state = _bytes_to_state(plaintext_bytes)

    # Initial round key addition
    state = _add_round_key(state, round_keys[0])

    # Rounds 1-9: full rounds
    for rnd in range(1, 10):
        state = _sub_bytes(state)
        state = _shift_rows(state)
        state = _mix_columns(state)
        state = _add_round_key(state, round_keys[rnd])

    # Round 10: final round (no MixColumns)
    state = _sub_bytes(state)
    state = _shift_rows(state)
    state = _add_round_key(state, round_keys[10])

    return _state_to_bytes(state)


def aes_decrypt_block(ciphertext_bytes, key_bytes):
    """
    Decrypt a single 16-byte block using AES-128.

    Inverse cipher applies transformations in reverse order:
        InvShiftRows, InvSubBytes, AddRoundKey, InvMixColumns

    Reference: FIPS 197, Section 5.3, Figure 12

    Args:
        ciphertext_bytes: list of 16 integers (0-255)
        key_bytes:        list of 16 integers (0-255)

    Returns:
        list of 16 integers (0-255): the plaintext block
    """
    assert len(ciphertext_bytes) == 16, "AES block must be 16 bytes"
    assert len(key_bytes) == 16, "AES-128 key must be 16 bytes"

    round_keys = _key_expansion(key_bytes)
    state = _bytes_to_state(ciphertext_bytes)

    # Initial round key addition (with last round key)
    state = _add_round_key(state, round_keys[10])

    # Rounds 9-1: inverse full rounds
    for rnd in range(9, 0, -1):
        state = _inv_shift_rows(state)
        state = _inv_sub_bytes(state)
        state = _add_round_key(state, round_keys[rnd])
        state = _inv_mix_columns(state)

    # Final inverse round (no InvMixColumns)
    state = _inv_shift_rows(state)
    state = _inv_sub_bytes(state)
    state = _add_round_key(state, round_keys[0])

    return _state_to_bytes(state)


# ---------------------------------------------------------------------------
# Convenience: bytes/int interface
# ---------------------------------------------------------------------------

def aes_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt a 16-byte plaintext block with a 16-byte key.

    Args:
        plaintext: bytes of length 16
        key:       bytes of length 16

    Returns:
        bytes of length 16: AES-128 ciphertext
    """
    pt = list(plaintext)
    kt = list(key)
    ct = aes_encrypt_block(pt, kt)
    return bytes(ct)


def aes_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypt a 16-byte ciphertext block with a 16-byte key.

    Args:
        ciphertext: bytes of length 16
        key:        bytes of length 16

    Returns:
        bytes of length 16: AES-128 plaintext
    """
    ct = list(ciphertext)
    kt = list(key)
    pt = aes_decrypt_block(ct, kt)
    return bytes(pt)


# ---------------------------------------------------------------------------
# NIST Test Vector Verification
# Reference: FIPS 197, Appendix B
# ---------------------------------------------------------------------------

def verify_nist_test_vector():
    """
    Verify correctness against the FIPS-197 Appendix B test vector.

    Key:    2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c
    Input:  32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34
    Output: 39 25 84 1d 02 dc 09 fb dc 11 85 97 19 6a 0b 32

    Returns:
        True if test vector passes, raises AssertionError otherwise
    """
    key = bytes([
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    ])
    plaintext = bytes([
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
    ])
    expected = bytes([
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
        0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32
    ])

    ciphertext = aes_encrypt(plaintext, key)
    assert ciphertext == expected, (
        f"NIST test vector FAILED:\n"
        f"  Expected: {expected.hex()}\n"
        f"  Got:      {ciphertext.hex()}"
    )

    # Verify decryption
    recovered = aes_decrypt(ciphertext, key)
    assert recovered == plaintext, (
        f"AES decryption FAILED:\n"
        f"  Expected: {plaintext.hex()}\n"
        f"  Got:      {recovered.hex()}"
    )

    print("[PASS] FIPS-197 Appendix B test vector verified.")
    print(f"  Key:        {key.hex()}")
    print(f"  Plaintext:  {plaintext.hex()}")
    print(f"  Ciphertext: {ciphertext.hex()}")
    return True


# ---------------------------------------------------------------------------
# Additional Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all AES-128 verification tests."""

    # Test 1: NIST test vector
    verify_nist_test_vector()

    # Test 2: Encrypt-decrypt roundtrip with random key and plaintext
    key = os.urandom(16)
    pt  = os.urandom(16)
    ct  = aes_encrypt(pt, key)
    recovered = aes_decrypt(ct, key)
    assert recovered == pt, "Roundtrip test FAILED"
    print("[PASS] Random roundtrip test passed.")

    # Test 3: Different keys produce different ciphertexts
    key2 = os.urandom(16)
    ct2  = aes_encrypt(pt, key2)
    assert ct != ct2, "Different keys should produce different ciphertexts"
    print("[PASS] Key sensitivity test passed.")

    # Test 4: Different plaintexts produce different ciphertexts
    pt2 = os.urandom(16)
    ct3 = aes_encrypt(pt2, key)
    assert ct != ct3, "Different plaintexts should produce different ciphertexts"
    print("[PASS] Plaintext sensitivity test passed.")

    # Test 5: Zero key and zero plaintext (edge case)
    zero_key = bytes(16)
    zero_pt  = bytes(16)
    ct_zero  = aes_encrypt(zero_pt, zero_key)
    assert aes_decrypt(ct_zero, zero_key) == zero_pt, "Zero roundtrip FAILED"
    print("[PASS] Zero key/plaintext edge case passed.")

    print("\n[ALL TESTS PASSED] aes128.py is verified and ready.")


if __name__ == "__main__":
    print("Running AES tests...")
    run_all_tests()