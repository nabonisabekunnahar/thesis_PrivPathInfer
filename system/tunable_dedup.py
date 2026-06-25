"""
tunable_dedup.py - Secure Threshold Deduplication with a Storage-Leakage Knob
=============================================================================
PrivPathInfer: Contribution 4 (Secure Threshold Deduplication)

Problem
-------
Full threshold deduplication (dedup_encryptor.py) encrypts each distinct
(feature, threshold) pair ONCE and lets every path reference it by a stored
index. This minimises Paillier-ciphertext storage, but the shared indices
reveal the *threshold-equality pattern*: the honest-but-curious cloud learns
which conditions across the model test the same value, and (Proposition 2 of
the security analysis) this pattern reveals the tree topology.

The non-deduplicated baseline (rule_encryptor.py) hides this pattern, because
each condition is encrypted with fresh Paillier randomness, so equal plaintexts
yield unlinkable ciphertexts (IND-CPA under DCR, Paillier 1999, Theorem 15).

This module implements the tunable family Pi_c that interpolates between the two.

Mechanism (replication, NOT dummy insertion)
---------------------------------------------
A public knob c in {1,2,...} U {inf} bounds how many conditions may reference a
single stored ciphertext. For a value-pair v with reuse count R:
    q(v,c) = ceil(R / c)      independent Paillier ciphertexts of v
created with fresh randomness (hence pairwise IND-CPA-unlinkable, Paillier 1999
Section 4), and v's R references are distributed across these q copies in
buckets of size <= c.

    c = 1   -> q = R  for every v  -> S(1) = T (total conditions) = baseline,
              no threshold-equality leakage.
    c = inf -> q = 1  for every v  -> S(inf) = U (unique values) = full dedup,
              full threshold-equality leakage (tree topology).

Honest scope (Remark 1 of the security analysis)
-------------------------------------------------
The PRP feature tag is deterministic and part of the baseline leakage. If a
feature carries only ONE distinct threshold in the tree, feature-equality
implies value-equality, so its replicas remain re-linkable through the feature
tag regardless of c. Splitting such a value-pair wastes storage for zero
privacy gain, so Pi_c replicates ONLY value-pairs whose feature carries >= 2
distinct thresholds. The reported leakage metric accounts for this floor.

References
----------
    Paillier, P. EUROCRYPT 1999 (IND-CPA under DCR; fresh-randomness
        unlinkability of equal plaintexts).
    Curtmola et al., J. Computer Security 2011 (leakage-function framework).
    dedup_encryptor.py (the c = inf endpoint of this family).

Security assumption: Paillier semantic security (IND-CPA) under the Decisional
Composite Residuosity assumption; pseudorandomness of the AES-based PRF/PRP.

Author: Mst Sabekunnahar Naboni (Roll: 2007034), BSc CSE, KUET - Thesis CSE 4000
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

from crypto.paillier import keygen, encrypt, decrypt, encode_threshold


# ---------------------------------------------------------------------------
# Storage constants (identical to dedup_encryptor.py for a fair comparison)
# ---------------------------------------------------------------------------
REF_BYTES   = 8     # one integer reference per condition-position
LABEL_BYTES = 2     # one class label per path


def ciphertext_bytes(paillier_bits: int) -> int:
    """Size of one Paillier ciphertext (an element of Z*_{n^2})."""
    return paillier_bits // 4   # |n^2| bits = 2*|n| bits = paillier_bits/4 bytes


def _value_key(feature_idx: int, threshold: float) -> Tuple[int, float]:
    """Deduplication key, matching dedup_encryptor.py exactly."""
    return (feature_idx, round(threshold, 6))


def _dir_symbol(direction: str) -> str:
    return '<=' if direction == 'left' else '>'


# ---------------------------------------------------------------------------
# Data structures (what the cloud stores)
# ---------------------------------------------------------------------------

@dataclass
class ReplicatedEntry:
    """One stored Paillier ciphertext. Referenced by <= c condition-positions."""
    cipher_id:     int
    enc_threshold: int    # Paillier ciphertext (the only value-bearing field)
    feature_tag:   int    # PRP(feature) in the full system; raw feature_idx here
    # NOTE: plaintext threshold is NEVER stored cloud-side; kept MI-side only.


@dataclass
class TunablePath:
    path_id:        int
    condition_refs: List[Tuple[int, str]]   # (cipher_id, direction-symbol)
    label:          int
    depth:          int


@dataclass
class TunableModel:
    cipher_table: Dict[int, ReplicatedEntry]
    paths:        List[TunablePath]
    pub_key:      Tuple[int, int]
    c:            float
    stats:        Dict


# ---------------------------------------------------------------------------
# Analytic profile  (counts + storage + leakage; no cryptography, exact + fast)
# ---------------------------------------------------------------------------

def profile(paths, c, paillier_bits: int = 1024,
            skip_feature_unique: bool = False) -> Dict:
    """
    Exact storage and leakage of Pi_c for the given paths, computed analytically.

    Storage (Theorem, part b) -- BASE scheme replicates every value:
        S(c) = sum_v q(v,c),  q(v,c) = ceil(R_v / c)   (1 if c = inf)
        bytes = S(c)*ciphertext_bytes + T*REF_BYTES + n_paths*LABEL_BYTES
        Clean endpoints: S(1) = T, S(inf) = U.

    Optional Remark-1 optimization (skip_feature_unique=True): a value whose
    feature carries only ONE distinct threshold is re-linkable via the PRP
    feature tag regardless of c, so replicating it buys no privacy. Skipping its
    replication (q=1) saves storage with ZERO change in leakage. This lowers
    S(1) below T whenever such values exist; on PIMA none exist (floor=1), so
    the two settings coincide.

    Leakage (max value-linkability lambda(c)): the largest number of conditions
    the cloud can link as testing one value. For a value-pair v with reuse R:
        feature has >= 2 thresholds -> min(R, c)   (capped by the knob)
        feature-unique threshold    -> R           (re-linkable via feature tag)
    lambda(c) = max over v of the above. Lower is more private.
        lambda(1)  = baseline floor (1 unless a feature is single-threshold)
        lambda(inf)= max_v R = full threshold-equality pattern (topology).

    Args:
        paths: list of LeafPath from PathExtractor.
        c:     int >= 1, or math.inf for full deduplication.
        skip_feature_unique: apply the Remark-1 storage optimization.
    """
    # reuse count R_v per value-pair, and distinct thresholds per feature
    reuse: Dict[Tuple[int, float], int] = {}
    feat_thresholds: Dict[int, set] = {}
    for p in paths:
        for cond in p.conditions:
            vk = _value_key(cond.feature_idx, cond.threshold)
            reuse[vk] = reuse.get(vk, 0) + 1
            feat_thresholds.setdefault(cond.feature_idx, set()).add(round(cond.threshold, 6))

    U = len(reuse)
    T = sum(reuse.values())
    n_paths = len(paths)

    def q(vk):
        R = reuse[vk]
        if c == math.inf:
            return 1
        if skip_feature_unique and len(feat_thresholds[vk[0]]) < 2:
            return 1
        return math.ceil(R / c)

    def link(vk):
        feat = vk[0]
        splittable = len(feat_thresholds[feat]) >= 2
        R = reuse[vk]
        if not splittable:
            return R                 # re-linkable via feature tag, knob has no effect
        return R if c == math.inf else min(R, c)

    S = sum(q(vk) for vk in reuse)
    lam = max((link(vk) for vk in reuse), default=0)
    # is the leakage floor set by a feature-unique threshold (not by c)?
    floor = max((reuse[vk] for vk in reuse if len(feat_thresholds[vk[0]]) < 2), default=1)

    ct_bytes = ciphertext_bytes(paillier_bits)
    total_bytes = S * ct_bytes + T * REF_BYTES + n_paths * LABEL_BYTES

    return {
        'c':                  ('inf' if c == math.inf else int(c)),
        'unique_values_U':    U,
        'total_conditions_T': T,
        'n_paths':            n_paths,
        'ciphertexts_S':      S,
        'storage_bytes':      total_bytes,
        'storage_kb':         round(total_bytes / 1024, 2),
        'max_linkability':    lam,         # lambda(c): lower = more private
        'leakage_floor':      floor,       # forced by feature-unique thresholds
        'max_reuse':          max(reuse.values(), default=0),
    }


# ---------------------------------------------------------------------------
# Real encryptor (builds an actual encrypted model; for correctness + demo)
# ---------------------------------------------------------------------------

class TunableDedupEncryptor:
    """
    Builds a real Pi_c encrypted model and verifies correctness/unlinkability.

    One Paillier keypair is reused across all ciphertexts. Every replica of a
    value is a *fresh* encryption (encrypt() draws fresh randomness r), so
    replicas of one value are pairwise unlinkable under IND-CPA.
    """

    def __init__(self, paillier_bits: int = 1024, _keys=None):
        self.bits = paillier_bits
        if _keys is not None:
            self.pub, self.priv = _keys
        else:
            self.pub, self.priv = keygen(paillier_bits)

    def build(self, paths, c, skip_feature_unique: bool = False) -> TunableModel:
        # group condition-positions by value-pair, recording (path_idx, cond)
        groups: Dict[Tuple[int, float], List] = {}
        feat_thresholds: Dict[int, set] = {}
        for pi, p in enumerate(paths):
            for ci, cond in enumerate(p.conditions):
                vk = _value_key(cond.feature_idx, cond.threshold)
                groups.setdefault(vk, []).append((pi, ci, cond))
                feat_thresholds.setdefault(cond.feature_idx, set()).add(round(cond.threshold, 6))

        cipher_table: Dict[int, ReplicatedEntry] = {}
        ref_of: Dict[Tuple[int, int], int] = {}   # (path_idx, cond_idx) -> cipher_id
        next_id = 0

        for vk, members in groups.items():
            feat, _ = vk
            R = len(members)
            if c == math.inf:
                q = 1
            elif skip_feature_unique and len(feat_thresholds[feat]) < 2:
                q = 1
            else:
                q = math.ceil(R / c)

            # create q fresh ciphertexts of this value
            thr_int = encode_threshold(members[0][2].threshold)
            cipher_ids = []
            for _ in range(q):
                cid = next_id; next_id += 1
                cipher_table[cid] = ReplicatedEntry(
                    cipher_id=cid,
                    enc_threshold=encrypt(thr_int, self.pub),   # fresh randomness
                    feature_tag=feat,
                )
                cipher_ids.append(cid)

            # distribute R references across the q ciphertexts, buckets <= c
            for k, (pi, ci, _) in enumerate(members):
                ref_of[(pi, ci)] = cipher_ids[k % q]

        # build path reference lists
        tpaths = []
        for pi, p in enumerate(paths):
            refs = [(ref_of[(pi, ci)], _dir_symbol(cond.direction))
                    for ci, cond in enumerate(p.conditions)]
            tpaths.append(TunablePath(pi, refs, p.label, p.depth))

        stats = profile(paths, c, self.bits, skip_feature_unique)
        assert stats['ciphertexts_S'] == len(cipher_table), "S(c) mismatch"
        return TunableModel(cipher_table, tpaths, self.pub, c, stats)

    def verify_correctness(self, model: TunableModel, original_paths) -> bool:
        """
        Decrypt every referenced ciphertext and confirm each path reconstructs to
        its ORIGINAL (feature, threshold, direction) sequence. Identical
        reconstruction => identical classification for all inputs (Theorem, a).
        """
        for tp, op in zip(model.paths, original_paths):
            if len(tp.condition_refs) != len(op.conditions):
                return False
            for (cid, dsym), cond in zip(tp.condition_refs, op.conditions):
                m = decrypt(model.cipher_table[cid].enc_threshold, model.pub_key, self.priv)
                if m != encode_threshold(cond.threshold):
                    return False
                if dsym != _dir_symbol(cond.direction):
                    return False
        return True

    def check_unlinkability(self, model: TunableModel) -> bool:
        """Replicas of one value must be DISTINCT ciphertexts (fresh randomness)."""
        seen = {}
        for e in model.cipher_table.values():
            seen.setdefault(e.enc_threshold, 0)
            seen[e.enc_threshold] += 1
        return all(v == 1 for v in seen.values())