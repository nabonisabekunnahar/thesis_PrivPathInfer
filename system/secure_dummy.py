"""
secure_dummy.py - Realistic, Inert Dummy Generation (PRIORITY A fix)
====================================================================
PrivPathInfer: hardens Contribution 3's batch-padding security.

The hole
--------
The original make_dummy_rule() pads update batches with rules using the
impossible condition  feature_0 <= -100, fixed feature index 0, and path_id -1.
This is distinguishable from real rules on THREE cloud-visible channels:

  1. enc_feature_idx = PRP(0) is identical for every dummy  -> a cluster of
     byte-identical feature tags outs the dummies (the PRP is deterministic).
  2. path_id = -1 is stored in plaintext              -> a constant sentinel.
  3. The condition is never satisfied by any real query -> across queries its
     comparison bit never flips, unlike real thresholds.

The original Theorem-4 argument only covers the PRF deletion-token channel and
says nothing about (1)-(3); the insert channel was never proven indistinguishable.

The fix
-------
Generate each dummy as a FRESH re-encryption of a real path chosen from the
model's own path distribution:
  - every condition uses a REAL (feature, threshold, direction), so each field is
    drawn from exactly the real-insert distribution (closes channels 1 and 3);
  - the dummy path gets a fresh path_id in the real range (closes channel 2);
  - fresh Paillier randomness makes the copy IND-CPA-unlinkable to its source.

Inertness: duplicating a path that returns label L for a region leaves that
region returning L, so classification is unchanged for every input. Dummies are
thus statistically identical to real inserts AND provably inert.

Honest residual: a duplicated path adds one more path to its label. Sampling the
source path uniformly from the model's paths keeps the dummy path distribution
equal to the model's own, so no per-label skew is introduced relative to the
model; over many updates the only growth is storage, disclosed as L_update's
batch_size already permits.

Author: Mst Sabekunnahar Naboni (Roll: 2007034), BSc CSE, KUET - Thesis CSE 4000
"""

import random
from dataclasses import replace
from typing import List

from system.path_extractor import LeafPath, PathCondition
from system.rule_encryptor import EncryptedRule, RuleEncryptor


class RealisticDummyGenerator:
    """
    Produces dummy paths/rules that match the real-insert distribution and are
    inert by construction (re-encrypted duplicates of real paths).
    """

    def __init__(self, real_paths: List[LeafPath], encryptor: RuleEncryptor,
                 seed: int = None):
        assert real_paths, "need at least one real path to sample from"
        self.real_paths = real_paths
        self.enc = encryptor
        self.rng = random.Random(seed)
        # fresh path_ids drawn ABOVE the existing range (indistinguishable from
        # a genuinely new path added during retraining)
        self._next_pid = max(p.path_id for p in real_paths) + 1

    def make_dummy_path(self) -> LeafPath:
        """A plaintext dummy path = a copy of a random real path, new path_id."""
        src = self.rng.choice(self.real_paths)
        pid = self._next_pid
        self._next_pid += 1
        return LeafPath(
            path_id=pid,
            conditions=[replace(c) for c in src.conditions],
            label=src.label,            # same label -> inert
            leaf_id=-1,
            depth=src.depth,
        )

    def encrypt_dummy_path(self, dpath: LeafPath) -> List[EncryptedRule]:
        """Encrypt a dummy path exactly like a real path (fresh randomness)."""
        return self.enc.encrypt_paths([dpath])

    def make_dummy_inserts(self, n: int) -> List[EncryptedRule]:
        """
        Produce exactly n dummy insert rules whose fields are realistic.
        Fills by duplicating whole real paths; a trailing partial path emits
        only NON-terminal conditions (is_last=False, label=None), which never
        complete a path and so never return a label -> inert.
        """
        out: List[EncryptedRule] = []
        while len(out) < n:
            rules = self.encrypt_dummy_path(self.make_dummy_path())
            for r in rules:
                if len(out) < n:
                    out.append(r)
                else:
                    break
        # any trailing rule that is the last of a truncated path must not carry a
        # label (would create a spurious returnable leaf); neutralize it.
        if out and out[-1].is_last:
            out[-1] = replace(out[-1], is_last=False, label=None)
        return out


# ---------------------------------------------------------------------------
# Plaintext classifier (MI-side, for the inertness check only)
# ---------------------------------------------------------------------------

def classify(paths: List[LeafPath], x) -> int:
    """Return the label of the first path whose conditions x satisfies."""
    for p in paths:
        ok = True
        for c in p.conditions:
            sat = (x[c.feature_idx] <= c.threshold) if c.direction == 'left' \
                  else (x[c.feature_idx] > c.threshold)
            if not sat:
                ok = False
                break
        if ok and p.conditions:           # complete, satisfied path
            return p.label
    # fall back: a leaf with no conditions (degenerate tree)
    return paths[0].label if paths else 0
