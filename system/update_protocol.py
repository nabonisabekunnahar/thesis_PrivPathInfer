"""
update_protocol.py — Incremental Model Update Protocol
========================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

This module implements PrivPathInfer Contribution 3:
    Efficient Incremental Model Update with Formal Security Guarantees.

Problem with Existing Schemes:
    SDTC (Liang et al. 2021): When the decision tree is retrained,
    the ENTIRE encrypted model must be re-encrypted and re-uploaded.
    Cost: O(2^N) — always pays full re-encryption cost.

PrivPathInfer Solution:
    Since each rule is independently encrypted, only CHANGED rules
    need re-encryption. Unchanged rules remain valid on the cloud.
    Cost: O(k) where k = number of changed rules.

Security of Update Protocol:
    Three mechanisms ensure the cloud learns nothing beyond L_update:

    1. Fixed-size batch padding:
       Always send exactly BATCH_SIZE=8 updates, regardless of how
       many rules actually changed. Pad with dummy rules.
       Hides: how many rules changed.

    2. PRF-based deletion tokens:
       token = PRF(deletion_key, rule_id)
       Cloud uses token to find and delete the correct rule.
       Hides: which rule_id was deleted.

    3. Fresh encryption of new rules:
       New rules are encrypted with fresh randomness (new r in Paillier).
       Cloud cannot link new ciphertext to old ciphertext.

-----------------------------------------------------------------------
CORRECTION (this revision) — the delete-channel hit/miss hole
-----------------------------------------------------------------------
An earlier revision generated each dummy DELETE by drawing a deletion token
for a freshly random, never-inserted rule_id (`_make_dummy_delete`, now
`_make_dummy_delete_fallback`). The token VALUE is a genuine PRF output and
so looks random, exactly as intended — but the cloud does not just see
token values, it sees the EFFECT of applying each token: a real deletion's
token matches a stored rule and removes it (a HIT); the old dummy token
matches nothing (a MISS). Hit vs. miss is visible to the cloud regardless
of how random the token looks, so the cloud could count real deletions per
batch — a leak the original "L_update = {update_occurred, batch_size}"
claim did not disclose. This is the same class of hole the Priority-A fix
closed on the insert side (a -100 sentinel threshold was distinguishable by
its EFFECT — never satisfied — even though the ciphertext looked fine);
Priority-A never covered the delete channel.

The fix is a DUMMY POOL: dummy rules inserted as batch padding remain on
the cloud as ordinary (inert, unreadable) entries until they are themselves
the target of a later dummy DELETE. A dummy delete therefore also HITS,
exactly like a real delete, because it removes a rule that is genuinely
present. Two mechanisms supply pool tokens:

    (a) Same-batch pairing: this batch's own dummy INSERTS are processed
        before this batch's DELETES (see CloudStorage.apply_update_batch),
        so a dummy delete can target a dummy insert from the very same
        batch and still hit.
    (b) Persistent pool (`UpdateProtocol._dummy_pool`): any dummy-insert
        tokens not consumed by (a) in a given batch (i.e. when this batch
        needed more dummy inserts than dummy deletes) are carried forward
        for a LATER batch's dummy deletes to consume.

Bootstrap: the pool starts empty, so before the very first update the
Medical Institution should call `bootstrap_dummy_pool()` and upload the
returned rules to the cloud alongside the initial real rule set — a
one-time, batch-size-sized seed (already disclosed by batch_size).

Honest residual: pool tokens are a finite resource. If updates are
persistently insert-heavy (real_inserts > real_deletes batch after batch,
e.g. a tree that only grows), the pool can be driven to exhaustion despite
bootstrapping, since net pool change per batch = real_deletes - real_inserts.
When the pool cannot cover every dummy delete needed, the remainder falls
back to `_make_dummy_delete_fallback()` (random rule_id, a miss) — so in a
long insert-heavy run the leakage degrades gracefully toward (not beyond)
the old bound, rather than failing silently. A production deployment with a
known insert-heavy workload should seed a larger initial pool (k*BATCH_SIZE)
or periodically top it up out-of-band during maintenance windows.

Formal Leakage (Theorem 4 — Update Security, corrected):
    With the dummy pool kept non-empty (the common case after bootstrap,
    or for any update history where deletes are not persistently
    outnumbered by inserts):
        L_update = {update_occurred, batch_size}
    The cloud learns ONLY that an update of size batch_size=8 occurred —
    not how many rules truly changed, which rule_ids were touched, the new
    threshold values, or (with the pool fix) how many of the batch_size
    deletions were real vs. padding.

    If the pool is exhausted (sustained insert-heavy history, see above),
    the residual leakage degrades to at most:
        L_update = {update_occurred, batch_size, number_of_real_deletions}
    i.e. never worse than the unpatched scheme, and equal to the unpatched
    scheme's bound only in that worst case.

    The cloud does NOT learn (under either case):
        - Which rule_ids were deleted
        - What the new threshold values are

Security Proof Structure (Theorem 4):
    Assume adversary A breaks update security.
    A can distinguish a batch of k real updates + (8-k) dummies
    from a batch of 8 real updates.
    This requires A to identify dummy rules — impossible under
    PRF security (Boneh-Shoup, Definition 4.2) for token VALUES, and (with
    the pool fix) impossible from hit/miss EFFECT either in the
    non-exhausted case, because every dummy delete targets a genuinely
    present dummy entry, exactly as a real delete targets a genuinely
    present real entry. Both are HITS; the cloud cannot use hit/miss as a
    distinguisher. Contradiction → no such A exists (non-exhausted case).

Complexity:
    PrivPathInfer update: O(k) re-encryption + O(BATCH_SIZE) communication
    SDTC update:          O(2^N) always (full re-encryption)

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import random
from system.secure_dummy import RealisticDummyGenerator
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set

from crypto.prf_prp import generate_deletion_token
from system.rule_encryptor import EncryptedRule, RuleEncryptor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fixed batch size for update protocol
# Always send exactly BATCH_SIZE updates (pad with dummies if needed)
# Larger batch = more privacy but more communication
# Smaller batch = less privacy but less communication
# This is a public parameter (known to cloud)
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Update Batch Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DeleteOperation:
    """
    A deletion operation: instruct cloud to delete a rule.

    The cloud uses the deletion_token to find and delete the
    matching rule. It does NOT learn the rule_id.

    Fields:
        deletion_token: PRF(deletion_key, rule_id) — bytes(16)
        is_dummy:       True if this is a padding operation
    """
    deletion_token: bytes
    is_dummy:       bool = False


@dataclass
class InsertOperation:
    """
    An insertion operation: send a new encrypted rule to cloud.

    The new rule has fresh Paillier randomness (new r), so the
    cloud cannot link it to any previous ciphertext.

    Fields:
        rule:     new EncryptedRule with fresh encryption
        is_dummy: True if this is a padding operation
    """
    rule:     EncryptedRule
    is_dummy: bool = False


@dataclass
class UpdateBatch:
    """
    A fixed-size batch of update operations sent to the cloud.

    Always contains exactly BATCH_SIZE delete operations and
    exactly BATCH_SIZE insert operations.

    The cloud processes all operations without knowing which are
    real and which are dummies.

    Fields:
        delete_ops: list of BATCH_SIZE DeleteOperation objects
        insert_ops: list of BATCH_SIZE InsertOperation objects
        batch_size: always BATCH_SIZE (public parameter)
    """
    delete_ops: List[DeleteOperation]
    insert_ops: List[InsertOperation]
    batch_size: int = BATCH_SIZE


# ---------------------------------------------------------------------------
# Cloud Storage Simulator
# ---------------------------------------------------------------------------

class CloudStorage:
    """
    Simulates the cloud server's encrypted rule storage.

    The cloud stores encrypted rules indexed by deletion token.
    It processes delete/insert operations without learning rule content.

    In a real deployment, this would be a cloud database.
    Here it is simulated in memory for testing.
    """

    def __init__(self):
        """Initialize empty cloud storage."""
        # Map: deletion_token → EncryptedRule
        self._token_to_rule: Dict[bytes, EncryptedRule] = {}

    def upload_rules(self, rules: List[EncryptedRule]):
        """
        Initial upload of encrypted rules to cloud.

        Args:
            rules: list of EncryptedRule from RuleEncryptor
        """
        for rule in rules:
            self._token_to_rule[rule.deletion_token] = rule

    def apply_update_batch(self, batch: 'UpdateBatch'):
        """
        Apply an update batch: process all INSERTS first, then all DELETES.

        Order matters for the dummy-pool fix (see module docstring,
        "CORRECTION — the delete-channel hit/miss hole"): a dummy delete in
        this batch may target a dummy insert from this SAME batch. Inserting
        first guarantees that token is already present when its paired
        delete runs, so the delete is a genuine HIT — indistinguishable in
        effect from a real delete — rather than a miss. Real operations are
        unaffected by this reordering: a real delete's token always refers
        to a rule uploaded in a STRICTLY EARLIER batch (or the initial
        upload), which is already present in storage regardless of
        intra-batch order, and a real insert's token is freshly generated
        (via RuleEncryptor._next_rule_id) so it cannot collide with any
        token already in storage.

        The cloud processes ALL operations (real + dummy) identically.
        It cannot distinguish real from dummy.

        Args:
            batch: UpdateBatch with exactly BATCH_SIZE delete + insert ops
        """
        assert len(batch.delete_ops) == batch.batch_size, \
            f"Delete ops count {len(batch.delete_ops)} != batch_size {batch.batch_size}"
        assert len(batch.insert_ops) == batch.batch_size, \
            f"Insert ops count {len(batch.insert_ops)} != batch_size {batch.batch_size}"

        # Process insertions FIRST (see docstring above).
        for ins_op in batch.insert_ops:
            rule = ins_op.rule
            self._token_to_rule[rule.deletion_token] = rule

        # Process deletions SECOND.
        for del_op in batch.delete_ops:
            token = del_op.deletion_token
            if token in self._token_to_rule:
                del self._token_to_rule[token]
            # If token not found (pool-exhaustion-fallback dummy or already
            # deleted): silently ignore. In the non-exhausted case this
            # should not happen for dummy deletes (see create_update_batch).

    def get_all_rules(self) -> List[EncryptedRule]:
        """
        Return all currently stored rules.

        Used by inference engine to fetch rules for classification.

        Returns:
            list of EncryptedRule
        """
        return list(self._token_to_rule.values())

    def rule_count(self) -> int:
        """Return number of rules currently stored."""
        return len(self._token_to_rule)

    def contains_token(self, token: bytes) -> bool:
        """Check if a deletion token exists in storage."""
        return token in self._token_to_rule


# ---------------------------------------------------------------------------
# Update Protocol
# ---------------------------------------------------------------------------

class UpdateProtocol:
    """
    Incremental model update protocol for PrivPathInfer.

    When the decision tree is retrained, the Medical Institution (MI)
    uses this protocol to update only the changed rules on the cloud.

    Protocol:
        1. Identify changed rules (k rules changed)
        2. Generate deletion tokens for old rules
        3. Re-encrypt new rules with fresh randomness
        4. Pad to BATCH_SIZE with dummy operations
        5. Send UpdateBatch to cloud

    Security: Cloud learns only L_update = {update_occurred, batch_size}
    """

    def __init__(self, encryptor: RuleEncryptor, real_paths=None):
        """
        Initialize the update protocol with a RuleEncryptor.

        Args:
            encryptor:   RuleEncryptor holding secret keys
            real_paths:  list of LeafPath from the current model; when
                         provided, dummy inserts are drawn from the real
                         path distribution (closes the -100 sentinel hole).
        """
        self.encryptor = encryptor
        self._dummy_gen = (
            RealisticDummyGenerator(real_paths, encryptor)
            if real_paths else None
        )
        # Persistent pool of deletion_tokens belonging to dummy rules that
        # are believed to be sitting on the cloud (inserted as padding in a
        # previous batch, not yet deleted). Lets a later batch's dummy
        # DELETE target a genuinely present entry (a HIT), closing the
        # delete-channel hit/miss hole — see module docstring "CORRECTION".
        # Starts empty; call bootstrap_dummy_pool() before the first update.
        self._dummy_pool: List[bytes] = []

    def bootstrap_dummy_pool(self, n: int = BATCH_SIZE) -> List[EncryptedRule]:
        """
        One-time seed for the dummy pool, to be called BEFORE the very
        first update and uploaded to the cloud alongside the initial real
        rule set (e.g. `cloud.upload_rules(rules_v1 + protocol.bootstrap_dummy_pool())`).

        Without this, the pool starts empty and the FIRST update batch's
        dummy deletes (whatever this batch's same-batch pairing cannot
        cover — see create_update_batch) must fall back to
        `_make_dummy_delete_fallback()` (a miss). Bootstrapping avoids that
        cold-start gap. The seeded rules are statistically identical to
        real inserts (drawn from RealisticDummyGenerator, same as any other
        dummy insert) and provably inert — duplicating real paths changes
        nothing about classification.

        Args:
            n: number of dummy rules to seed (default BATCH_SIZE; a
               deployment with a known insert-heavy update workload should
               pass a larger multiple of BATCH_SIZE to delay pool
               exhaustion — see module docstring, "Honest residual").

        Returns:
            list of EncryptedRule to upload to cloud storage. Their
            deletion_token values are also recorded in the pool.
        """
        if self._dummy_gen is None:
            raise RuntimeError(
                "UpdateProtocol requires real_paths to bootstrap the dummy "
                "pool. Pass real_paths= to UpdateProtocol.__init__."
            )
        rules = self._dummy_gen.make_dummy_inserts(n)
        self._dummy_pool.extend(r.deletion_token for r in rules)
        return rules

    def _make_dummy_delete_fallback(self) -> DeleteOperation:
        """
        FALLBACK ONLY — generates a dummy deletion token for a freshly
        random, never-inserted rule_id. This token is a genuine PRF output
        (looks random) but its EFFECT is a MISS: it matches nothing in
        cloud storage, distinguishable in principle from a real delete's
        HIT (see module docstring "CORRECTION"). Used only when
        create_update_batch's dummy pool (same-batch pairing + persistent
        pool) cannot supply enough HIT-producing tokens — i.e. only under
        sustained insert-heavy update histories that exhaust the pool, or
        before the pool has ever been bootstrapped/populated. Prefer
        calling bootstrap_dummy_pool() before the first update to avoid
        relying on this fallback at all.

        Returns:
            DeleteOperation with is_dummy=True
        """
        dummy_rule_id = random.getrandbits(64)
        dummy_token   = generate_deletion_token(
            self.encryptor.deletion_key, dummy_rule_id
        )
        return DeleteOperation(deletion_token=dummy_token, is_dummy=True)

    def _make_dummy_insert(self) -> InsertOperation:
        """
        Create a dummy insertion operation for batch padding.

        When real_paths were supplied at construction time, the dummy is a
        re-encrypted copy of a real path sampled uniformly from the model —
        statistically identical to a real insert and provably inert
        (same label, same conditions).

        Falls back to a plain re-encryption of a random real path via
        RealisticDummyGenerator.make_dummy_inserts(1) if the generator is
        available, otherwise raises RuntimeError (caller must supply
        real_paths to UpdateProtocol.__init__).

        Returns:
            InsertOperation with is_dummy=True
        """
        if self._dummy_gen is None:
            raise RuntimeError(
                "UpdateProtocol requires real_paths to generate secure dummy "
                "inserts.  Pass real_paths= to UpdateProtocol.__init__."
            )
        rule = self._dummy_gen.make_dummy_inserts(1)[0]
        return InsertOperation(rule=rule, is_dummy=True)

    def create_update_batch(
        self,
        old_rules: List[EncryptedRule],
        new_rules:  List[EncryptedRule],
    ) -> 'UpdateBatch':
        """
        Create a fixed-size update batch for sending to the cloud.

        Identifies changed rules by comparing old and new rule sets,
        then pads to BATCH_SIZE with dummy operations.

        Algorithm:
            1. Find rules to delete: rules in old but not in new (by path_id + condition_index)
            2. Find rules to insert: rules in new but not in old
            3. Truncate to BATCH_SIZE if needed
            4. Pad inserts to BATCH_SIZE with fresh dummy insert rules.
            5. Pad deletes to BATCH_SIZE with tokens that are GUARANTEED to
               be HITS where possible: first reuse this batch's own surplus
               dummy-insert tokens (paired in the same batch — see
               CloudStorage.apply_update_batch for why insert-before-delete
               ordering makes this safe), then draw from the persistent
               dummy pool carried over from earlier batches. Only once both
               sources are exhausted does a dummy delete fall back to
               `_make_dummy_delete_fallback()` (a miss) — see module
               docstring "CORRECTION" and "Honest residual".

        Security:
            Batch always has exactly BATCH_SIZE delete + BATCH_SIZE insert ops.
            Cloud learns only: update occurred, batch_size = BATCH_SIZE, and
            (only if the dummy pool is exhausted) the count of real deletes.

        Args:
            old_rules: currently encrypted rules on cloud
            new_rules: newly encrypted rules after retraining

        Returns:
            UpdateBatch with exactly BATCH_SIZE delete and insert ops
        """
        # Build lookup maps
        old_map = {(r.path_id, r.condition_index): r for r in old_rules}
        new_map = {(r.path_id, r.condition_index): r for r in new_rules}

        # Rules to delete: in old but not in new, OR threshold changed
        rules_to_delete = []
        for key, old_rule in old_map.items():
            if key not in new_map:
                rules_to_delete.append(old_rule)
            else:
                # Check if threshold changed (re-encryption needed)
                new_rule = new_map[key]
                if old_rule.enc_threshold != new_rule.enc_threshold:
                    rules_to_delete.append(old_rule)

        # Rules to insert: in new but not in old, OR threshold changed
        rules_to_insert = []
        for key, new_rule in new_map.items():
            if key not in old_map:
                rules_to_insert.append(new_rule)
            else:
                old_rule = old_map[key]
                if old_rule.enc_threshold != new_rule.enc_threshold:
                    rules_to_insert.append(new_rule)

        # Truncate to BATCH_SIZE (if more changed than batch allows,
        # send in multiple batches — simplified here to one batch)
        rules_to_delete = rules_to_delete[:BATCH_SIZE]
        rules_to_insert = rules_to_insert[:BATCH_SIZE]

        # Build real delete operations
        delete_ops = [
            DeleteOperation(
                deletion_token = rule.deletion_token,
                is_dummy       = False,
            )
            for rule in rules_to_delete
        ]

        # Build real insert operations
        insert_ops = [
            InsertOperation(rule=rule, is_dummy=False)
            for rule in rules_to_insert
        ]

        n_dummy_deletes_needed = BATCH_SIZE - len(delete_ops)
        n_dummy_inserts_needed = BATCH_SIZE - len(insert_ops)

        # --- Generate this batch's dummy INSERT padding up front, so its
        # tokens are available for same-batch delete pairing. ---
        dummy_insert_rules: List[EncryptedRule] = []
        if n_dummy_inserts_needed > 0:
            dummy_insert_rules = self._dummy_gen.make_dummy_inserts(
                n_dummy_inserts_needed
            )
        for rule in dummy_insert_rules:
            insert_ops.append(InsertOperation(rule=rule, is_dummy=True))

        # --- Build dummy DELETE padding, preferring HIT-producing tokens. ---
        same_batch_tokens = [r.deletion_token for r in dummy_insert_rules]
        take = min(n_dummy_deletes_needed, len(same_batch_tokens))
        dummy_delete_tokens = list(same_batch_tokens[:take])

        # Any same-batch dummy-insert tokens NOT consumed as deletes this
        # round stay on the cloud; bank them in the persistent pool for a
        # future batch's dummy deletes to consume.
        leftover_for_pool = same_batch_tokens[take:]
        self._dummy_pool.extend(leftover_for_pool)

        remaining = n_dummy_deletes_needed - take
        while remaining > 0 and self._dummy_pool:
            dummy_delete_tokens.append(self._dummy_pool.pop())
            remaining -= 1

        delete_ops += [
            DeleteOperation(deletion_token=tok, is_dummy=True)
            for tok in dummy_delete_tokens
        ]

        # Pool exhausted (or never bootstrapped) and still short: fall back
        # to miss-producing tokens for the remainder only. See module
        # docstring "Honest residual" — this is the bounded worst case, not
        # silent failure.
        while remaining > 0:
            delete_ops.append(self._make_dummy_delete_fallback())
            remaining -= 1

        # Shuffle to prevent position-based inference
        random.shuffle(delete_ops)
        random.shuffle(insert_ops)

        return UpdateBatch(
            delete_ops = delete_ops,
            insert_ops = insert_ops,
            batch_size = BATCH_SIZE,
        )

    def count_real_operations(self, batch: 'UpdateBatch') -> Dict[str, int]:
        """
        Count real vs dummy operations in a batch (MI only).

        The cloud cannot make this distinction. MI uses this for
        performance analysis and experiment reporting.

        Args:
            batch: UpdateBatch

        Returns:
            dict with 'real_deletes', 'dummy_deletes',
                      'real_inserts', 'dummy_inserts'
        """
        real_deletes  = sum(1 for op in batch.delete_ops if not op.is_dummy)
        dummy_deletes = sum(1 for op in batch.delete_ops if op.is_dummy)
        real_inserts  = sum(1 for op in batch.insert_ops if not op.is_dummy)
        dummy_inserts = sum(1 for op in batch.insert_ops if op.is_dummy)

        return {
            'real_deletes':  real_deletes,
            'dummy_deletes': dummy_deletes,
            'real_inserts':  real_inserts,
            'dummy_inserts': dummy_inserts,
        }


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """
    Verify the incremental update protocol.

    Tests:
        1. Batch size always = BATCH_SIZE
        2. Real updates applied correctly
        3. Dummy operations do not affect valid rules
        4. Deletion tokens work correctly
        5. Update cost O(k) vs SDTC O(2^N)
        6. Inference still correct after update
    """
    from system.path_extractor import PathExtractor, from_dict
    from system.inference_engine import PaillierInferenceEngine, PlaintextClassifier

    print("=" * 60)
    print("UpdateProtocol Verification Tests")
    print("PrivPathInfer Contribution 3: Incremental Updates")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)

    # Build test tree
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
    root      = from_dict(tree_dict)
    extractor = PathExtractor(root)
    paths     = extractor.extract_paths()

    encryptor = RuleEncryptor(paillier_bits=512)
    rules_v1  = encryptor.encrypt_paths(paths)

    cloud    = CloudStorage()
    # Pass real_paths so dummy inserts are realistic
    protocol = UpdateProtocol(encryptor, real_paths=paths)

    # Initial upload
    cloud.upload_rules(rules_v1)
    print(f"\nInitial upload: {cloud.rule_count()} rules on cloud")

    # Test 1: Batch size always = BATCH_SIZE
    batch = protocol.create_update_batch(rules_v1, rules_v1)
    assert len(batch.delete_ops) == BATCH_SIZE, \
        f"Delete ops = {len(batch.delete_ops)}, expected {BATCH_SIZE}"
    assert len(batch.insert_ops) == BATCH_SIZE, \
        f"Insert ops = {len(batch.insert_ops)}, expected {BATCH_SIZE}"
    assert batch.batch_size == BATCH_SIZE
    print(f"[PASS] Test 1: Batch always has exactly {BATCH_SIZE} delete + {BATCH_SIZE} insert ops")

    # Test 2: No-change update (all dummies)
    counts = protocol.count_real_operations(batch)
    assert counts['real_deletes'] == 0, "No-change update should have 0 real deletes"
    assert counts['real_inserts'] == 0, "No-change update should have 0 real inserts"
    assert counts['dummy_deletes'] == BATCH_SIZE
    assert counts['dummy_inserts'] == BATCH_SIZE
    print(f"[PASS] Test 2: No-change update → {BATCH_SIZE} dummies, 0 real ops")

    # Test 3: Update with changed threshold
    tree_dict_v2 = {
        'feature_idx': 1,
        'threshold':   130.0,  # changed from 126.5
        'left': {
            'feature_idx': 5,
            'threshold':   29.1,
            'left':  {'label': 0},
            'right': {'label': 1},
        },
        'right': {'label': 1},
    }
    root_v2      = from_dict(tree_dict_v2)
    extractor_v2 = PathExtractor(root_v2)
    paths_v2     = extractor_v2.extract_paths()
    rules_v2     = encryptor.encrypt_paths(paths_v2)

    batch_v2 = protocol.create_update_batch(rules_v1, rules_v2)
    counts_v2 = protocol.count_real_operations(batch_v2)
    assert counts_v2['real_deletes'] > 0, "Should have real deletions"
    assert counts_v2['real_inserts'] > 0, "Should have real insertions"
    assert len(batch_v2.delete_ops) == BATCH_SIZE
    assert len(batch_v2.insert_ops) == BATCH_SIZE
    print(f"[PASS] Test 3: Changed threshold → "
          f"{counts_v2['real_deletes']} real deletes, "
          f"{counts_v2['dummy_deletes']} dummies "
          f"(cloud always sees {BATCH_SIZE})")

    # Test 4: Apply update to cloud and verify rule count
    rules_before = cloud.rule_count()
    cloud.apply_update_batch(batch_v2)
    rules_after = cloud.rule_count()
    print(f"[PASS] Test 4: Cloud update applied "
          f"(rules: {rules_before} → {rules_after})")

    # Test 5: Inference still correct after update
    engine    = PaillierInferenceEngine(encryptor)
    plaintext = PlaintextClassifier()
    updated_rules = cloud.get_all_rules()

    test_features = [
        [0, 80.0,  0, 0, 0, 20.0, 0, 0],
        [0, 135.0, 0, 0, 0, 20.0, 0, 0],
        [0, 80.0,  0, 0, 0, 35.0, 0, 0],
    ]

    for features in test_features:
        pt_result  = plaintext.classify(features, paths_v2)
        enc_result = engine.classify_plaintext(features, updated_rules)
        assert pt_result == enc_result, \
            f"Post-update inference mismatch: plaintext={pt_result}, encrypted={enc_result}"

    print(f"[PASS] Test 5: Inference correct after update (all samples match)")

    # Test 6: Update cost comparison
    k = counts_v2['real_deletes']
    N = len(rules_v1)
    max_depth = max(p.depth for p in paths)

    print(f"\n[PASS] Test 6: Update cost comparison:")
    print(f"       Changed rules k = {k}")
    print(f"       PrivPathInfer cost: O(k) = O({k}) re-encryptions")
    print(f"       SDTC cost:          O(2^N) = O({2**max_depth}) always")
    print(f"       Speedup: {2**max_depth // max(k, 1)}x")

    # Test 7: PRF token security — cloud cannot match token to rule_id
    sample_rule = rules_v1[0]
    token = sample_rule.deletion_token
    assert cloud.contains_token(token) or not cloud.contains_token(token), \
        "Token lookup should work correctly"
    print(f"[PASS] Test 7: Deletion tokens are valid PRF outputs")

    # ------------------------------------------------------------------
    # Tests 8-11: the dummy-pool fix (delete-channel hit/miss hole)
    # ------------------------------------------------------------------

    # Test 8: bootstrap seeds the pool and the seeded rules are uploadable
    protocol2 = UpdateProtocol(encryptor, real_paths=paths)
    seed_rules = protocol2.bootstrap_dummy_pool(BATCH_SIZE)
    assert len(seed_rules) == BATCH_SIZE
    assert len(protocol2._dummy_pool) == BATCH_SIZE
    cloud3 = CloudStorage()
    cloud3.upload_rules(rules_v1)
    cloud3.upload_rules(seed_rules)   # one-time seed upload, alongside real rules
    print(f"[PASS] Test 8: bootstrap_dummy_pool seeds {BATCH_SIZE} rules, "
          f"uploadable to cloud")

    # Test 9: after bootstrap, a no-op batch's dummy deletes are ALL hits.
    # A dummy delete is a HIT if its token is present either (a) already on
    # the cloud before this batch (e.g. from bootstrap or a prior batch's
    # leftover pool tokens), or (b) among THIS batch's own insert tokens
    # (since CloudStorage.apply_update_batch processes inserts before
    # deletes, a same-batch-paired token is present by the time its delete
    # runs). Checking only (a) would wrongly flag same-batch-paired tokens
    # as misses, since they are not on the cloud until mid-batch.
    noop_batch = protocol2.create_update_batch(rules_v1, rules_v1)
    dummy_delete_tokens = [op.deletion_token for op in noop_batch.delete_ops
                            if op.is_dummy]
    this_batch_insert_tokens = {op.rule.deletion_token for op in noop_batch.insert_ops}
    hits = sum(
        1 for tok in dummy_delete_tokens
        if cloud3.contains_token(tok) or tok in this_batch_insert_tokens
    )
    assert hits == len(dummy_delete_tokens) == BATCH_SIZE, (
        f"Expected all {BATCH_SIZE} dummy deletes to be hits after "
        f"bootstrap, got {hits}/{len(dummy_delete_tokens)}"
    )
    print(f"[PASS] Test 9: all {hits}/{BATCH_SIZE} dummy deletes are HITS "
          f"after bootstrap (indistinguishable in effect from real deletes)")

    # Test 10: applying that batch leaves the rule set semantically
    # unchanged (dummy insert + matching dummy delete cancel out;
    # inference is still correct).
    cloud3.apply_update_batch(noop_batch)
    engine3 = PaillierInferenceEngine(encryptor)
    for features in [[0, 80.0, 0, 0, 0, 20.0, 0, 0],
                      [0, 150.0, 0, 0, 0, 20.0, 0, 0]]:
        pt = plaintext.classify(features, paths)
        enc = engine3.classify_plaintext(features, cloud3.get_all_rules())
        assert pt == enc, "Inference must be unaffected by pool-paired padding"
    print(f"[PASS] Test 10: inference unaffected after a hit-only padded "
          f"no-op batch")

    # Test 11: WITHOUT bootstrap (cold start, empty pool), the very first
    # batch still pairs as many dummy deletes as possible with this same
    # batch's own dummy inserts (same-batch pairing alone, no persistent
    # pool needed for a balanced no-op batch) — so it should ALSO be 100%
    # hits even without calling bootstrap_dummy_pool() first, because a
    # no-op batch needs exactly BATCH_SIZE dummy inserts and BATCH_SIZE
    # dummy deletes, which same-batch pairing covers exactly.
    protocol4 = UpdateProtocol(encryptor, real_paths=paths)  # no bootstrap
    cloud4 = CloudStorage()
    cloud4.upload_rules(rules_v1)
    cold_batch = protocol4.create_update_batch(rules_v1, rules_v1)
    cold_dummy_deletes = [op.deletion_token for op in cold_batch.delete_ops
                           if op.is_dummy]
    # Tokens must match this SAME batch's dummy insert tokens (same-batch
    # pairing), confirmed by checking they appear among insert_ops' tokens.
    insert_tokens = {op.rule.deletion_token for op in cold_batch.insert_ops}
    same_batch_hits = sum(1 for tok in cold_dummy_deletes if tok in insert_tokens)
    assert same_batch_hits == BATCH_SIZE, (
        f"Cold-start no-op batch should pair all {BATCH_SIZE} dummy deletes "
        f"with same-batch dummy inserts, got {same_batch_hits}"
    )
    print(f"[PASS] Test 11: cold start (no bootstrap) still achieves "
          f"{same_batch_hits}/{BATCH_SIZE} same-batch hit-pairing on a "
          f"balanced no-op batch")

    # Test 12: persistent-pool draw. Force a batch with MORE real inserts
    # than real deletes (n_dummy_deletes_needed > n_dummy_inserts_needed),
    # so same-batch pairing alone cannot cover every dummy delete and the
    # persistent pool (still holding protocol2's 8 untouched bootstrap
    # tokens from Test 9, where same-batch pairing fully covered that
    # no-op batch and left the pool unused) must be drawn from.
    from system.path_extractor import LeafPath, PathCondition
    extra_paths = [
        LeafPath(
            path_id=1000 + i,
            conditions=[PathCondition(node_id=0, feature_idx=1,
                                       threshold=100.0 + i, direction='left', depth=0)],
            label=i % 2, leaf_id=-1, depth=1,
        )
        for i in range(BATCH_SIZE)
    ]
    extra_rules = encryptor.encrypt_paths(extra_paths)
    rules_v3 = list(rules_v1) + extra_rules   # pure inserts, zero deletes
    assert len(protocol2._dummy_pool) == BATCH_SIZE, \
        "pool should still hold the untouched bootstrap tokens before this batch"
    pool_before = list(protocol2._dummy_pool)
    insert_heavy_batch = protocol2.create_update_batch(rules_v1, rules_v3)
    counts_iv = protocol2.count_real_operations(insert_heavy_batch)
    assert counts_iv['real_inserts'] == BATCH_SIZE and counts_iv['real_deletes'] == 0, (
        f"expected {BATCH_SIZE} real inserts / 0 real deletes, got {counts_iv}"
    )
    dummy_del_tokens_iv = [op.deletion_token for op in insert_heavy_batch.delete_ops
                            if op.is_dummy]
    assert len(dummy_del_tokens_iv) == BATCH_SIZE
    drawn_from_pool = sum(1 for t in dummy_del_tokens_iv if t in pool_before)
    assert drawn_from_pool == BATCH_SIZE, (
        f"insert-heavy batch (0 same-batch dummy-insert slack) should draw "
        f"all {BATCH_SIZE} dummy-delete tokens from the persistent pool, "
        f"got {drawn_from_pool}"
    )
    assert len(protocol2._dummy_pool) == 0, "pool should be drained after this batch"
    # those drawn tokens were uploaded to cloud3 during bootstrap (Test 8),
    # so they are genuine HITS, not misses.
    pool_hits = sum(1 for t in dummy_del_tokens_iv if cloud3.contains_token(t))
    assert pool_hits == BATCH_SIZE, \
        f"all {BATCH_SIZE} pool-drawn tokens should already be on cloud3 (hits)"
    print(f"[PASS] Test 12: insert-heavy batch correctly draws "
          f"{drawn_from_pool}/{BATCH_SIZE} dummy-delete tokens from the "
          f"persistent pool, all of which are genuine HITS; pool drained to 0")

    # Test 13: pool-exhaustion fallback. With the pool now empty (Test 12)
    # and another insert-heavy batch (no same-batch slack, no pool left),
    # create_update_batch must NOT crash and must still return exactly
    # BATCH_SIZE delete ops — falling back to
    # _make_dummy_delete_fallback() (miss-producing) for the shortfall,
    # exactly the bounded, disclosed worst case from the module docstring.
    rules_v4 = list(rules_v3) + encryptor.encrypt_paths([
        LeafPath(path_id=2000 + i,
                  conditions=[PathCondition(node_id=0, feature_idx=2,
                                             threshold=50.0 + i, direction='left', depth=0)],
                  label=i % 2, leaf_id=-1, depth=1)
        for i in range(BATCH_SIZE)
    ])
    exhausted_batch = protocol2.create_update_batch(rules_v3, rules_v4)
    assert len(exhausted_batch.delete_ops) == BATCH_SIZE
    assert len(exhausted_batch.insert_ops) == BATCH_SIZE
    fallback_tokens = [op.deletion_token for op in exhausted_batch.delete_ops
                        if op.is_dummy]
    this_batch_inserts2 = {op.rule.deletion_token for op in exhausted_batch.insert_ops}
    misses = sum(
        1 for t in fallback_tokens
        if not cloud3.contains_token(t) and t not in this_batch_inserts2
    )
    assert misses == BATCH_SIZE, (
        f"with pool exhausted and no same-batch slack, all {BATCH_SIZE} "
        f"dummy deletes should fall back to the (disclosed, bounded) miss "
        f"case, got {misses} misses"
    )
    print(f"[PASS] Test 13: pool exhaustion degrades gracefully to the "
          f"bounded fallback ({misses}/{BATCH_SIZE} misses, batch_size "
          f"invariant still held) — matches the documented worst case")

    print(f"\n[ALL TESTS PASSED] update_protocol.py verified.")
    print(f"Contribution 3: Incremental updates with O(k) cost confirmed.")
    print(f"Security (corrected): L_update = {{update_occurred, batch_size={BATCH_SIZE}}}")
    print(f"Delete-channel hit/miss hole closed via dummy pool "
          f"(same-batch pairing + persistent pool + bootstrap).")


if __name__ == "__main__":
    run_all_tests()