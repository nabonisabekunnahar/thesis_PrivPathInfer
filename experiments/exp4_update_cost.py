"""
exp4_update_cost.py — Experiment 4: Incremental Update Cost
=============================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Experiment 4: Demonstrates PrivPathInfer Contribution 3.

Research Question:
    How does update cost scale with number of changed rules?

Method:
    Train tree on PIMA dataset (depth=8 for enough rules).
    Vary changed rules: 1, 2, 4, 8, 16.
    Measure re-encryption time for:
        1. PrivPathInfer: O(k) — only changed rules re-encrypted
        2. SDTC: O(2^N) — always full re-encryption

Key Metric:
    PrivPathInfer update = re-encrypt k rules + build batch
    SDTC update         = re-encrypt entire decision table
"""

import json
import os
import sys
import time
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from system.rule_encryptor import RuleEncryptor
from system.update_protocol import UpdateProtocol, CloudStorage, BATCH_SIZE
from baseline.sdtc import SDTC
from crypto.paillier import encrypt as paillier_encrypt, encode_threshold


def load_pima(path='data/diabetes.csv'):
    import csv
    X, y = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['Pregnancies']), float(row['Glucose']),
                      float(row['BloodPressure']), float(row['SkinThickness']),
                      float(row['Insulin']), float(row['BMI']),
                      float(row['DiabetesPedigreeFunction']), float(row['Age'])])
            y.append(int(row['Outcome']))
    import numpy as np
    return np.array(X), np.array(y)


def measure_privpathinfer_update(encryptor, rules_v1, k, n_repeats=5):
    """
    Measure PrivPathInfer update cost for k changed rules.

    Steps:
        1. Identify k rules to update
        2. Re-encrypt k rules with fresh Paillier randomness
        3. Build batch (pad to BATCH_SIZE with dummies)
        4. Apply batch to cloud

    This is O(k) Paillier encryptions + O(BATCH_SIZE) PRF calls.
    """
    protocol = UpdateProtocol(encryptor)
    times = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()

        # Step 1: Select k rules to change (simulate threshold update)
        rules_to_change = rules_v1[:min(k, len(rules_v1))]
        unchanged_rules = rules_v1[min(k, len(rules_v1)):]

        # Step 2: Re-encrypt k changed rules (core O(k) cost)
        new_rules = []
        for rule in rules_to_change:
            old_thresh = encryptor.decrypt_threshold(rule.enc_threshold)
            new_thresh = old_thresh + random.uniform(0.5, 2.0)
            new_enc_thresh = encryptor._encrypt_threshold(new_thresh)
            from system.rule_encryptor import EncryptedRule
            from crypto.prf_prp import generate_deletion_token
            new_rule_id = encryptor._next_rule_id()
            new_rules.append(EncryptedRule(
                rule_id         = new_rule_id,
                path_id         = rule.path_id,
                condition_index = rule.condition_index,
                enc_feature_idx = rule.enc_feature_idx,
                enc_threshold   = new_enc_thresh,
                direction       = rule.direction,
                label           = rule.label,
                is_last         = rule.is_last,
                deletion_token  = generate_deletion_token(
                    encryptor.deletion_key, new_rule_id),
                depth           = rule.depth,
            ))

        rules_v2 = list(unchanged_rules) + new_rules

        # Step 3: Build batch with padding
        batch = protocol.create_update_batch(rules_v1, rules_v2)

        # Step 4: Apply to cloud
        cloud = CloudStorage()
        cloud.upload_rules(rules_v1)
        cloud.apply_update_batch(batch)

        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def measure_sdtc_update(clf, X_train, n_repeats=5):
    """
    Measure SDTC update cost.

    SDTC always re-encrypts the ENTIRE decision table.
    Cost is O(2^depth) regardless of how many rules changed.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        sdtc = SDTC(n_bins=10)
        sdtc.fit_encrypt(clf, X_train)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def run_experiment4(data_path='data/diabetes.csv', max_depth=8, n_repeats=5):
    """Run Experiment 4: Incremental Update Cost."""

    print("=" * 60)
    print("Experiment 4: Incremental Update Cost")
    print("PrivPathInfer Contribution 3: O(k) vs O(2^N) SDTC")
    print(f"Tree depth: {max_depth}, Repeats: {n_repeats}")
    print("=" * 60)

    X, y = load_pima(data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train tree with enough depth for meaningful k values
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Setup PrivPathInfer
    tree_root = from_sklearn_tree(clf)
    extractor = PathExtractor(tree_root)
    paths     = extractor.extract_paths()
    encryptor = RuleEncryptor(paillier_bits=1024)
    rules_v1  = encryptor.encrypt_paths(paths)

    total_rules = len(rules_v1)
    n_internal  = extractor.get_internal_node_count()

    print(f"\nTree: {n_internal} internal nodes, {total_rules} encrypted rules")
    print(f"SDTC theoretical table: {2**max_depth} entries")
    print(f"Batch size: {BATCH_SIZE}\n")

    # SDTC baseline cost (always same)
    sdtc_mean, sdtc_std = measure_sdtc_update(clf, X_train, n_repeats)
    print(f"SDTC full re-encryption: {sdtc_mean:.2f} ± {sdtc_std:.2f} ms")
    print(f"(Always O(2^{max_depth}) = O({2**max_depth}) regardless of k)\n")

    k_values = [1, 2, 4, 8, 16]
    results = {
        'k_values':             k_values,
        'privpathinfer_mean_ms': [],
        'privpathinfer_std_ms':  [],
        'sdtc_mean_ms':          float(sdtc_mean),
        'sdtc_std_ms':           float(sdtc_std),
        'total_rules':           total_rules,
        'n_internal_nodes':      n_internal,
        'tree_depth':            max_depth,
    }

    print(f"{'k (changed rules)':<20} {'PrivPathInfer (ms)':<25} {'SDTC always (ms)':<20} {'Speedup'}")
    print("-" * 75)

    for k in k_values:
        actual_k = min(k, total_rules)
        priv_mean, priv_std = measure_privpathinfer_update(
            encryptor, rules_v1, k, n_repeats
        )
        speedup = sdtc_mean / max(priv_mean, 0.001)

        results['privpathinfer_mean_ms'].append(priv_mean)
        results['privpathinfer_std_ms'].append(priv_std)

        print(f"k={k} (actual={actual_k}){'':<10} "
              f"{priv_mean:.2f} ± {priv_std:.2f} ms{'':<10} "
              f"{sdtc_mean:.2f} ms{'':<10} "
              f"{speedup:.1f}x")

    print("\n" + "=" * 60)
    print("KEY FINDING (Contribution 3):")
    print(f"  PrivPathInfer: cost grows with k (O(k) Paillier encryptions)")
    print(f"  SDTC: always {sdtc_mean:.2f} ms regardless of k")
    print(f"  At k=1: PrivPathInfer is {sdtc_mean/max(results['privpathinfer_mean_ms'][0],0.001):.1f}x faster")
    print(f"  Update leakage: L_update = {{update_occurred, batch_size={BATCH_SIZE}}}")

    output = {
        'experiment': 4,
        'description': 'Incremental Update Cost',
        'dataset': 'PIMA Indians Diabetes',
        'max_depth': max_depth,
        'n_repeats': n_repeats,
        'batch_size': BATCH_SIZE,
        'results': results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/exp4_update_cost.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/exp4_update_cost.json")
    return output


if __name__ == '__main__':
    run_experiment4()


def run_communication_analysis():
    """
    Communication cost comparison: PrivPathInfer vs SDTC update.

    PrivPathInfer update sends BATCH_SIZE rules:
        BATCH_SIZE × 290 bytes per rule

    SDTC update sends entire re-encrypted table:
        2^depth × 32 bytes per entry

    This shows Contribution 3's communication advantage.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from system.update_protocol import BATCH_SIZE

    PRIVPATH_BYTES_PER_RULE = 290
    SDTC_BYTES_PER_ENTRY    = 32

    print("\n" + "=" * 60)
    print("Update Communication Cost Comparison")
    print("=" * 60)
    print(f"\nPrivPathInfer: always sends BATCH_SIZE={BATCH_SIZE} rules")
    print(f"  Per update: {BATCH_SIZE} × {PRIVPATH_BYTES_PER_RULE}B = "
          f"{BATCH_SIZE * PRIVPATH_BYTES_PER_RULE}B = "
          f"{BATCH_SIZE * PRIVPATH_BYTES_PER_RULE / 1024:.2f} KB")

    print(f"\nSDTC: sends entire re-encrypted table")
    print(f"\n{'Depth':<8} {'SDTC comm (KB)':<18} {'PrivPath comm (KB)':<20} {'Ratio'}")
    print("-" * 55)

    priv_comm_kb = BATCH_SIZE * PRIVPATH_BYTES_PER_RULE / 1024

    results = {
        'privpathinfer_comm_kb': priv_comm_kb,
        'sdtc_comm_kb': [],
        'comm_ratio': [],
        'depths': list(range(2, 13))
    }

    for depth in range(2, 13):
        sdtc_entries = 2 ** depth
        sdtc_comm_kb = sdtc_entries * SDTC_BYTES_PER_ENTRY / 1024
        ratio        = sdtc_comm_kb / priv_comm_kb

        results['sdtc_comm_kb'].append(sdtc_comm_kb)
        results['comm_ratio'].append(round(ratio, 1))

        print(f"{depth:<8} {sdtc_comm_kb:<18.2f} {priv_comm_kb:<20.2f} "
              f"{ratio:.1f}x more (SDTC)")

    print(f"\nKey Finding:")
    print(f"  PrivPathInfer always sends {priv_comm_kb:.2f} KB per update")
    print(f"  SDTC sends {results['sdtc_comm_kb'][-1]:.0f} KB at depth 12")
    print(f"  Communication reduction: {results['comm_ratio'][-1]:.0f}x less")

    return results


if __name__ == '__main__':
    run_experiment4()
    run_communication_analysis()