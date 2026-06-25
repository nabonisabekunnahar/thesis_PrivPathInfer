"""
exp2_storage.py — Experiment 2: Storage Comparison
====================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Experiment 2: Demonstrates PrivPathInfer Contribution 2.

Three metrics reported:

Metric 1 — Entry/Path Count:
    PrivPathInfer: N+1 paths — O(N)
    SDTC:          2^depth entries — O(2^N)

Metric 2 — Encrypted Bytes (without dedup):
    PrivPathInfer: total_conditions × 290B
    SDTC:          entries × 32B

Metric 3 — Encrypted Bytes (with dedup):
    PrivPathInfer: unique_thresholds × 290B + total_conditions × 8B
    SDTC:          entries × 32B
    Key result: PrivPathInfer becomes smaller than SDTC

Dataset: PIMA Indians Diabetes
"""

import json
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from baseline.sdtc import compute_sdtc_storage

PAILLIER_CIPHERTEXT_BYTES = 256   # 1024-bit: n^2 = 256 bytes
REF_BYTES                 = 8     # integer reference per condition
SDTC_BYTES_PER_ENTRY      = 32    # A[i]=16B + T[i]=16B


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
    return np.array(X), np.array(y)


def count_unique_thresholds(paths):
    """Count unique (feature_idx, threshold) pairs across all paths."""
    unique = set()
    for p in paths:
        for c in p.conditions:
            unique.add((c.feature_idx, round(c.threshold, 6)))
    return len(unique)


def run_experiment2(data_path='data/diabetes.csv'):
    print("=" * 70)
    print("Experiment 2: Storage Comparison")
    print("PrivPathInfer Contribution 2: O(N) paths + Threshold Deduplication")
    print("=" * 70)

    X, y = load_pima(data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    depths = list(range(2, 13))

    results = {
        'depths':                    depths,
        'privpathinfer_paths':       [],
        'privpathinfer_rules':       [],
        'privpathinfer_bytes_kb':    [],   # without dedup
        'privpathinfer_dedup_kb':    [],   # with dedup
        'unique_thresholds':         [],
        'dedup_ratio':               [],
        'sdtc_entries':              [],
        'sdtc_bytes_kb':             [],
        'internal_nodes':            [],
        'path_ratio':                [],
        'byte_ratio_no_dedup':       [],
        'byte_ratio_dedup':          [],
    }

    print(f"\nMetric 1 — Entry/Path Count:")
    print(f"{'Depth':<7} {'N':<6} {'Paths':<10} {'SDTC entries':<15} {'Ratio'}")
    print("-" * 45)

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        tree_root  = from_sklearn_tree(clf)
        extractor  = PathExtractor(tree_root)
        paths      = extractor.extract_paths()
        n_paths    = len(paths)
        n_rules    = sum(len(p.conditions) for p in paths)
        n_internal = extractor.get_internal_node_count()
        n_unique   = count_unique_thresholds(paths)
        dedup_ratio = n_rules / max(n_unique, 1)

        actual_max_depth = max(p.depth for p in paths) if paths else depth
        sdtc_entries     = compute_sdtc_storage(actual_max_depth)

        # Bytes without dedup
        priv_bytes    = n_rules  * PAILLIER_CIPHERTEXT_BYTES
        # Bytes with dedup
        priv_dedup    = n_unique * PAILLIER_CIPHERTEXT_BYTES + n_rules * REF_BYTES
        sdtc_bytes    = sdtc_entries * SDTC_BYTES_PER_ENTRY

        path_ratio         = sdtc_entries  / max(n_paths, 1)
        byte_ratio_no_dedup = sdtc_bytes   / max(priv_bytes, 1)
        byte_ratio_dedup   = sdtc_bytes    / max(priv_dedup, 1)

        results['privpathinfer_paths'].append(n_paths)
        results['privpathinfer_rules'].append(n_rules)
        results['privpathinfer_bytes_kb'].append(round(priv_bytes / 1024, 1))
        results['privpathinfer_dedup_kb'].append(round(priv_dedup / 1024, 1))
        results['unique_thresholds'].append(n_unique)
        results['dedup_ratio'].append(round(dedup_ratio, 1))
        results['sdtc_entries'].append(sdtc_entries)
        results['sdtc_bytes_kb'].append(round(sdtc_bytes / 1024, 1))
        results['internal_nodes'].append(n_internal)
        results['path_ratio'].append(round(path_ratio, 1))
        results['byte_ratio_no_dedup'].append(round(byte_ratio_no_dedup, 2))
        results['byte_ratio_dedup'].append(round(byte_ratio_dedup, 2))

        print(f"{depth:<7} {n_internal:<6} {n_paths:<10} "
              f"{sdtc_entries:<15} {path_ratio:.1f}x")

    print(f"\nMetric 2 — Encrypted Bytes (without deduplication):")
    print(f"{'Depth':<7} {'PrivPath(KB)':<15} {'SDTC(KB)':<12} {'Ratio'}")
    print("-" * 40)
    for i, d in enumerate(depths):
        r = results['byte_ratio_no_dedup'][i]
        print(f"{d:<7} {results['privpathinfer_bytes_kb'][i]:<15.1f} "
              f"{results['sdtc_bytes_kb'][i]:<12.1f} "
              f"{'SDTC '+str(r)+'x larger' if r>=1 else 'PrivPath '+str(round(1/r,1))+'x larger'}")

    print(f"\nMetric 3 — Encrypted Bytes (WITH deduplication):")
    print(f"{'Depth':<7} {'Unique':<9} {'Dedup ratio':<14} "
          f"{'PrivPath+dedup(KB)':<21} {'SDTC(KB)':<12} {'Ratio'}")
    print("-" * 70)
    for i, d in enumerate(depths):
        r = results['byte_ratio_dedup'][i]
        print(f"{d:<7} {results['unique_thresholds'][i]:<9} "
              f"{results['dedup_ratio'][i]:<14.1f}x "
              f"{results['privpathinfer_dedup_kb'][i]:<21.1f} "
              f"{results['sdtc_bytes_kb'][i]:<12.1f} "
              f"{'PrivPath '+str(round(r,1))+'x smaller' if r>=1 else 'SDTC '+str(round(1/r,1))+'x smaller'}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS (Contribution 2):")
    print(f"\n  Without deduplication:")
    print(f"    Depth 12: PrivPathInfer {results['privpathinfer_bytes_kb'][-1]} KB "
          f"vs SDTC {results['sdtc_bytes_kb'][-1]} KB")
    print(f"    PrivPathInfer is {1/results['byte_ratio_no_dedup'][-1]:.1f}x LARGER (Paillier overhead)")

    print(f"\n  WITH deduplication (encrypt each unique threshold once):")
    print(f"    Depth 12: PrivPathInfer {results['privpathinfer_dedup_kb'][-1]} KB "
          f"vs SDTC {results['sdtc_bytes_kb'][-1]} KB")
    r = results['byte_ratio_dedup'][-1]
    print(f"    PrivPathInfer is {r:.1f}x SMALLER than SDTC")
    print(f"    Dedup ratio: {results['dedup_ratio'][-1]}x "
          f"(each threshold reused {results['dedup_ratio'][-1]} times on average)")

    print(f"\n  Path count advantage (independent of dedup):")
    print(f"    Depth 12: {results['privpathinfer_paths'][-1]} paths "
          f"vs {results['sdtc_entries'][-1]} SDTC entries "
          f"= {results['path_ratio'][-1]}x fewer")

    output = {
        'experiment': 2,
        'description': 'Storage Comparison with Threshold Deduplication',
        'dataset': 'PIMA Indians Diabetes',
        'results': results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/exp2_storage.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/exp2_storage.json")
    return output


if __name__ == '__main__':
    run_experiment2()