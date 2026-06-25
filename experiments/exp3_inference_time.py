"""
exp3_inference_time.py — Experiment 3: Per-Query Inference Time
================================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Experiment 3: Measures per-query inference time for all methods.

Method:
    Train tree on PIMA dataset.
    Measure mean ± std inference time over 100 queries for:
        1. PrivPathInfer Paillier mode
        2. PrivPathInfer ORE mode (theoretical)
        3. SDTC
        4. Plaintext DT (reference)
"""

import json
import os
import sys
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from system.rule_encryptor import RuleEncryptor
from system.inference_engine import PaillierInferenceEngine, PlaintextClassifier
from baseline.sdtc import SDTC
from baseline.sdtc_full import SDTCFull


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


def run_experiment3(data_path='data/diabetes.csv', n_queries=100, max_depth=5):
    """Run Experiment 3: Per-Query Inference Time."""

    print("=" * 60)
    print("Experiment 3: Per-Query Inference Time")
    print(f"Queries: {n_queries}, Tree depth: {max_depth}")
    print("=" * 60)

    X, y = load_pima(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Setup PrivPathInfer
    tree_root = from_sklearn_tree(clf)
    extractor = PathExtractor(tree_root)
    paths     = extractor.extract_paths()
    encryptor = RuleEncryptor(paillier_bits=1024)
    rules     = encryptor.encrypt_paths(paths)
    engine    = PaillierInferenceEngine(encryptor)
    pt_clf    = PlaintextClassifier()

    # Setup SDTC
    sdtc = SDTC(n_bins=10)
    sdtc.fit_encrypt(clf, X_train)

    # Select test samples
    test_samples = X_test[:n_queries]

    results = {}

    # 1. Plaintext DT
    print("\nMeasuring plaintext DT...")
    times = []
    for sample in test_samples:
        t0 = time.perf_counter()
        pt_clf.classify(list(sample), paths)
        times.append(time.perf_counter() - t0)
    results['plaintext'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms':  float(np.std(times) * 1000),
        'times':   [t * 1000 for t in times]
    }
    print(f"  Mean: {results['plaintext']['mean_ms']:.4f} ms ± "
          f"{results['plaintext']['std_ms']:.4f} ms")

    # 2. PrivPathInfer Paillier mode
    print("Measuring PrivPathInfer Paillier mode...")
    times = []
    for sample in test_samples:
        t0 = time.perf_counter()
        engine.classify_plaintext(list(sample), rules)
        times.append(time.perf_counter() - t0)
    results['privpathinfer_paillier'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms':  float(np.std(times) * 1000),
        'times':   [t * 1000 for t in times]
    }
    print(f"  Mean: {results['privpathinfer_paillier']['mean_ms']:.2f} ms ± "
          f"{results['privpathinfer_paillier']['std_ms']:.2f} ms")

    # 3. SDTC (simplified baseline)
    print("Measuring SDTC (simplified)...")
    times = []
    for sample in test_samples:
        t0 = time.perf_counter()
        sdtc.classify(sample)
        times.append(time.perf_counter() - t0)
    results['sdtc'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms':  float(np.std(times) * 1000),
        'times':   [t * 1000 for t in times]
    }
    print(f"  Mean: {results['sdtc']['mean_ms']:.4f} ms ± "
          f"{results['sdtc']['std_ms']:.4f} ms")

    # 4. SDTC Full (Algorithm 1, Liang et al. 2021)
    print("Measuring SDTC Full (Algorithm 1)...")
    sdtc_full = SDTCFull(n_bins=10)
    sdtc_full.fit_encrypt(clf, X_train)
    times = []
    for sample in test_samples:
        t0 = time.perf_counter()
        sdtc_full.classify(sample)
        times.append(time.perf_counter() - t0)
    results['sdtc_full'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms':  float(np.std(times) * 1000),
        'times':   [t * 1000 for t in times]
    }
    print(f"  Mean: {results['sdtc_full']['mean_ms']:.4f} ms ± "
          f"{results['sdtc_full']['std_ms']:.4f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<30} {'Mean (ms)':<15} {'Std (ms)'}")
    print("-" * 55)
    for method, data in results.items():
        print(f"{method:<30} {data['mean_ms']:<15.4f} {data['std_ms']:.4f}")

    # Save
    output = {
        'experiment': 3,
        'description': 'Per-Query Inference Time',
        'n_queries': n_queries,
        'max_depth': max_depth,
        'results': results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/exp3_inference_time.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/exp3_inference_time.json")
    return output


if __name__ == '__main__':
    run_experiment3()