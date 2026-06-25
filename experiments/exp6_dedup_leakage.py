"""
exp6_dedup_leakage.py - Experiment 6: Storage-Leakage Tradeoff (Contribution 4)
================================================================================
Demonstrates Pi_c (system/tunable_dedup.py): a public knob c that trades
Paillier-ciphertext storage against the threshold-equality leakage.

For each tree depth and each c in {1,2,4,8,16,32,inf}:
    - ciphertexts  S(c) = sum_v ceil(R_v / c)
    - storage (KB)
    - max value-linkability lambda(c)   (lower = more private)

Endpoint checks (Theorem):
    c = 1   -> S = T (total conditions),  lambda = baseline floor
    c = inf -> S = U (unique values),     lambda = max_v R_v (full equality)

A single real 1024-bit Paillier model is built (depth 8, c = 4) to verify
end-to-end correctness (decrypt == original) and replica unlinkability.

Dataset: PIMA Indians Diabetes.
"""

import os
import sys
import json
import math
import csv

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from system.tunable_dedup import profile, TunableDedupEncryptor
from crypto.paillier import keygen

C_VALUES = [1, 2, 4, 8, 16, 32, math.inf]
DEPTHS = [8, 12]
PAILLIER_BITS = 1024


def load_pima(path):
    X, y = [], []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            X.append([float(row['Pregnancies']), float(row['Glucose']),
                      float(row['BloodPressure']), float(row['SkinThickness']),
                      float(row['Insulin']), float(row['BMI']),
                      float(row['DiabetesPedigreeFunction']), float(row['Age'])])
            y.append(int(row['Outcome']))
    return np.array(X), np.array(y)


def build_paths(X_train, y_train, depth):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    return PathExtractor(from_sklearn_tree(clf)).extract_paths()


def run(data_path, out_dir):
    print("=" * 74)
    print("Experiment 6: Storage-Leakage Tradeoff of Tunable Threshold Dedup")
    print("=" * 74)

    X, y = load_pima(data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    for depth in DEPTHS:
        paths = build_paths(X_train, y_train, depth)
        rows = [profile(paths, c, PAILLIER_BITS) for c in C_VALUES]
        results[depth] = rows

        U = rows[0]['unique_values_U']
        T = rows[0]['total_conditions_T']
        maxR = rows[0]['max_reuse']
        print(f"\n--- Depth {depth}:  paths={rows[0]['n_paths']}  "
              f"T(conditions)={T}  U(unique)={U}  max reuse={maxR} ---")
        print(f"{'c':>5} {'ciphertexts S':>14} {'storage KB':>12} "
              f"{'max-linkability':>16}")
        print("-" * 50)
        for r in rows:
            print(f"{str(r['c']):>5} {r['ciphertexts_S']:>14} "
                  f"{r['storage_kb']:>12.2f} {r['max_linkability']:>16}")

        # endpoint assertions (Theorem)
        first, last = rows[0], rows[-1]
        assert first['c'] == 1 and first['ciphertexts_S'] == T, "c=1 must give S=T"
        assert last['c'] == 'inf' and last['ciphertexts_S'] == U, "c=inf must give S=U"
        assert last['max_linkability'] == maxR, "c=inf must leak full equality"
        print(f"  [endpoint check] c=1: S={T}=T OK | "
              f"c=inf: S={U}=U OK | lambda(inf)={maxR}=max reuse OK")
        if first['leakage_floor'] > 1:
            print(f"  [note] leakage floor {first['leakage_floor']} forced by a "
                  f"feature-unique threshold (Remark 1)")

    # ---- one real-crypto build: correctness + unlinkability ----
    print("\n" + "=" * 74)
    print(f"Real {PAILLIER_BITS}-bit Paillier build (depth 8, c=4): "
          f"correctness + unlinkability")
    print("=" * 74)
    paths8 = build_paths(X_train, y_train, 8)
    enc = TunableDedupEncryptor(PAILLIER_BITS, _keys=keygen(PAILLIER_BITS))
    model = enc.build(paths8, c=4)
    ok = enc.verify_correctness(model, paths8)
    unlink = enc.check_unlinkability(model)
    print(f"  ciphertexts built : {len(model.cipher_table)}")
    print(f"  correctness (decrypt == original, all paths) : {ok}")
    print(f"  replica unlinkability (all ciphertexts distinct): {unlink}")
    assert ok and unlink, "real-crypto verification failed"

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'exp6_dedup_leakage.json'), 'w') as f:
        json.dump({'experiment': 6,
                   'description': 'Storage-Leakage tradeoff of tunable dedup Pi_c',
                   'dataset': 'PIMA Indians Diabetes',
                   'paillier_bits': PAILLIER_BITS,
                   'c_values': [('inf' if c == math.inf else c) for c in C_VALUES],
                   'results': {str(d): results[d] for d in DEPTHS},
                   'real_build': {'depth': 8, 'c': 4,
                                  'ciphertexts': len(model.cipher_table),
                                  'correct': ok, 'unlinkable': unlink}}, f, indent=2)
    print(f"\nSaved results/exp6_dedup_leakage.json")
    return results


def make_figure(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xlabels = [str(r['c']) for r in results[DEPTHS[0]]]
    x = list(range(len(xlabels)))

    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(11, 4.2))
    for ax, depth in zip(axes, DEPTHS):
        rows = results[depth]
        storage = [r['storage_kb'] for r in rows]
        leak = [r['max_linkability'] for r in rows]

        ax.plot(x, storage, 'o-', color='black', label='Storage (KB)')
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_xlabel('linkability knob  c')
        ax.set_ylabel('Storage (KB)')
        ax.set_title(f'Tree depth {depth}')

        ax2 = ax.twinx()
        ax2.plot(x, leak, 's--', color='gray', label='Max value-linkability')
        ax2.set_ylabel('Max value-linkability  (lower = more private)')

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper center',
                  fontsize=8, frameon=False)

    fig.suptitle('Secure Threshold Deduplication: Storage vs. Leakage on PIMA',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure {out_path}")


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else 'data/diabetes.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'results'
    res = run(data, out)
    make_figure(res, os.path.join(out, 'fig6_storage_leakage.png'))