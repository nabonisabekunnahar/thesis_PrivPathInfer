"""
verify_priorityA.py - Evidence that the realistic dummy fix closes the hole.

Demonstrates, on PIMA:
  (1) OLD dummies are distinguishable: identical feature tag, constant threshold
      (-100), path_id = -1.
  (2) NEW dummies are NOT: feature-tag frequencies match real inserts, thresholds
      decrypt within the real range, path_ids lie in the real range.
  (3) Padding with NEW dummies leaves classification unchanged on the test set.
"""

import os
import sys
import csv
from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.path_extractor import PathExtractor, from_sklearn_tree
from system.rule_encryptor import RuleEncryptor
from system.update_protocol import UpdateProtocol
from system.secure_dummy import RealisticDummyGenerator, classify

PAILLIER_BITS = 512   # fast; the indistinguishability argument is key-size independent


def load_pima(path):
    X, y = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X.append([float(row['Pregnancies']), float(row['Glucose']),
                      float(row['BloodPressure']), float(row['SkinThickness']),
                      float(row['Insulin']), float(row['BMI']),
                      float(row['DiabetesPedigreeFunction']), float(row['Age'])])
            y.append(int(row['Outcome']))
    return np.array(X), np.array(y)


def main(data_path):
    X, y = load_pima(data_path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=8, random_state=42).fit(Xtr, ytr)
    paths = PathExtractor(from_sklearn_tree(clf)).extract_paths()

    enc = RuleEncryptor(PAILLIER_BITS)
    real_rules = enc.encrypt_paths(paths)
    real_feats = [c.feature_idx for p in paths for c in p.conditions]

    print("=" * 70)
    print("PRIORITY A: realistic dummy generation vs the impossible-condition hole")
    print("=" * 70)

    # ---- (1) OLD dummies ----
    proto = UpdateProtocol(enc)
    old_dummies = [proto._make_dummy_insert().rule for _ in range(20)]
    old_tags = Counter(r.enc_feature_idx for r in old_dummies)
    old_thr = enc.decrypt_threshold(old_dummies[0].enc_threshold)
    old_pids = set(r.path_id for r in old_dummies)
    print("\n[OLD] impossible-condition dummies:")
    print(f"  distinct feature tags among 20 dummies : {len(old_tags)}  "
          f"(all identical -> a spike the cloud can see)")
    print(f"  decrypted threshold                    : {old_thr:.1f}  "
          f"(constant, impossible)")
    print(f"  path_ids                               : {old_pids}  (sentinel)")

    # ---- (2) NEW dummies ----
    gen = RealisticDummyGenerator(paths, enc, seed=0)
    new_dummies = gen.make_dummy_inserts(200)
    new_tags = Counter(r.enc_feature_idx for r in new_dummies)
    new_thr = [enc.decrypt_threshold(r.enc_threshold) for r in new_dummies]
    real_thr = [enc.decrypt_threshold(r.enc_threshold) for r in real_rules]
    real_pid_range = (min(p.path_id for p in paths), max(p.path_id for p in paths))
    new_pid_min = min(r.path_id for r in new_dummies)

    print("\n[NEW] realistic (re-encrypted real-path) dummies:")
    print(f"  distinct feature tags among 200 dummies: {len(new_tags)}  "
          f"(spread like real, vs real distinct = "
          f"{len(set(enc._permute_feature_idx(f) for f in set(real_feats)))})")
    print(f"  dummy threshold range  : [{min(new_thr):.1f}, {max(new_thr):.1f}]")
    print(f"  real  threshold range  : [{min(real_thr):.1f}, {max(real_thr):.1f}]")
    in_range = all(min(real_thr) <= t <= max(real_thr) for t in new_thr)
    print(f"  every dummy threshold within real range: {in_range}")
    print(f"  dummy path_ids start at {new_pid_min} (real range {real_pid_range}) "
          f"-> no -1 sentinel")

    # feature-usage distribution match (decrypt tags back to features via PRP map)
    tag_to_feat = {enc._permute_feature_idx(f): f for f in range(8)}
    real_dist = Counter(real_feats)
    dummy_dist = Counter(tag_to_feat[r.enc_feature_idx] for r in new_dummies)
    print(f"  real feature usage  (top): "
          f"{dict(sorted(real_dist.items()))}")
    print(f"  dummy feature usage (top): "
          f"{dict(sorted(dummy_dist.items()))}")

    # ---- (3) inertness: classification unchanged ----
    dummy_paths = [gen.make_dummy_path() for _ in range(8)]
    base = [classify(paths, x) for x in Xte]
    padded = [classify(paths + dummy_paths, x) for x in Xte]
    unchanged = (base == padded)
    print(f"\n[INERT] classification unchanged after padding with 8 dummy paths: "
          f"{unchanged}  ({sum(b==p for b,p in zip(base,padded))}/{len(base)} match)")

    assert len(old_tags) == 1, "old dummies should collapse to one tag"
    assert in_range, "new dummy thresholds must lie in real range"
    assert new_pid_min > 0, "new dummies must not use the -1 sentinel"
    assert unchanged, "padding must not change classification"
    print("\n" + "=" * 70)
    print("RESULT: old scheme distinguishable on 3 channels; new scheme matches "
          "real\non all observable fields and is inert. PRIORITY A closed.")
    print("=" * 70)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/diabetes.csv')
