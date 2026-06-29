"""
exp8_update_comm.py - Experiment 8: Incremental-Update Communication (C3 headline)
==================================================================================
The robust, dataset-INDEPENDENT advantage of PrivPathInfer.

Key fact: an update batch always carries exactly BATCH_SIZE delete + BATCH_SIZE
insert operations, so PrivPathInfer's per-update communication is CONSTANT
(~2.5 KB at 1024-bit) regardless of tree depth or dataset. SDTC must re-encrypt
the whole decision table on every update: 2^depth entries x 32 B, exponential in
depth. The advantage therefore grows without bound as trees deepen, and is
identical across datasets at the same depth.

Outputs:
  (A) Depth sweep: PrivPathInfer (flat) vs SDTC (exponential) communication.
  (B) Cross-dataset table: each dataset's natural tree depth and the resulting
      update-communication advantage (shows robustness across PIMA / Breast
      Cancer / Heart).
"""

import os
import sys
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.datasets import load_dataset, DATASETS
from system.path_extractor import PathExtractor, from_sklearn_tree
from system.rule_encryptor import RuleEncryptor
from system.update_protocol import UpdateProtocol, BATCH_SIZE

SDTC_BYTES_PER_ENTRY = 32   # A[i]=16B + T[i]=16B
PAILLIER_BITS = 1024


def _field_bytes(v):
    if isinstance(v, int):
        return max(1, (v.bit_length() + 7) // 8)
    if isinstance(v, (bytes, bytearray)):
        return len(v)
    if isinstance(v, str):
        return 1
    return 4


def _rule_bytes(r):
    # enc_threshold (Paillier ct) + enc_feature_idx (PRP) + deletion_token (PRF)
    # + direction + 4 small integer fields (path_id, cond_idx, label, is_last, depth)
    return (_field_bytes(r.enc_threshold) + _field_bytes(r.enc_feature_idx)
            + _field_bytes(r.deletion_token) + _field_bytes(r.direction) + 4 * 4)


def measure_pp_update_bytes():
    """Measure one real 1024-bit update batch. CONSTANT across depth/dataset."""
    X, y, _, _ = load_dataset('pima')
    clf = DecisionTreeClassifier(max_depth=8, random_state=42).fit(X, y)
    paths = PathExtractor(from_sklearn_tree(clf)).extract_paths()
    enc = RuleEncryptor(paillier_bits=PAILLIER_BITS)
    rules = enc.encrypt_paths(paths)
    proto = UpdateProtocol(enc, real_paths=paths)
    batch = proto.create_update_batch(rules, list(rules)[:-3])   # a few real changes
    ins = sum(_rule_bytes(op.rule) for op in batch.insert_ops)
    dele = sum(_field_bytes(op.deletion_token) for op in batch.delete_ops)
    return ins + dele


def sdtc_update_bytes(depth):
    return (2 ** depth) * SDTC_BYTES_PER_ENTRY


def natural_depth(X, y, cap=12):
    """Tree depth when grown fully, capped at a realistic deployment depth
    (deep trees on small data overfit; uncapped depths produce misleading,
    overfit-driven advantage numbers)."""
    clf = DecisionTreeClassifier(random_state=42).fit(X, y)   # grow fully
    paths = PathExtractor(from_sklearn_tree(clf)).extract_paths()
    full = max((p.depth for p in paths), default=0)
    return min(full, cap), full


def run():
    pp_bytes = measure_pp_update_bytes()
    print("=" * 70)
    print("Experiment 8: Incremental-Update Communication (Contribution 3)")
    print("=" * 70)
    print(f"PrivPathInfer update = CONSTANT {pp_bytes} B = {pp_bytes/1024:.2f} KB "
          f"({BATCH_SIZE} insert + {BATCH_SIZE} delete ops, 1024-bit) — "
          f"independent of depth and dataset.\n")

    # (A) depth sweep
    depths = list(range(2, 15))
    print("(A) Depth sweep — update communication:")
    print(f"{'depth':>6} {'SDTC KB':>12} {'PrivPathInfer KB':>18} {'advantage':>10}")
    sweep = []
    for d in depths:
        s = sdtc_update_bytes(d)
        adv = s / pp_bytes
        sweep.append({'depth': d, 'sdtc_kb': round(s/1024, 2),
                      'pp_kb': round(pp_bytes/1024, 2), 'advantage': round(adv, 1)})
        print(f"{d:>6} {s/1024:>12.2f} {pp_bytes/1024:>18.2f} {adv:>9.1f}x")

    # (B) cross-dataset at natural depth (capped to a realistic deployment depth)
    print("\n(B) Cross-dataset — advantage at each dataset's (capped) tree depth:")
    print(f"{'dataset':>34} {'depth':>6} {'full':>5} {'SDTC KB':>10} {'PP KB':>8} {'advantage':>10}")
    per_ds = {}
    for name in DATASETS:
        X, y, _, disp = load_dataset(name)
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        d, full = natural_depth(Xtr, ytr)
        s = sdtc_update_bytes(d)
        adv = s / pp_bytes
        per_ds[name] = {'display': disp, 'depth': int(d), 'full_depth': int(full),
                        'sdtc_kb': round(s/1024, 2), 'pp_kb': round(pp_bytes/1024, 2),
                        'advantage': round(adv, 1)}
        print(f"{disp:>34} {d:>6} {full:>5} {s/1024:>10.2f} {pp_bytes/1024:>8.2f} {adv:>9.1f}x")

    os.makedirs('results', exist_ok=True)
    out = {'pp_update_bytes': pp_bytes, 'pp_update_kb': round(pp_bytes/1024, 2),
           'batch_size': BATCH_SIZE, 'paillier_bits': PAILLIER_BITS,
           'depth_sweep': sweep, 'per_dataset': per_ds}
    with open('results/exp8_update_comm.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/exp8_update_comm.json")
    return out


def make_figure(out, path='results/fig9_update_comm_depth.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'})
    C_PRIV, C_SDTC = '#1a6faf', '#d62728'
    sw = out['depth_sweep']
    d = [r['depth'] for r in sw]
    sdtc = [r['sdtc_kb'] for r in sw]
    pp = [r['pp_kb'] for r in sw]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(d, sdtc, 'o-', color=C_SDTC, label='SDTC (full re-encryption, $2^{depth}$)')
    ax.plot(d, pp, 's-', color=C_PRIV, label='PrivPathInfer (constant batch)')
    ax.set_yscale('log')
    ax.set_xlabel('Tree depth')
    ax.set_ylabel('Update communication (KB, log scale)')
    ax.set_title('Figure 9: Incremental-update communication grows exponentially\n'
                 'for SDTC but stays constant for PrivPathInfer')
    # mark each dataset's natural depth
    for name, info in out['per_dataset'].items():
        nd = info['depth']
        ax.axvline(nd, color='gray', ls=':', alpha=0.6)
        ax.annotate(info['display'].split('(')[0].strip(),
                    xy=(nd, max(pp)), rotation=90, fontsize=7.5,
                    color='gray', va='bottom', ha='right')
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


if __name__ == '__main__':
    res = run()
    make_figure(res)
