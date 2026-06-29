"""
datasets.py - Unified multi-dataset loader for PrivPathInfer experiments.

Provides three binary medical datasets so experiments can validate across more
than PIMA (addresses the "why one dataset?" defense question):

  pima           - PIMA Indians Diabetes      (768 x 8,  all continuous)
  breast_cancer  - Breast Cancer Wisconsin Dx (569 x 30, all continuous)
  heart          - Heart Disease (Cleveland)  (303 x 13, mixed cont./categorical)

Breast Cancer and Heart are also used by Tai et al. 2017, giving direct
baseline-comparison parity.

Usage:
    from experiments.datasets import load_dataset, DATASETS
    X, y, feature_names, display = load_dataset('breast_cancer')
"""

import csv
import os
import numpy as np

PIMA_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
HEART_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

DATASETS = ['pima', 'breast_cancer', 'heart', 'framingham']

DISPLAY = {
    'pima':          'PIMA Diabetes (8 cont.)',
    'breast_cancer': 'Breast Cancer Wisconsin (30 cont.)',
    'heart':         'Heart Disease Cleveland (13 mixed)',
    'framingham':    'Framingham CHD (15 mixed, deep)',
}


def _read(path, feature_cols, target_col):
    X, y = [], []
    # utf-8-sig strips any BOM (the heart CSV has one)
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            X.append([float(row[c]) for c in feature_cols])
            y.append(int(float(row[target_col])))
    return np.array(X), np.array(y), list(feature_cols)


def load_dataset(name, data_dir='data'):
    """Return (X, y, feature_names, display_name)."""
    name = name.lower()
    if name in ('pima', 'diabetes'):
        X, y, f = _read(os.path.join(data_dir, 'diabetes.csv'), PIMA_COLS, 'Outcome')
        return X, y, f, DISPLAY['pima']
    if name in ('breast_cancer', 'breast', 'bcw'):
        path = os.path.join(data_dir, 'breast_cancer.csv')
        with open(path, newline='') as fh:
            cols = next(csv.reader(fh))
        feats = [c for c in cols if c != 'target']
        X, y, f = _read(path, feats, 'target')
        return X, y, f, DISPLAY['breast_cancer']
    if name in ('heart', 'heart_disease', 'cleveland'):
        X, y, f = _read(os.path.join(data_dir, 'heart.csv'), HEART_COLS, 'target')
        return X, y, f, DISPLAY['heart']
    if name in ('framingham', 'chd'):
        path = os.path.join(data_dir, 'framingham.csv')
        with open(path, newline='') as fh:
            cols = next(csv.reader(fh))
        feats = [c for c in cols if c != 'target']
        X, y, f = _read(path, feats, 'target')
        return X, y, f, DISPLAY['framingham']
    raise ValueError(f"unknown dataset: {name}")


if __name__ == '__main__':
    for d in DATASETS:
        X, y, f, disp = load_dataset(d)
        print(f"{d:14s} {X.shape[0]:4d} x {X.shape[1]:2d}  "
              f"classes={sorted(set(y.tolist()))}  {disp}")
