"""Benchmark the three detectors on the public WUSTL-EHMS-2020 dataset (Hady et al., IEEE Access 2020).

The dataset is not redistributed with this project. Download ``wustl-ehms-2020.csv`` from
https://www.cse.wustl.edu/~jain/ehms/index.html (or its Kaggle mirror), place it in data/ and run:

    python -m medtwin.benchmark_wustl --csv data/wustl-ehms-2020.csv

The script mirrors the comparison in the original paper (network-only, biometric-only and combined
features) and adds our three models. Address, port, MAC and packet-number columns are removed
because they identify the attacker host rather than describing behaviour (a known leakage source).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .models import MODEL_NAMES, fit_model, make_model

ID_COLUMNS = {"Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac", "Packet_num"}
BIOMETRIC = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]


def load(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    drop = ID_COLUMNS | {label_col, "Attack Category", "Attack_Category"}
    X = df[[c for c in df.columns if c not in drop]].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any()].fillna(0.0)
    bio = [c for c in BIOMETRIC if c in X.columns]
    net = [c for c in X.columns if c not in bio]
    return X, y, {"network_only": net, "biometric_only": bio, "combined": net + bio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(C.DATA_DIR / "wustl-ehms-2020.csv"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    X, y, sets = load(args.csv)
    print(f"rows {len(X):,} | attacks {y.sum():,} ({y.mean():.1%}) | features {X.shape[1]}")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    rows = []
    for set_name, cols in sets.items():
        Xs = X[cols].to_numpy(dtype=np.float32)
        for name in MODEL_NAMES:
            folds = []
            for tr, te in cv.split(Xs, y):
                t = time.perf_counter()
                m = fit_model(name, make_model(name), Xs[tr], y[tr])
                fit_s = time.perf_counter() - t
                p = m.predict_proba(Xs[te])[:, 1]
                pred = (p >= 0.5).astype(int)
                folds.append(dict(accuracy=accuracy_score(y[te], pred), precision=precision_score(y[te], pred, zero_division=0),
                                  recall=recall_score(y[te], pred), f1=f1_score(y[te], pred), auc=roc_auc_score(y[te], p),
                                  fit_s=fit_s))
            f = pd.DataFrame(folds)
            row = dict(feature_set=set_name, model=name, **{k: f[k].mean() for k in f}, f1_sd=f["f1"].std())
            rows.append(row)
            print(f"  {set_name:15s} {name:13s} acc {row['accuracy']:.4f} F1 {row['f1']:.4f} AUC {row['auc']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "wustl_benchmark.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
