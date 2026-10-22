"""Train, tune and compare the three detectors (experiment E1).

Protocol (leakage-aware, following Arp et al. 2022):
  1. Hyperparameters are tuned on a *temporal* hold-out: the first 75 % of the
     training run fits the model, the last 25 % validates it.
  2. Five-fold StratifiedGroupKFold cross-validation on the training run, with
     30-minute time blocks as groups, estimates stability.
  3. The deployed model is chosen on cross-validation macro-F1 (ties broken by
     latency) and only then evaluated once on the independent test run, which
     was simulated with a different seed, different patients and a different
     time of day.

Usage: python -m medtwin.train [--tune] [--skip-cv]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .models import MODEL_NAMES, fit_model, full_proba, make_isolation_forest, make_model, xy

PARAM_GRID = {
    "Random Forest": {"n_estimators": [200, 400], "max_depth": [None, 24], "min_samples_leaf": [1, 3]},
    "XGBoost": {"max_depth": [6, 10], "learning_rate": [0.1, 0.05], "n_estimators": [300, 500]},
    "MLP": {"hidden_layer_sizes": [(128, 64, 32), (256, 128, 64)], "alpha": [1e-4, 1e-3]},
}
PARAMS_FILE = C.RESULTS_DIR / "best_params.json"


def load_flows(name: str) -> pd.DataFrame:
    return pd.read_csv(C.DATA_DIR / f"{name}_flows.csv", low_memory=False)


def time_blocks(df: pd.DataFrame, minutes: float = 30.0) -> np.ndarray:
    return (df["t_start"] // (minutes * 60)).astype(int).to_numpy()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def single_latency_ms(model, x_row: np.ndarray, reps: int = 100) -> float:
    """Median time to score one flow, measured single-threaded so that thread-pool start-up
    (which dominates for one row) does not penalise the parallel tree ensembles."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    old = getattr(est, "n_jobs", None)
    if old is not None:
        est.set_params(n_jobs=1)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict_proba(x_row)
        times.append(time.perf_counter() - t)
    if old is not None:
        est.set_params(n_jobs=old)
    return float(np.median(times) * 1000)


def metrics_from_proba(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    yb, pb = (y != 0).astype(int), (pred != 0).astype(int)
    p_att = 1.0 - proba[:, 0]
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    present = sorted(set(y) | set(pred))
    return {
        "accuracy": accuracy_score(yb, pb),
        "precision": precision_score(yb, pb, zero_division=0),
        "recall": recall_score(yb, pb, zero_division=0),
        "f1": f1_score(yb, pb, zero_division=0),
        "fpr": fp / max(1, fp + tn),
        "roc_auc": roc_auc_score(yb, p_att) if len(set(yb)) > 1 else float("nan"),
        "mc_accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", labels=present, zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", labels=present, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", labels=present, zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }


# --------------------------------------------------------------------------
# Tuning and cross-validation
# --------------------------------------------------------------------------
def tune(train: pd.DataFrame, names=MODEL_NAMES) -> dict:
    cut = train["t_start"].quantile(0.75)
    fit_df, val_df = train[train["t_start"] < cut], train[train["t_start"] >= cut]
    Xf, yf = xy(fit_df)
    Xv, yv = xy(val_df)
    best, log = {}, []
    for name in names:
        grid = PARAM_GRID[name]
        keys = list(grid)
        scores = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            t = time.perf_counter()
            m = fit_model(name, make_model(name, **params), Xf, yf)
            s = metrics_from_proba(yv, full_proba(m, Xv))["macro_f1"]
            scores.append((s, params))
            log.append(dict(model=name, params=str(params), val_macro_f1=round(s, 4), fit_s=round(time.perf_counter() - t, 1)))
            print(f"  tune {name:13s} {params} -> val macro-F1 {s:.4f}")
        best[name] = max(scores, key=lambda z: z[0])[1]
    pd.DataFrame(log).to_csv(C.RESULTS_DIR / "tuning_log.csv", index=False)
    PARAMS_FILE.write_text(json.dumps({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                       for k, v in best.items()}, indent=2))
    return best


def load_params() -> dict:
    if PARAMS_FILE.exists():
        raw = json.loads(PARAMS_FILE.read_text())
        return {k: {kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in raw.items()}
    return {n: {} for n in MODEL_NAMES}


def cross_validate(train: pd.DataFrame, params: dict, folds: int = 5) -> pd.DataFrame:
    X, y = xy(train)
    groups = time_blocks(train)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        for name in MODEL_NAMES:
            m = fit_model(name, make_model(name, **params.get(name, {})), X[tr], y[tr])
            met = metrics_from_proba(y[va], full_proba(m, X[va]))
            rows.append(dict(fold=k + 1, model=name, **{kk: met[kk] for kk in ("f1", "macro_f1", "accuracy", "mc_accuracy", "fpr")}))
            print(f"  CV fold {k + 1} {name:13s} macro-F1 {met['macro_f1']:.4f} binary-F1 {met['f1']:.4f}")
    cvdf = pd.DataFrame(rows)
    cvdf.to_csv(C.RESULTS_DIR / "cv_folds.csv", index=False)
    return cvdf


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="run the hyperparameter search first")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_flows("train"), load_flows("test")
    print(f"train flows {len(train):,} | test flows {len(test):,}")
    params = tune(train) if args.tune else load_params()
    print("hyperparameters:", params)

    if not args.skip_cv:
        cvdf = cross_validate(train, params)
    else:
        cvdf = pd.read_csv(C.RESULTS_DIR / "cv_folds.csv")
    cv_summary = cvdf.groupby("model").agg(cv_macro_f1=("macro_f1", "mean"), cv_macro_f1_sd=("macro_f1", "std"),
                                           cv_f1=("f1", "mean"), cv_f1_sd=("f1", "std"))

    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    rows, preds = [], test[["window", "t_start", "init_ip", "resp_ip", "svc_port", "device_id", "attack", "campaign",
                            "intensity", "label"]].copy()
    for name in MODEL_NAMES:
        t = time.perf_counter()
        model = fit_model(name, make_model(name, **params.get(name, {})), Xtr, ytr)
        train_s = time.perf_counter() - t
        t = time.perf_counter()
        proba = full_proba(model, Xte)
        infer_s = time.perf_counter() - t
        met = metrics_from_proba(yte, proba)
        path = C.MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, path, compress=3)
        met.update(model=name, train_s=train_s, infer_us_per_flow=infer_s / len(Xte) * 1e6,
                   latency_ms=single_latency_ms(model, Xte[:1]), size_kb=path.stat().st_size / 1024)
        rows.append(met)
        pred = proba.argmax(1)
        preds[f"pred_{name}"] = [C.CLASSES[i] for i in pred]
        preds[f"p_attack_{name}"] = 1 - proba[:, 0]
        cm = confusion_matrix(yte, pred, labels=range(len(C.CLASSES)))
        pd.DataFrame(cm, index=C.CLASSES, columns=C.CLASSES).to_csv(C.RESULTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.csv")
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=range(len(C.CLASSES)), zero_division=0)
        pd.DataFrame(dict(cls=C.CLASSES, precision=p, recall=r, f1=f, support=s)).to_csv(
            C.RESULTS_DIR / f"per_class_{name.lower().replace(' ', '_')}.csv", index=False)
        print(f"TEST {name:13s} acc {met['accuracy']:.4f} P {met['precision']:.4f} R {met['recall']:.4f} "
              f"F1 {met['f1']:.4f} macroF1 {met['macro_f1']:.4f} AUC {met['roc_auc']:.4f} train {train_s:.1f}s")

    comp = pd.DataFrame(rows).set_index("model").join(cv_summary)
    # Selection uses cross-validation only (never the test run); latency breaks near-ties
    ranked = comp.sort_values(["cv_macro_f1", "latency_ms"], ascending=[False, True])
    top = ranked.index[0]
    close = ranked[ranked["cv_macro_f1"] >= ranked["cv_macro_f1"].iloc[0] - 0.002]
    best = close.sort_values("latency_ms").index[0] if len(close) > 1 else top
    comp["selected"] = comp.index == best
    comp.to_csv(C.RESULTS_DIR / "model_comparison.csv")
    preds.to_csv(C.RESULTS_DIR / "test_predictions.csv", index=False)

    # Unsupervised safety net trained on normal traffic only
    iso = make_isolation_forest()
    iso.fit(Xtr[ytr == 0])
    joblib.dump(iso, C.MODEL_DIR / "isolation_forest.joblib", compress=3)
    iso_flag = (iso.predict(Xte) == -1).astype(int)
    preds["iforest_flag"] = iso_flag
    preds.to_csv(C.RESULTS_DIR / "test_predictions.csv", index=False)

    best_model = joblib.load(C.MODEL_DIR / f"{best.lower().replace(' ', '_')}.joblib")
    joblib.dump(best_model, C.MODEL_DIR / "best_model.joblib", compress=3)
    card = dict(selected_model=best, features=C.ALL_FEATURES, classes=C.CLASSES, window_seconds=C.WINDOW_SECONDS,
                alert_threshold=0.5, hyperparameters={k: str(v) for k, v in params.get(best, {}).items()},
                test_metrics={k: float(v) for k, v in comp.loc[best].items() if isinstance(v, (int, float, np.floating))},
                selection_rule="highest 5-fold time-blocked CV macro-F1; near-ties (<0.002) broken by single-flow latency")
    (C.MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print("\nSelected model:", best)
    print(comp.round(4).to_string())


if __name__ == "__main__":
    main()
