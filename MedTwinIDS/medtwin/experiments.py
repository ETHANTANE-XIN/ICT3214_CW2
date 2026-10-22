"""Experiments E1b-E5 plus explainability and runtime assets.

E1b Robustness: bootstrap confidence intervals on the test run and three extra unseen ward runs.
E2  Feature-set ablation: network only / biometric only / network+biometric / full twin.
E3  Zero-day generalisation: leave one attack family out of training, then test on it; includes the
    label-free twin consistency rules (medtwin.rules) as a designed remedy for unseen attacks.
E4  Attack intensity: detection recall for low / medium / high intensity campaigns.
E5  Window size: accuracy versus time-to-detect for 2 s, 5 s and 10 s windows.
Diag Spoofing diagnostic: impostor versus victim flows for each model.
XAI SHAP feature attribution for XGBoost and impurity importance for Random Forest.

Usage: python -m medtwin.experiments [--only assets,robust,e2,e3,e4,e5,xai,diag]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .models import MODEL_NAMES, fit_model, full_proba, xy, make_model
from .parser import packets_to_features
from .rules import TwinRules
from .train import load_flows, load_params, metrics_from_proba


def best_model_name() -> str:
    return json.loads((C.MODEL_DIR / "model_card.json").read_text())["selected_model"]


# --------------------------------------------------------------------------
def runtime_assets(train: pd.DataFrame):
    normal = train[train["attack"] == C.NORMAL]
    stats = pd.DataFrame({"mean": normal[C.ALL_FEATURES].mean(), "std": normal[C.ALL_FEATURES].std()})
    stats.to_csv(C.MODEL_DIR / "normal_stats.csv")
    iso = joblib.load(C.MODEL_DIR / "isolation_forest.joblib")
    s = iso.score_samples(normal[C.ALL_FEATURES].to_numpy(dtype=np.float32))
    thr = dict(standard_threshold=float(np.quantile(s, 0.01)), strict_threshold=float(np.quantile(s, 0.001)))
    (C.MODEL_DIR / "iforest_threshold.json").write_text(json.dumps(thr, indent=2))
    TwinRules().fit(train).save(C.MODEL_DIR / "twin_rules.json")
    print("runtime assets saved:", thr)
    return thr


# --------------------------------------------------------------------------
def e1_robustness(test: pd.DataFrame, n_boot: int = 1000):
    """Bootstrap 95% intervals on the test run, and the saved models on three extra unseen ward runs."""
    from sklearn.metrics import f1_score
    from .simulate import run_simulation

    pr = pd.read_csv(C.RESULTS_DIR / "test_predictions.csv")
    y = pr["attack"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    preds = {m: pr[f"pred_{m}"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy() for m in MODEL_NAMES}
    rng = np.random.default_rng(0)
    boot = {m: [] for m in MODEL_NAMES}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for m in MODEL_NAMES:
            boot[m].append(f1_score(y[idx], preds[m][idx], average="macro", labels=range(len(C.CLASSES)),
                                    zero_division=0))
    boot = {m: np.array(v) for m, v in boot.items()}
    best = best_model_name()
    rows = []
    for m in MODEL_NAMES:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        row = dict(model=m, macro_f1_ci_low=lo, macro_f1_ci_high=hi)
        if m != best:  # paired difference: same resamples for both models
            dlo, dhi = np.percentile(boot[best] - boot[m], [2.5, 97.5])
            row.update(diff_vs_best_low=dlo, diff_vs_best_high=dhi)
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e1_bootstrap.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    reg = pd.read_csv(C.DATA_DIR / "devices.csv")
    models = {m: joblib.load(C.MODEL_DIR / f"{m.lower().replace(' ', '_')}.joblib") for m in MODEL_NAMES}
    extra = []
    for seed, start in ((3001, 2.0), (3002, 11.0), (3003, 17.0)):
        cfg = C.ScenarioConfig(seed=seed, hours=1.0, start_hour=start, campaigns_per_hour=10.0)
        _, packets = run_simulation(cfg, verbose=False)
        flows, _ = packets_to_features(packets, reg)
        X, yy = xy(flows)
        for m, model in models.items():
            met = metrics_from_proba(yy, full_proba(model, X))
            extra.append(dict(seed=seed, start_hour=start, model=m, flows=len(flows), **met))
            print(f"  E1b run seed={seed} {m:13s} macro-F1 {met['macro_f1']:.4f} binary-F1 {met['f1']:.4f}")
    pd.DataFrame(extra).to_csv(C.RESULTS_DIR / "e1_extra_runs.csv", index=False)


# --------------------------------------------------------------------------
def e2_ablation(train, test, params):
    rows = []
    for set_name, feats in C.FEATURE_SETS.items():
        Xtr, ytr = xy(train, feats)
        Xte, yte = xy(test, feats)
        for name in MODEL_NAMES:
            m = fit_model(name, make_model(name, **params.get(name, {})), Xtr, ytr)
            proba = full_proba(m, Xte)
            met = metrics_from_proba(yte, proba)
            pred = proba.argmax(1)
            rec = {f"recall_{a}": float(((pred == i) & (yte == i)).sum() / max(1, (yte == i).sum()))
                   for i, a in enumerate(C.CLASSES) if i > 0}
            rows.append(dict(feature_set=set_name, n_features=len(feats), model=name, f1=met["f1"],
                             macro_f1=met["macro_f1"], fpr=met["fpr"], **rec))
            print(f"  E2 {set_name:18s} {name:13s} macro-F1 {met['macro_f1']:.4f} binary-F1 {met['f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "e2_ablation.csv", index=False)
    return df


def e3_zero_day(train, test, params, iso_thr):
    iso = joblib.load(C.MODEL_DIR / "isolation_forest.joblib")
    Xte_all, yte_all = xy(test)
    iso_scores = iso.score_samples(Xte_all)
    iso_flag = iso_scores < iso_thr["standard_threshold"]
    # label-free twin rules: thresholds come from normal training traffic only, so they are the same
    # whichever attack family is held out
    rule_flag = TwinRules.load(C.MODEL_DIR / "twin_rules.json").flag(test)
    best = best_model_name()
    normal_mask = yte_all == 0
    rows = []
    for k, attack in enumerate(C.ATTACKS, start=1):
        tr = train[train["attack"] != attack]
        Xtr, ytr = xy(tr)
        held = yte_all == k
        sup_flags = {}
        for name in MODEL_NAMES:
            m = fit_model(name, make_model(name, **params.get(name, {})), Xtr, ytr)
            pred = full_proba(m, Xte_all).argmax(1)
            flag = pred != 0
            sup_flags[name] = flag
            rows.append(dict(held_out=attack, detector=name, recall=float(flag[held].mean()),
                             fpr=float(flag[normal_mask].mean())))
            print(f"  E3 held-out {attack:20s} {name:13s} recall {flag[held].mean():.3f}")
        rows.append(dict(held_out=attack, detector="Isolation Forest", recall=float(iso_flag[held].mean()),
                         fpr=float(iso_flag[normal_mask].mean())))
        rows.append(dict(held_out=attack, detector="Twin rules", recall=float(rule_flag[held].mean()),
                         fpr=float(rule_flag[normal_mask].mean())))
        for name in MODEL_NAMES:
            hyb = sup_flags[name] | iso_flag
            rows.append(dict(held_out=attack, detector=f"Hybrid ({name} + IF)", recall=float(hyb[held].mean()),
                             fpr=float(hyb[normal_mask].mean())))
        for label, flag in ((f"Hybrid ({best} + twin rules)", sup_flags[best] | rule_flag),
                            (f"Hybrid ({best} + IF + twin rules)", sup_flags[best] | iso_flag | rule_flag)):
            rows.append(dict(held_out=attack, detector=label, recall=float(flag[held].mean()),
                             fpr=float(flag[normal_mask].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "e3_zero_day.csv", index=False)
    # sensitivity of the rules to their threshold quantile (reported, not used to choose the default)
    from . import rules as R
    sens = []
    for q in (0.999, 0.9995, 0.9999):
        default, R.QUANTILE = R.QUANTILE, q
        flag = TwinRules().fit(train).flag(test)
        R.QUANTILE = default
        row = {a: float(flag[(yte_all == k)].mean()) for k, a in enumerate(C.ATTACKS, start=1)}
        sens.append(dict(quantile=q, fpr=float(flag[normal_mask].mean()), **row))
    pd.DataFrame(sens).to_csv(C.RESULTS_DIR / "e3_rules_sensitivity.csv", index=False)
    return df


def e4_intensity():
    pr = pd.read_csv(C.RESULTS_DIR / "test_predictions.csv")
    att = pr[pr["label"] == 1]
    rows = []
    for name in MODEL_NAMES:
        det = att[f"pred_{name}"] != C.NORMAL
        g = det.groupby([att["attack"], att["intensity"]]).mean().rename("recall").reset_index()
        g["model"] = name
        rows.append(g)
    df = pd.concat(rows)
    df.to_csv(C.RESULTS_DIR / "e4_intensity.csv", index=False)
    return df


def detection_delays(flows: pd.DataFrame, flagged: np.ndarray, campaigns: pd.DataFrame, W: float) -> pd.DataFrame:
    """Time from campaign start to the end of the first window in which one of its flows is flagged."""
    f = flows.assign(flag=flagged)
    rows = []
    for c in campaigns.itertuples():
        hit = f[(f["campaign"] == c.campaign) & f["flag"]]
        delay = float((hit["window"].min() + 1) * W - c.start) if len(hit) else np.nan
        rows.append(dict(campaign=c.campaign, attack=c.attack, intensity=c.intensity, delay_s=delay,
                         detected=bool(len(hit))))
    return pd.DataFrame(rows)


def e5_window(params, best):
    tr_pk = pd.read_parquet(C.DATA_DIR / "train_packets.parquet")
    te_pk = pd.read_parquet(C.DATA_DIR / "test_packets.parquet")
    reg = pd.read_csv(C.DATA_DIR / "devices.csv")
    camps = pd.read_csv(C.DATA_DIR / "test_campaigns.csv")
    rows, delays = [], []
    for W in (2.0, 5.0, 10.0):
        t = time.perf_counter()
        tr, _ = packets_to_features(tr_pk, reg, W)
        te, _ = packets_to_features(te_pk, reg, W)
        parse_s = time.perf_counter() - t
        Xtr, ytr = xy(tr)
        Xte, yte = xy(te)
        m = fit_model(best, make_model(best, **params.get(best, {})), Xtr, ytr)
        proba = full_proba(m, Xte)
        met = metrics_from_proba(yte, proba)
        d = detection_delays(te, proba.argmax(1) != 0, camps, W)
        d["window"] = W
        delays.append(d)
        rows.append(dict(window_s=W, train_flows=len(tr), test_flows=len(te), f1=met["f1"], macro_f1=met["macro_f1"],
                         fpr=met["fpr"], campaigns_detected=float(d["detected"].mean()),
                         mean_delay_s=float(d["delay_s"].mean()), median_delay_s=float(d["delay_s"].median()),
                         parse_s=parse_s))
        print(f"  E5 W={W:4.1f}s macro-F1 {met['macro_f1']:.4f} detected {d['detected'].mean():.3f} "
              f"mean delay {d['delay_s'].mean():.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "e5_window.csv", index=False)
    pd.concat(delays).to_csv(C.RESULTS_DIR / "e5_delays.csv", index=False)
    return df


def xai(train, test, params):
    import shap
    rows = {}
    xgb = joblib.load(C.MODEL_DIR / "xgboost.joblib")
    rng = np.random.default_rng(0)
    # stratified sample so that rare attacks are represented
    idx = np.concatenate([rng.choice(np.flatnonzero(test["attack"] == a), size=min(300, (test["attack"] == a).sum()),
                                     replace=False) for a in C.CLASSES])
    Xs = test.iloc[idx][C.ALL_FEATURES].to_numpy(dtype=np.float32)
    sv = shap.TreeExplainer(xgb).shap_values(Xs)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        imp = np.abs(sv).mean(axis=(0, 2))
        per_class = np.abs(sv).mean(axis=0)  # (features, classes)
        pd.DataFrame(per_class, index=C.ALL_FEATURES, columns=C.CLASSES).to_csv(C.RESULTS_DIR / "shap_per_class.csv")
    else:
        imp = np.abs(sv).mean(axis=0)
    rows["XGBoost_SHAP"] = imp / imp.sum()
    rf = joblib.load(C.MODEL_DIR / "random_forest.joblib")
    rows["RandomForest_Gini"] = rf.feature_importances_ / rf.feature_importances_.sum()
    df = pd.DataFrame(rows, index=C.ALL_FEATURES)
    df["group"] = ["network" if f in C.NETWORK_FEATURES else "biometric" if f in C.BIOMETRIC_FEATURES else "twin"
                   for f in df.index]
    df.sort_values("XGBoost_SHAP", ascending=False).to_csv(C.RESULTS_DIR / "feature_importance.csv")
    print(df.groupby("group")[["XGBoost_SHAP", "RandomForest_Gini"]].sum().round(3))
    return df


def spoofing_diagnostic(test):
    """How each model handles the impostor side of spoofing: flows whose sender address differs from the
    registered address of the device identity they claim (MQTT client-ID takeover)."""
    reg = pd.read_csv(C.DATA_DIR / "devices.csv")
    ip_of = dict(zip(reg.device_id, reg.ip))
    sp = test[test["attack"] == "spoofing"].copy()
    impostor = (sp["device_id"].map(ip_of) != sp["init_ip"]).to_numpy()
    X = sp[C.ALL_FEATURES].to_numpy(dtype=np.float32)
    out = {"impostor_flows": int(impostor.sum()), "victim_flows": int((~impostor).sum())}
    for key, name in (("rf", "random_forest"), ("xgb", "xgboost"), ("mlp", "mlp")):
        pred = full_proba(joblib.load(C.MODEL_DIR / f"{name}.joblib"), X).argmax(1)
        spoof = C.CLASSES.index("spoofing")
        out[f"{key}_impostor"] = float((pred[impostor] == spoof).mean())
        out[f"{key}_impostor_as_normal"] = float((pred[impostor] == 0).mean())
        out[f"{key}_victim"] = float((pred[~impostor] == spoof).mean())
    (C.RESULTS_DIR / "spoofing_diagnostic.json").write_text(json.dumps(out, indent=2))
    print("  spoofing diagnostic:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="assets,robust,e2,e3,e4,e5,xai,diag")
    args = ap.parse_args()
    todo = set(args.only.split(","))
    train, test = load_flows("train"), load_flows("test")
    params = load_params()
    best = best_model_name()
    thr = runtime_assets(train) if "assets" in todo else json.loads((C.MODEL_DIR / "iforest_threshold.json").read_text())
    if "robust" in todo:
        e1_robustness(test)
    if "e4" in todo:
        e4_intensity()
    if "diag" in todo:
        spoofing_diagnostic(test)
    if "xai" in todo:
        xai(train, test, params)
    if "e2" in todo:
        e2_ablation(train, test, params)
    if "e3" in todo:
        e3_zero_day(train, test, params, thr)
    if "e5" in todo:
        e5_window(params, best)


if __name__ == "__main__":
    main()
