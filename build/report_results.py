"""Results, discussion and conclusion of the final report. All numbers come from the result files."""
import json
import math

import pandas as pd

from common import FIG, RES, f3, f4, pct

CLS = ["normal", "dos_flood", "spoofing", "data_alteration", "unauthorised_access", "reconnaissance", "exfiltration"]
NAMES = {"normal": "Normal", "dos_flood": "DoS flood", "spoofing": "Spoofing", "data_alteration": "Data alteration",
         "unauthorised_access": "Unauth. access", "reconnaissance": "Reconnaissance", "exfiltration": "Exfiltration"}
MODELS = ["Random Forest", "XGBoost", "MLP"]


def num(v, d=3):
    """Number with a typographic minus sign (no minus on a value that rounds to zero)."""
    v = 0.0 if round(v, d) == 0 else v
    return f"{v:.{d}f}".replace("-", "\u2212")


ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


def refs(text, d):
    """Fill {FIG}/{FIGNEXT}/{TAB}/{TABNEXT} with the builder's current numbering."""
    return (text.replace("{FIGNEXT2}", f"Fig. {d.fig_no + 2}").replace("{FIGNEXT}", f"Fig. {d.fig_no + 1}")
            .replace("{FIG}", f"Fig. {d.fig_no}").replace("{TABNEXT}", f"Table {ROMAN[d.tab_no + 1]}")
            .replace("{TAB}", f"Table {ROMAN[d.tab_no]}").replace("{TABPREV}", f"Table {ROMAN[d.tab_no - 1]}"))


def load():
    ctx = {}
    ctx["pc"] = {m: pd.read_csv(RES / f"per_class_{m.lower().replace(' ', '_')}.csv").set_index("cls") for m in MODELS}
    ctx["ab"] = pd.read_csv(RES / "e2_ablation.csv")
    ctx["zd"] = pd.read_csv(RES / "e3_zero_day.csv")
    ctx["it"] = pd.read_csv(RES / "e4_intensity.csv")
    ctx["ws"] = pd.read_csv(RES / "e5_window.csv")
    ctx["fi"] = pd.read_csv(RES / "feature_importance.csv", index_col=0)
    ctx["spc"] = pd.read_csv(RES / "shap_per_class.csv", index_col=0)
    ctx["live"] = json.loads((RES / "e6_live_live_summary.json").read_text())["overall"]
    ctx["replay"] = json.loads((RES / "e6_replay_live_summary.json").read_text())["overall"]
    ctx["live_c"] = pd.read_csv(RES / "e6_live_live_campaigns.csv")
    ctx["replay_c"] = pd.read_csv(RES / "e6_replay_live_campaigns.csv")
    ctx["diag"] = json.loads((RES / "spoofing_diagnostic.json").read_text())
    ctx["boot"] = pd.read_csv(RES / "e1_bootstrap.csv").set_index("model")
    ctx["extra"] = pd.read_csv(RES / "e1_extra_runs.csv")
    ctx["sens"] = pd.read_csv(RES / "e3_rules_sensitivity.csv")
    ctx["live_rules"] = json.loads((RES / "e6_live_rules_live_summary.json").read_text())["overall"]
    return ctx


def results(d, C, ctx):
    comp, best = ctx["comp"], ctx["best"]
    pc, ab, zd, it, ws, fi, spc = (ctx[k] for k in ("pc", "ab", "zd", "it", "ws", "fi", "spc"))
    rf, xg, ml = (comp.loc[m] for m in MODELS)
    diag = ctx["diag"]
    d.h1("Results and Insight")

    # ---------------------------------------------------------------- E1
    d.h2("E1: Model Comparison")
    boot, extra = ctx["boot"], ctx["extra"]
    ex = extra.groupby("model")["macro_f1"].agg(["mean", "std"])
    pctf = lambda v: f"{100 * v:.2f}"
    metric_rows = [
        ("Accuracy (%)", lambda r, m: pctf(r["accuracy"])),
        ("Precision (%)", lambda r, m: pctf(r["precision"])),
        ("Recall (%)", lambda r, m: pctf(r["recall"])),
        ("F1-score (%)", lambda r, m: pctf(r["f1"])),
        ("False positive rate (%)", lambda r, m: pctf(r["fpr"])),
        ("ROC AUC", lambda r, m: f"{r['roc_auc']:.4f}"),
        ("Macro F1, 7 classes", lambda r, m: f3(r["macro_f1"])),
        ("  95% bootstrap interval", lambda r, m: f"{boot.loc[m, 'macro_f1_ci_low']:.3f}\u2013{boot.loc[m, 'macro_f1_ci_high']:.3f}"),
        ("Macro F1, 5-fold CV", lambda r, m: f"{r['cv_macro_f1']:.3f} \u00b1 {r['cv_macro_f1_sd']:.3f}"),
        ("Macro F1, 3 extra runs", lambda r, m: f"{ex.loc[m, 'mean']:.3f} \u00b1 {ex.loc[m, 'std']:.3f}"),
        ("Training time (s)", lambda r, m: f"{r['train_s']:.1f}"),
        ("Batch inference (\u00b5s / flow)", lambda r, m: f"{r['infer_us_per_flow']:.1f}"),
        ("Single-flow latency (ms)", lambda r, m: f"{r['latency_ms']:.2f}"),
        ("Model size (MB)", lambda r, m: f"{r['size_kb'] / 1024:.1f}"),
    ]
    rows = [[label] + [fn(comp.loc[m], m) for m in MODELS] for label, fn in metric_rows]
    d.table("Detection Quality and Processing Time on the Independent Test Run",
            ["Metric", "RF", "**XGBoost**", "MLP"], rows, [1.38, 0.64, 0.66, 0.64], size=6.5,
            align=["left", "center", "center", "center"],
            footnote="Accuracy to ROC AUC: attack versus normal. Latency: median time to score one flow on one CPU "
                     "thread. Extra runs: three new 1 h wards (seeds 3001 to 3003, starting 02:00, 11:00 and 17:00).")
    d.p(refs("{TAB} shows that all three models", d) + f" separate attacks from normal traffic almost perfectly: binary "
        f"F1-scores are {f4(rf['f1'])} (RF), {f4(xg['f1'])} (XGBoost) and {f4(ml['f1'])} (MLP), with precision of at "
        f"least {min(rf['precision'], xg['precision'], ml['precision']):.3f}, recall of at least "
        f"{min(rf['recall'], xg['recall'], ml['recall']):.3f}, false positive "
        f"rates of at most {100 * max(rf['fpr'], xg['fpr'], ml['fpr']):.2f}% and ROC AUC of at least "
        f"{min(rf['roc_auc'], xg['roc_auc'], ml['roc_auc']):.4f}. Binary accuracy alone would therefore hide the real "
        f"differences. The macro F1-score, which gives each attack class equal weight, does not: XGBoost reaches "
        f"{f3(xg['macro_f1'])}, the MLP {f3(ml['macro_f1'])} and RF only {f3(rf['macro_f1'])}.")
    rows = [[NAMES[c]] + [f"{pc[m].loc[c, 'f1']:.3f}" for m in MODELS] + [f"{int(pc[best].loc[c, 'support']):,}"]
            for c in CLS]
    d.table("Per-class F1-score on the Test Run", ["Class", "RF", "XGBoost", "MLP", "Flows"], rows,
            [1.2, 0.5, 0.6, 0.5, 0.6], size=6.5)
    loud = min(pc[m].loc[c, "f1"] for m in MODELS for c in ("dos_flood", "reconnaissance", "exfiltration",
                                                           "unauthorised_access"))
    f1s = lambda c: ", ".join(f"{pc[m].loc[c, 'f1']:.3f}" for m in MODELS)
    # the explanation below is about RF, so it must still be the weakest model on spoofing
    assert min(MODELS, key=lambda m: pc[m].loc["spoofing", "f1"]) == "Random Forest", "update the spoofing text"
    d.p(refs("{TAB} locates the gap.", d) + f" DoS, reconnaissance, exfiltration and unauthorised access are recognised by every "
        f"model (F1 of {math.floor(loud * 100) / 100:.2f} or more), because their footprints are loud: fan-out, volume or refused logins. The two "
        f"integrity attacks are harder: F1-scores are {f1s('spoofing')} on spoofing and {f1s('data_alteration')} on "
        f"data alteration (RF, XGBoost, MLP). A closer look at spoofing shows why RF lags there: it labels "
        f"{pct(diag['rf_impostor_as_normal'], 0)} of the impostor's flows (the attacker publishing under "
        f"a victim's identity) as normal, whereas XGBoost and the MLP catch {pct(diag['xgb_impostor'], 0)} and "
        f"{pct(diag['mlp_impostor'], 0)} of them. These flows look like ordinary telemetry; the only evidence is a "
        "handful of twin features (duplicated or missing sequence numbers, a heart rate that disagrees with the "
        "patient's wearable, and an unregistered or unexpected sender). RF considers a random subset of about seven "
        "features at each split and averages independently grown trees, so these few weak signals are often outvoted. "
        "Boosting fits each new tree to the errors of the previous ones, so XGBoost concentrates on exactly these "
        "hard flows; the MLP combines all features in every unit and also captures them.")

    d.h2("Why XGBoost Was Selected")
    ex_runs = extra.pivot(index="seed", columns="model", values="macro_f1")
    wins = int((ex_runs.idxmax(axis=1) == best).sum())
    others = sorted([("RF", rf), ("the MLP", ml)], key=lambda o: -o[1]["cv_macro_f1"])
    narrowest = xg["cv_macro_f1_sd"] <= min(rf["cv_macro_f1_sd"], ml["cv_macro_f1_sd"])
    fastest = xg["train_s"] <= min(rf["train_s"], ml["train_s"])
    d.p(f"The selection rule uses cross-validation only. XGBoost has the highest cross-validation macro F1-score "
        f"({xg['cv_macro_f1']:.3f} \u00b1 {xg['cv_macro_f1_sd']:.3f})"
        + (" and the smallest spread across folds" if narrowest else "") + ", ahead of "
        + " and ".join(f"{n} ({o['cv_macro_f1']:.3f} \u00b1 {o['cv_macro_f1_sd']:.3f})" for n, o in others)
        + f". On the test run, a paired cluster bootstrap (resampling whole campaigns) shows that its lead over RF is "
        f"real (95% interval of the macro-F1 difference {num(boot.loc['Random Forest', 'diff_vs_best_low'])} to "
        f"{num(boot.loc['Random Forest', 'diff_vs_best_high'])}) but that it is statistically tied with the MLP "
        f"({num(boot.loc['MLP', 'diff_vs_best_low'])} to {num(boot.loc['MLP', 'diff_vs_best_high'])}). To separate "
        f"them we ran the saved models on three further unseen wards: XGBoost was best in {wins} of 3 "
        f"(mean {ex.loc[best, 'mean']:.3f}, against {ex.loc['MLP', 'mean']:.3f} for the MLP and "
        f"{ex.loc['Random Forest', 'mean']:.3f} for RF). In processing time XGBoost is also a good fit: "
        + (f"it trains fastest ({xg['train_s']:.1f} s against {rf['train_s']:.1f} s and {ml['train_s']:.1f} s)"
           if fastest else f"it trains in {xg['train_s']:.1f} s (RF {rf['train_s']:.1f} s, MLP {ml['train_s']:.1f} s)")
        + f", scores one flow in {xg['latency_ms']:.2f} ms and is {xg['size_kb'] / 1024:.1f} MB on disk. The MLP is "
        f"faster per flow ({ml['latency_ms']:.2f} ms) but "
        + ("less stable and " if ml["cv_macro_f1_sd"] > xg["cv_macro_f1_sd"] else "")
        + "needs scaling and oversampling; RF is "
        f"{rf['latency_ms'] / xg['latency_ms']:.0f} times slower per flow and {rf['size_kb'] / xg['size_kb']:.0f} times "
        "larger. With about 30 flows per 5 s window, XGBoost needs well under 0.1 s of model time per window, so "
        "accuracy and stability, not speed, decided the choice.")

    # ---------------------------------------------------------------- E2
    d.h2("E2: Do Twin Features Help?")
    d.figure(FIG / "fig_ablation.png",
             "Ablation on the test run: macro F1-score and spoofing recall for each feature set and model.", width_in=3.3)
    d.p(refs(ctx["text_e2"], d))

    # ---------------------------------------------------------------- E3
    d.h2("E3: Unseen (Zero-day) Attacks")
    d.figure(FIG / "fig_zero_day.png",
             "Recall on an attack family removed from training, per detector. FPR: mean false positive rate on normal "
             "test traffic.", width_in=3.3)
    d.p(refs(C(ctx["text_e3"]), d))
    d.p(ctx["text_e3_rules"])

    # ---------------------------------------------------------------- E4 + E5
    d.h2("E4 and E5: Attack Intensity and Window Length")
    d.figure(FIG / "fig_intensity.png", f"Recall of {ctx['best']} by attack intensity on the test run.", width_in=3.3)
    d.p(refs(ctx["text_e4"], d))
    d.figure(FIG / "fig_window.png", "Effect of the flow-window length on macro F1-score and mean time-to-detect "
             f"({ctx['best']}, test run).", width_in=3.3)
    d.p(refs(ctx["text_e5"], d))

    # ---------------------------------------------------------------- XAI
    d.h2("Explainability")
    d.figure(FIG / "fig_shap.png", "Top 15 features by mean absolute SHAP value for XGBoost, coloured by group.",
             width_in=3.3)
    d.p(refs(C(ctx["text_xai"]), d))
    d.figure(FIG / "fig_twin_residual.png",
             "Monitor and wearable heart rate for one patient: (a) a benign clinical episode, where both sensors move "
             "together, and (b) a data-alteration attack, where only the monitor's reported value changes.",
             width_in=3.3)

    # ---------------------------------------------------------------- E6
    d.h2("E6: Real-time Detection and Risk Scoring")
    d.p(refs(ctx["text_e6"], d))
    lc, rc = ctx["live_c"], ctx["replay_c"]
    rows = []
    for c in CLS[1:]:
        r = rc[rc["attack"] == c]
        l = lc[lc["attack"] == c]
        rows.append([NAMES[c], f"{int(r['detected'].sum())}/{len(r)}",
                     f"{r['delay_s'].median():.1f}" if r["delay_s"].notna().any() else "n/a",
                     f"{int(l['detected'].sum())}/{len(l)}",
                     f"{l['delay_s'].median():.1f}" if l["delay_s"].notna().any() else "n/a"])
    d.table("Streaming Detection per Attack Family",
            ["Attack", "Replay detected", "Replay median s", "Live detected", "Live median s"], rows,
            [1.05, 0.6, 0.6, 0.55, 0.6], size=6.5,
            footnote="Replay: the 3 h test log streamed window by window. Live: a new twin (seed 99, 60 min). "
                     "Time-to-detect runs from campaign start to the end of the first alerting window.")
    d.figure(FIG / "fig_realtime.png", "Risk of the four highest-risk devices during the live run. "
             + ctx["rt_fig_note"], width_in=3.3)
    d.figure(FIG / "dash_live_crop.png", "Live Ward Monitor during the live run: an infusion pump stopped by a rogue "
             "command (High), the attacking workstation (High) and the targeted gateway (Critical).", width_in=3.3)

    # ---------------------------------------------------------------- discussion
    d.h2("Discussion, Implications and Limitations")
    d.p(C(ctx["text_discussion"]))
    d.p(ctx["text_limits"])


def conclusion(d, C, ctx):
    d.h1("Conclusion and Future Work")
    d.p(ctx["text_conclusion"])
    d.p(C(ctx["text_future"]))
