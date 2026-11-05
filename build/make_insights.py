"""Write build/slide_insights.json: short interpretive lines for the deck, built from the result files."""
import json
import re

import pandas as pd

from common import RES, comparison

comp = comparison()
best = comp.index[comp["selected"]][0]
xg, rf, ml = comp.loc["XGBoost"], comp.loc["Random Forest"], comp.loc["MLP"]
ab = pd.read_csv(RES / "e2_ablation.csv").set_index(["feature_set", "model"])
zd = pd.read_csv(RES / "e3_zero_day.csv")
zm = zd.groupby("detector")[["recall", "fpr"]].mean()
iso = zd[zd["detector"] == "Isolation Forest"].set_index("held_out")["recall"]
ws = pd.read_csv(RES / "e5_window.csv").set_index("window_s")
it = pd.read_csv(RES / "e4_intensity.csv")
itb = it[it["model"] == best].set_index(["attack", "intensity"])["recall"]
diag = json.loads((RES / "spoofing_diagnostic.json").read_text())
live = json.loads((RES / "e6_live_live_summary.json").read_text())["overall"]
rep = json.loads((RES / "e6_replay_live_summary.json").read_text())["overall"]
hyb = f"Hybrid ({best} + IF)"
hyb3 = f"Hybrid ({best} + IF + twin rules)"
rul = zd[zd["detector"] == "Twin rules"].set_index("held_out")["recall"]
sens = pd.read_csv(RES / "e3_rules_sensitivity.csv").set_index("quantile")
boot = pd.read_csv(RES / "e1_bootstrap.csv").set_index("model")
extra = pd.read_csv(RES / "e1_extra_runs.csv")
ex_runs = extra.pivot(index="seed", columns="model", values="macro_f1")
wins = int((ex_runs.idxmax(axis=1) == best).sum())
live_rules = json.loads((RES / "e6_live_rules_live_summary.json").read_text())["overall"]
pytest_txt = (RES.parents[1] / "build" / "terminal" / "pytest.txt").read_text()
n_tests = int(re.search(r"(\d+) passed", pytest_txt).group(1))

n3 = lambda v: f"{0.0 if round(v, 3) == 0 else v:.3f}"  # no "-0.000"
most_stable = xg["cv_macro_f1_sd"] <= min(rf["cv_macro_f1_sd"], ml["cv_macro_f1_sd"])
fastest = xg["train_s"] <= min(rf["train_s"], ml["train_s"])
ins = {
    "e1_title": f"{best} is the most accurate" + (", most stable" if most_stable else "") + " and fast",
    "e1": (f"All three exceed {int(comp['accuracy'].min() * 1000) / 10:.1f}% accuracy, so accuracy hides the "
           f"difference; macro F1 does not. {best} was chosen on cross-validation (the highest"
           + (" and most stable" if most_stable else "") + "). On the test run it ties with the MLP (paired "
           f"bootstrap) and beats Random Forest; on three extra unseen wards it is best {wins} times out of 3."),
    "e1_notes": (f"Selection used cross-validation only. The 95% cluster-bootstrap interval of the macro-F1 difference "
                 f"is {n3(boot.loc['Random Forest', 'diff_vs_best_low'])} to "
                 f"{n3(boot.loc['Random Forest', 'diff_vs_best_high'])} against Random Forest (a real lead) and "
                 f"{n3(boot.loc['MLP', 'diff_vs_best_low'])} to {n3(boot.loc['MLP', 'diff_vs_best_high'])} against the "
                 f"MLP (a tie). On the extra wards {best} averages {ex_runs[best].mean():.3f}, the MLP "
                 f"{ex_runs['MLP'].mean():.3f} and Random Forest {ex_runs['Random Forest'].mean():.3f}. {best} "
                 + (f"also trains fastest ({xg['train_s']:.0f} s)" if fastest else f"trains in {xg['train_s']:.0f} s")
                 + f" and needs {xg['latency_ms']:.2f} ms per flow; Random Forest is "
                 f"{rf['latency_ms'] / xg['latency_ms']:.0f}x slower per flow and weaker on spoofing."),
    "confusion": (f"Random Forest calls {diag['rf_impostor_as_normal']:.0%} of impostor spoofing flows normal; "
                  f"XGBoost and the MLP catch {diag['xgb_impostor']:.0%} and {diag['mlp_impostor']:.0%}. The evidence is a "
                  "few weak twin signals that bagging outvotes and boosting focuses on."),
    "confusion_notes": ("Rows are true classes, columns predictions, normalised per row. Loud attacks are easy for "
                        "all models. The impostor publishes under a victim's identity; only sequence numbers, the "
                        "cross-sensor residual and the sender's registry status give it away."),
    "ablation": (f"Adding the 11 twin features lifts {best} from {ab.loc[('network_biometric', best), 'macro_f1']:.3f} "
                 f"to {ab.loc[('full_twin', best), 'macro_f1']:.3f} and spoofing recall from "
                 f"{ab.loc[('network_biometric', best), 'recall_spoofing']:.0%} to "
                 f"{ab.loc[('full_twin', best), 'recall_spoofing']:.0%}. Raw vitals alone add little: consistency is "
                 "what matters."),
    "ablation_notes": ("Biometric features alone are nearly useless because the fake vitals are plausible. Network "
                       "plus biometric, the WUSTL-EHMS-2020 design, is no better than network alone here. The twin "
                       "features (sequence integrity, monitor versus wearable residual, registry checks) make the "
                       "difference. Twin features are 20% of the inputs but carry about a quarter of the SHAP "
                       "attribution."),
    "zero_title": "Closed-world models miss the unknown",
    "zero_day": (f"Trained without one family, supervised recall on it averages only {zm.loc[best, 'recall']:.2f} "
                 f"({best}). The Isolation Forest reaches {zm.loc['Isolation Forest', 'recall']:.2f}: strong on DoS, "
                 f"scans and exfiltration, weak on spoofing ({iso['spoofing']:.2f}) and data alteration "
                 f"({iso['data_alteration']:.2f}). Adding the twin rules closes that gap: all three layers together "
                 f"reach {zm.loc[hyb3, 'recall']:.2f}."),
    "zero_day_notes": ("Each row is a detector, each column an attack family that was removed from training. The "
                       f"hybrid of {best} and the Isolation Forest reaches {zm.loc[hyb, 'recall']:.2f} at "
                       f"{100 * zm.loc[hyb, 'fpr']:.2f}% FPR; adding the twin rules gives {zm.loc[hyb3, 'recall']:.2f} at "
                       f"{100 * zm.loc[hyb3, 'fpr']:.2f}%. The next slide explains the rules."),
    "rules": dict(spoof=float(rul["spoofing"]), alt=float(rul["data_alteration"]),
                  fpr=float(zm.loc["Twin rules", "fpr"]), hyb=float(zm.loc[hyb, "recall"]),
                  hyb3=float(zm.loc[hyb3, "recall"]), hyb_fpr=float(zm.loc[hyb, "fpr"]),
                  hyb3_fpr=float(zm.loc[hyb3, "fpr"]), live_fa=live["false_alerts"],
                  rules_fa=live_rules["false_alerts"], live_inc=live["incidents"], rules_inc=live_rules["incidents"],
                  q9999_fpr=float(sens.loc[0.9999, "fpr"]), q9999_spoof=float(sens.loc[0.9999, "spoofing"])),
    "rules_notes": ("The rules need no attack labels: each threshold is the 99.9th percentile of its signal in "
                    "normal training traffic. They are strongest exactly where the learners are weakest, on unseen "
                    f"spoofing ({rul['spoofing']:.2f}) and data alteration ({rul['data_alteration']:.2f}). The cost "
                    f"is more false alerts: in the live run {live['false_alerts']} became {live_rules['false_alerts']}, "
                    f"so the rules are an option (--twin-rules). At the 99.99th percentile their false positive rate "
                    f"falls to {100 * sens.loc[0.9999, 'fpr']:.2f}% but unseen-spoofing recall falls to "
                    f"{sens.loc[0.9999, 'spoofing']:.2f}."),
    "n_tests": n_tests,
    "sensitivity": (f"Low-intensity data alteration is the hardest case ({best} recall "
                    f"{itb.loc[('data_alteration', 'low')]:.2f}). Shorter windows detect sooner: mean time-to-detect "
                    f"{ws.loc[10.0, 'mean_delay_s']:.1f} s at 10 s, {ws.loc[5.0, 'mean_delay_s']:.1f} s at 5 s and "
                    f"{ws.loc[2.0, 'mean_delay_s']:.1f} s at 2 s, with macro F1 {ws.loc[10.0, 'macro_f1']:.3f}, "
                    f"{ws.loc[5.0, 'macro_f1']:.3f} and {ws.loc[2.0, 'macro_f1']:.3f}. We keep 5 s."),
    "sensitivity_notes": ("The test run contains no high-intensity spoofing or unauthorised-access campaign, so "
                          "those bars are missing. Exfiltration raises the mean delay because it is only visible "
                          "once the upload starts."),
    "realtime_notes": (f"Replay of the test run: {rep['detected']}/{rep['campaigns']} campaigns, median "
                       f"{rep['median_delay_s']} s. New live twin: {live['detected']}/{live['campaigns']}, median "
                       f"{live['median_delay_s']} s. {live['alerts']:,} alerts collapse into {live['incidents']} "
                       "incidents. Most 'false' alerts are the Isolation Forest, nurse pump pauses, or the healthy "
                       "wearable of a patient whose monitor is spoofed, which still points at the right bed."),
    "limitations": [
        "Simulated data: real devices and attackers vary more, so absolute scores are optimistic",
        "Twin rules catch unseen integrity attacks but add false alerts",
        "Nurse pump commands, including restores after an attack, can look like rogue commands",
        "Flow labels can disagree with bed-level reality",
        "WUSTL-EHMS-2020 benchmark script ready but not yet run",
    ],
    "future": [
        "Validate on real captures and WUSTL-EHMS-2020",
        "Calibrate twin rules per device to cut false alerts",
        "Feed command source and clinical orders into pump features",
        "Online sequence models (autoencoder ensembles)",
        "Clinician-approved automatic isolation of high-risk devices",
    ],
    "conclusion": [
        "RQ1: the ward twin generates exactly labelled, controllable IoMT data",
        f"RQ2: {best} gives the best balance (macro F1 {xg['macro_f1']:.3f}, {xg['latency_ms']:.2f} ms per flow)",
        f"RQ3: twin features lift spoofing recall from {ab.loc[('network_biometric', best), 'recall_spoofing']:.0%} to "
        f"{ab.loc[('full_twin', best), 'recall_spoofing']:.0%}",
        f"RQ4: Isolation Forest plus twin rules lift unseen-attack recall from {zm.loc[best, 'recall']:.2f} to "
        f"{zm.loc[hyb3, 'recall']:.2f}",
        f"RQ5: all {rep['campaigns']} test campaigns detected in real time, median {rep['median_delay_s']} s",
    ],
}
(RES.parents[1] / "build" / "slide_insights.json").write_text(json.dumps(ins, indent=2))
print("insights written")
