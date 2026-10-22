"""Appendix of the final report (not counted in the 8-page limit)."""
import json

from common import FIG, MEMBERS, PROJ, f4

MODELS = ["Random Forest", "XGBoost", "MLP"]
NAMES = {"dos_flood": "DoS flood", "spoofing": "Spoofing", "data_alteration": "Data alteration",
         "unauthorised_access": "Unauth. access", "reconnaissance": "Reconnaissance", "exfiltration": "Exfiltration"}


def appendix(d, C, ctx):
    comp, params = ctx["comp"], ctx["params"]
    d.h5("Appendix A: Task Allocation and Contributions")
    d.p("Replace the placeholders with each member's name and agreed contribution share before submission.")
    rows = [
        [MEMBERS[0]["name"], "Project lead; patient physiology and device layer of the twin; report integration", "20%"],
        [MEMBERS[1]["name"], "Network layer and attack campaign generator; dataset validation", "20%"],
        [MEMBERS[2]["name"], "Parser, cleaner and feature engineering; WUSTL-EHMS-2020 benchmark script", "20%"],
        [MEMBERS[3]["name"], "Models, tuning, evaluation protocol, experiments E1 to E5, SHAP", "20%"],
        [MEMBERS[4]["name"], "Real-time detector, risk engine, dashboard, user manual", "20%"],
    ]
    d.table("Task Allocation and Contribution", ["Member", "Main tasks", "Share"], rows, [1.25, 1.75, 0.4], size=6.5,
            align=["left", "left", "center"],
            footnote="All members contributed to the literature review, slides, presentation and proofreading.")

    d.h5("Appendix B: Reflections")
    d.p("**Group reflection.** Building our own data source changed how we thought about evaluation. Our first "
        "models scored almost perfectly, which made us suspicious rather than satisfied, and checking the errors "
        "showed two parser problems: flows were oriented by port number, which flipped some scan probes, and an "
        "over-strict plausibility filter deleted extreme vital signs that were in fact evidence of tampering. "
        "Fixing both, and adding benign look-alike events, gave results we trust more. We also learnt to separate "
        "model selection from testing, and that the best model on the binary task is not always the best on rare "
        "classes.")
    for m in MEMBERS:
        d.p(f"**{m['name']}.** [Individual reflection: what you did, what you learnt, what you would do "
            "differently.]")

    d.h5("Appendix C: Additional Results")
    d.figure(FIG / "fig_confusion.png", "Row-normalised confusion matrices of the three models on the independent test "
             "run.", width_in=6.6)
    d.figure(FIG / "fig_per_class.png", "Per-class F1-score of each model on the independent test run.", width_in=3.3)
    d.figure(FIG / "fig_class_distribution.png", "Flow windows per class in the training and test runs (log scale).",
             width_in=3.3)
    rows = []
    for m in ["Random Forest", "XGBoost", "MLP"]:
        r = comp.loc[m]
        rows.append([m, f"{r['cv_macro_f1']:.3f} ± {r['cv_macro_f1_sd']:.3f}", f"{r['cv_f1']:.4f} ± {r['cv_f1_sd']:.4f}",
                     f4(r["fpr"]), f"{r['roc_auc']:.5f}", f"{r['infer_us_per_flow']:.1f}", f"{r['size_kb'] / 1024:.2f}"])
    d.table("Cross-validation and Additional Test Metrics",
            ["Model", "CV macro F1", "CV binary F1", "Test FPR", "Test AUC", "µs/flow", "MB"], rows,
            [0.6, 0.62, 0.62, 0.42, 0.45, 0.35, 0.3], size=6)
    rows = [[m, ", ".join(f"{k}={v}" for k, v in params.get(m, {}).items())] for m in ["Random Forest", "XGBoost", "MLP"]]
    d.table("Selected Hyperparameters (Temporal Hold-out Search)", ["Model", "Setting"], rows, [0.9, 2.5], size=6.5,
            align=["left", "left"])

    ex = ctx["extra"]
    rows = []
    for seed, g in ex.groupby("seed"):
        g = g.set_index("model")
        rows.append([str(seed), f"{g['start_hour'].iloc[0]:02.0f}:00", f"{g['flows'].iloc[0]:,}"]
                    + [f"{g.loc[m, 'macro_f1']:.3f}" for m in MODELS] + [f"{g.loc[m, 'f1']:.4f}" for m in MODELS])
    d.table("Saved Models on Three Extra Unseen Wards (1 h Each)",
            ["Seed", "Start", "Flows", "RF macro F1", "XGB macro F1", "MLP macro F1", "RF F1", "XGB F1", "MLP F1"],
            rows, [0.4, 0.4, 0.5, 0.62, 0.62, 0.62, 0.5, 0.5, 0.5], size=6)

    it = ctx["it"].pivot_table(index=["attack", "intensity"], columns="model", values="recall")
    rows = []
    for att in NAMES:
        for lv in ("low", "medium", "high"):
            if (att, lv) in it.index:
                rows.append([NAMES[att], lv] + [f"{it.loc[(att, lv), m]:.3f}" for m in MODELS])
            else:
                rows.append([NAMES[att], lv, "no campaign", "", ""])
    d.table("Recall by Attack Family and Intensity on the Test Run", ["Attack", "Intensity", "RF", "XGBoost", "MLP"],
            rows, [1.0, 0.6, 0.55, 0.55, 0.55], size=6)

    tr = json.loads((PROJ / "models" / "twin_rules.json").read_text())["thresholds"]
    rows = [
        ["R1", "Sender is not in the asset registry", "none (any)"],
        ["R2", "MAC address or TTL differs from the registry", "none (any)"],
        ["R3", "Duplicated or missing sequence numbers in a window",
         f"duplicates > {tr['SeqDup']:.0f}, gaps > {tr['SeqGap']:.0f}"],
        ["R4", "Monitor and wearable on one patient disagree",
         f"heart rate > {tr['TwinResidualHR']:.1f} bpm, SpO2 > {tr['TwinResidualSpO2']:.1f} points"],
        ["R5", "Relative pump-rate change", f"> {tr['PumpRateDelta']:.0f} (any change)"],
    ]
    d.table("Twin Consistency Rules and Thresholds (99.9th Percentile of Normal Training Traffic)",
            ["Rule", "Flags a flow when", "Threshold"], rows, [0.4, 2.4, 2.0], size=6.5,
            align=["center", "left", "left"])
    sens = ctx["sens"]
    rows = [[f"{100 * r['quantile']:.2f}th", f"{100 * r['fpr']:.2f}%"] + [f"{r[a]:.2f}" for a in NAMES]
            for _, r in sens.iterrows()]
    d.table("Twin Rules: Recall on Each Unseen Family and False Positive Rate by Threshold Percentile",
            ["Percentile", "FPR"] + [NAMES[a] for a in NAMES], rows, [0.6, 0.45] + [0.62] * 6, size=6)

    d.h5("Appendix D: Dashboard")
    d.figure(FIG / "dash_model_evaluation.png", "Dashboard: model evaluation page.", width_in=4.6)
    d.figure(FIG / "dash_model_robustness.png", "Dashboard: robustness of the saved models on three extra wards and "
             "bootstrap intervals.", width_in=4.6)
    d.figure(FIG / "dash_zero_day.png", "Dashboard: zero-day results with the Isolation Forest, twin rules and "
             "hybrids.", width_in=4.6)
    d.figure(FIG / "dash_experiments.png", "Dashboard: experiments page (ablation).", width_in=4.6)

    d.h5("Appendix E: Screenshots of Programming Results")
    d.p("Console output of the main programs, captured by running the command in each title bar. Long logs are "
        "shortened where marked; nothing else is edited.")
    d.figure(FIG / "term_simulate.png", "Dataset generation: for the training and test runs the twin reports its seed, "
             "length and start time, the packets and flow windows produced, and the flow windows per class.",
             width_in=6.0)
    d.figure(FIG / "term_train.png", "Model training: temporal hold-out tuning, time-blocked cross-validation, the "
             "single test on the independent run and the selection of XGBoost.", width_in=6.0)
    d.figure(FIG / "term_realtime.png", "Real-time detection on a new live twin: colour-coded alerts with ward time, "
             "attack class, device, confidence, risk score and the three most abnormal features, then the run "
             "summary.", width_in=6.6)
    d.figure(FIG / "term_pytest.png", f"Automated tests: all {ctx['n_tests']} tests pass.", width_in=6.0)

    d.h5("Appendix F: Source Code and Reproduction")
    d.p("The submission contains the MedTwinIDS source code, generated datasets, trained models, result files and a "
        "user manual. The full pipeline is reproduced with: `python -m medtwin.simulate --preset all`, "
        "`python -m medtwin.train --tune`, `python -m medtwin.experiments`, `python -m medtwin.realtime --replay test "
        "--minutes 180 --speed 0 --quiet --save-as replay`, `python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 "
        "--start-hour 14 --quiet --save-as live` (and the same command with `--twin-rules --save-as live_rules`), "
        "`python -m medtwin.figures` and `streamlit run dashboard/app.py`; `python -m pytest -v` runs the tests. The "
        "optional `python -m medtwin.benchmark_wustl` script evaluates the same models on WUSTL-EHMS-2020.")
