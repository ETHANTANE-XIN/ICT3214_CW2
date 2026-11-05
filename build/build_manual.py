"""Build the user manual (.docx)."""
import json
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from common import GROUP_ID, OUT, RES, ROOT, TITLE, comparison, f3, f4, report
from finalise import finalise

TIMES = json.loads((RES / "timings.json").read_text()) if (RES / "timings.json").exists() else {}
N_TESTS = int(re.search(r"(\d+) passed", (ROOT / "build" / "terminal" / "pytest.txt").read_text()).group(1))


def t(key, default):
    return TIMES.get(key, default)


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)

st = doc.styles
st["Normal"].font.name = "Calibri"
st["Normal"].font.size = Pt(10.5)
st["Normal"].paragraph_format.space_after = Pt(6)
for name, size, colour in (("Title", 24, "1F3A5F"), ("Heading 1", 15, "1F3A5F"), ("Heading 2", 12, "2A78D6"),
                           ("Heading 3", 11, "333333")):
    s = st[name]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor.from_string(colour)
    rpr = s.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(a), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]


def shade(cell_or_par, hex_fill):
    el = cell_or_par._tc.get_or_add_tcPr() if hasattr(cell_or_par, "_tc") else cell_or_par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    el.append(shd)


def para(text, style=None, bold_prefix=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    for i, part in enumerate(text.split("`")):
        r = p.add_run(part)
        if i % 2 == 1:
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
    return p


def code(lines):
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        shade(p, "F2F4F7")  # before spacing/indent so the pPr children stay in schema order
        p.paragraph_format.space_after = Pt(0) if i < len(lines) - 1 else Pt(8)
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        rpr = r._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        rf.set(qn("w:hAnsi"), "Consolas")


def bullets(items):
    for it in items:
        para(it, style="List Bullet")


def steps(items):
    for it in items:
        para(it, style="List Number")


def table(header, rows, widths_cm):
    tb = doc.add_table(rows=1 + len(rows), cols=len(header))
    tb.style = "Table Grid"
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = tb.cell(i, j)
            cell.width = Cm(widths_cm[j])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            for k, part in enumerate(str(val).split("`")):
                r = p.add_run(part)
                r.font.size = Pt(9)
                if k % 2 == 1:
                    r.font.name = "Consolas"
                if i == 0:
                    r.bold = True
            if i == 0:
                shade(cell, "DCE8F7")
    doc.add_paragraph()


tr, te = report("train"), report("test")
comp = comparison()
best = comp.index[comp["selected"]][0]

# ------------------------------------------------------------------ cover
p = doc.add_paragraph(style="Title")
p.add_run("MedTwin-IDS User Manual")
para(TITLE)
para(f"ICT3214 Security Analytics, Coursework 2. Group {GROUP_ID}.")
para("This manual explains how to set up the environment, generate the digital-twin datasets, train and evaluate the "
     "three detection models, run every experiment in the report, run real-time detection and use the dashboard. "
     "Following it end to end reproduces all results, tables and figures in the final report.")
table(["Item", "Details"], [
    ["Software", "Python 3.13 (tested on Linux and Windows 11); requirements.txt pins the package versions used "
                 "for the results"],
    ["Hardware", "4 CPU cores and 8 GB RAM recommended; about 1.5 GB free disk space"],
    ["Network access", "Only needed once, to install Python packages. The simulation never touches a real network."],
    ["Total run time", f"About {t('total_min', 45)} minutes for the full pipeline on a 4-core machine"],
], [3.5, 12.5])

doc.add_heading("Contents", level=1)
for line in ["1. Package contents", "2. Installation", "3. Quick start", "4. Step-by-step guide",
             "5. Reproducing the report", "6. Configuration", "7. Troubleshooting", "8. Safety and ethics"]:
    para(line)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ------------------------------------------------------------------ 1
doc.add_heading("1. Package contents", level=1)
para("Unzip the submission. The source code is in the `MedTwinIDS` folder:")
code(["MedTwinIDS/",
      "  medtwin/          Python package (simulator, parser, models, experiments, real-time detector)",
      "  dashboard/app.py  Streamlit dashboard",
      "  data/             generated datasets (packet logs, flow features, ground truth)",
      "  models/           trained models, model card, runtime assets",
      "  results/          metrics tables (CSV) and figures/",
      "  runtime/          files written by the real-time detector for the dashboard",
      "  docs/diagrams/    architecture.puml (PlantUML source) and architecture.png",
      "  tests/            automated tests (pytest)",
      "  requirements.txt  Python dependencies",
      "  README.md         short overview"])
table(["Module", "What it does"], [
    ["`medtwin/config.py`", "Ward layout, addresses, attack catalogue, feature groups, analyst playbook"],
    ["`medtwin/twin.py`", "Digital twin: patient physiology, medical devices, network packet events, capture artefacts"],
    ["`medtwin/attacks.py`", "Schedules and generates the six labelled attack families at three intensities"],
    ["`medtwin/simulate.py`", "Command-line tool that runs the twin and writes datasets"],
    ["`medtwin/parser.py`", "Cleans packet logs and extracts 55 flow-window features"],
    ["`medtwin/models.py`", "Random Forest, XGBoost, MLP and Isolation Forest definitions"],
    ["`medtwin/train.py`", "Tuning, time-blocked cross-validation, independent test, model selection (E1)"],
    ["`medtwin/experiments.py`", "E1b robustness, E2 ablation, E3 zero-day, E4 intensity, E5 window length, SHAP"],
    ["`medtwin/rules.py`", "Label-free twin consistency rules R1 to R5 (thresholds from normal training traffic)"],
    ["`medtwin/realtime.py`", "Real-time detector on the live twin or a replayed packet log"],
    ["`medtwin/risk.py`", "Per-device risk scores, incident grouping and explanations"],
    ["`medtwin/figures.py`", "Creates the report figures in results/figures/"],
    ["`medtwin/benchmark_wustl.py`", "Optional benchmark on the public WUSTL-EHMS-2020 dataset"],
    ["`tests/test_medtwin.py`", "Automated tests of the twin, parser, models, rules, risk engine and real-time detector"],
], [4.5, 11.5])

# ------------------------------------------------------------------ 2
doc.add_heading("2. Installation", level=1)
para("Open a terminal in the `MedTwinIDS` folder and create a virtual environment.")
doc.add_heading("Linux and macOS", level=3)
code(["python3 -m venv .venv", "source .venv/bin/activate", "pip install --upgrade pip",
      "pip install -r requirements.txt"])
doc.add_heading("Windows (PowerShell)", level=3)
code(["py -m venv .venv", ".venv\\Scripts\\Activate.ps1", "python -m pip install --upgrade pip",
      "pip install -r requirements.txt"])
para("Check the installation:")
code(["python -c \"import sklearn, xgboost, streamlit, shap; print('ok')\""])
para("All commands below are run from the `MedTwinIDS` folder with the virtual environment active. On Windows, "
     "type `python` where this manual shows `python`.")

# ------------------------------------------------------------------ 3
doc.add_heading("3. Quick start", level=1)
para("The submission already contains the generated data, trained models and results, so you can open the dashboard "
     "straight away (step 6). To reproduce everything from scratch, run these steps in order:")
table(["Step", "Command", "Time*"], [
    ["1", "`python -m medtwin.simulate --preset all`", t("simulate", "2 min")],
    ["2", "`python -m medtwin.train --tune`", t("train", "11 min")],
    ["3", "`python -m medtwin.experiments`", t("experiments", "20 min")],
    ["4a", "`python -m medtwin.realtime --replay test --minutes 180 --speed 0 --quiet --save-as replay`", t("replay", "9 min")],
    ["4b", "`python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --quiet --save-as live`", t("realtime", "3 min")],
    ["4c", "`python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --twin-rules --quiet "
           "--save-as live_rules`", t("realtime_rules", "4 min")],
    ["5", "`python -m medtwin.figures`", t("figures", "1 min")],
    ["6", "`streamlit run dashboard/app.py`", "opens in browser"],
    ["7", "`python -m medtwin.realtime --minutes 20 --speed 10` (second terminal)", "2 min"],
    ["8", "`python -m pytest -v` (automated tests, optional)", t("tests", "1.5 min")],
], [1.2, 12.0, 2.8])
para("*Measured on a 4-core Linux machine. Results are deterministic: the same seeds give the same datasets, and the "
     "models use fixed random states, so metrics should match the report to the reported precision (tiny differences "
     "can appear across library versions).")

# ------------------------------------------------------------------ 4
doc.add_heading("4. Step-by-step guide", level=1)
doc.add_heading("4.1 Generate the datasets (digital twin)", level=2)
para("The simulator builds the ward twin (8 beds, 24 medical devices, gateway, nurse station, EHR server, update "
     "server and two workstations), runs it, injects attack campaigns, writes the raw packet log and then parses it "
     "into flow features.")
code(["python -m medtwin.simulate --preset all"])
para("This creates two independent runs:")
table(["Run", "Seed", "Length", "Starts", "Packets", "Flow windows", "Attack flows"], [
    ["train", "7", f"{tr['hours']:.0f} h", "08:00", f"{tr['packets']:,}", f"{tr['flows']:,}", f"{tr['attack_flows']:,}"],
    ["test", "2026", f"{te['hours']:.0f} h", "20:00", f"{te['packets']:,}", f"{te['flows']:,}", f"{te['attack_flows']:,}"],
], [1.6, 1.3, 1.5, 1.5, 3.0, 3.0, 2.5])
para("Output files in `data/` (shown for the train run; the test run is the same with `test_`):")
table(["File", "Content"], [
    ["`devices.csv`", "Asset registry of the twin: device ID, type, IP, MAC, bed, expected TTL, criticality"],
    ["`train_packets.parquet`", "Raw packet log (one row per packet, with ground-truth labels)"],
    ["`train_flows.csv`", "Cleaned flow-window dataset: metadata, 55 features, label, attack class, campaign"],
    ["`train_campaigns.csv`", "Ground truth for each attack campaign: type, intensity, start, end, source, target"],
    ["`train_benign_events.csv`", "Clinical episodes and benign network events (congestion, updates, reboots)"],
    ["`train_report.json`", "Cleaning report and class counts"],
], [5.0, 11.0])
para("Custom runs: choose a name, seed, length and attack rate, for example:")
code(["python -m medtwin.simulate --name demo --hours 0.5 --seed 11 --campaigns-per-hour 12"])
table(["Option", "Meaning", "Default"], [
    ["`--preset all|train|test`", "Use the configuration from the report", "none"],
    ["`--name`", "Prefix of the output files", "custom"],
    ["`--seed`", "Random seed (patients, devices, attack timing)", "11"],
    ["`--hours`", "Simulated ward time", "1.0"],
    ["`--start-hour`", "Clock time at the start of the run", "8.0"],
    ["`--campaigns-per-hour`", "Attack campaigns per simulated hour", "10"],
    ["`--window`", "Flow window length in seconds", "5"],
    ["`--no-packets`", "Do not save the raw packet log", "off"],
], [4.5, 8.5, 3.0])

doc.add_heading("4.2 Train, tune and evaluate the three models (E1)", level=2)
code(["python -m medtwin.train --tune"])
steps([
    "Tunes each model on a temporal hold-out (about the first 75% of the training run fits and the rest validates; "
    "the cut never splits a campaign). "
    "Without `--tune`, the saved settings in `results/best_params.json` are reused.",
    "Runs five-fold cross-validation with 30-minute time blocks as groups (each campaign stays in the block where "
    "it starts).",
    "Trains each model on the full training run and tests it once on the independent test run.",
    "Selects the deployed model on cross-validation macro F1-score (near-ties broken by latency) and saves it.",
    "Trains the Isolation Forest on normal training traffic only.",
])
table(["Output", "Content"], [
    ["`results/model_comparison.csv`", "Accuracy, precision, recall, F1, FPR, ROC AUC, macro F1, timing, size, CV scores"],
    ["`results/cv_folds.csv`", "Per-fold cross-validation scores"],
    ["`results/tuning_log.csv`, `best_params.json`", "Hyperparameter search log and chosen settings"],
    ["`results/confusion_*.csv`, `per_class_*.csv`", "Confusion matrices and per-class scores for each model"],
    ["`results/test_predictions.csv`", "Test-set predictions and attack probabilities of every model"],
    ["`models/*.joblib`", "Trained models; `best_model.joblib` is the deployed one"],
    ["`models/model_card.json`", "Selected model, features, classes, window, threshold, test metrics"],
], [6.0, 10.0])
para(f"Expected result: {best} is selected, with test macro F1-score {f3(comp.loc[best, 'macro_f1'])} and binary "
     f"F1-score {f4(comp.loc[best, 'f1'])}.")

doc.add_heading("4.3 Run the experiments (E1b to E5 and explainability)", level=2)
code(["python -m medtwin.experiments                 # all experiments",
      "python -m medtwin.experiments --only e2,e3   # a subset"])
table(["Code", "Experiment", "Output"], [
    ["assets", "Normal-traffic statistics, Isolation Forest thresholds and twin-rule thresholds for the live detector",
     "`models/normal_stats.csv`, `iforest_threshold.json`, `twin_rules.json`"],
    ["robust", "E1b: 95% bootstrap intervals of macro F1 and of the paired differences between models, and the saved "
               "models scored on three extra unseen wards (seeds 3001 to 3003)",
     "`results/e1_bootstrap.csv`, `e1_extra_runs.csv`"],
    ["e2", "Feature-set ablation: network, biometric, network+biometric, full twin", "`results/e2_ablation.csv`"],
    ["e3", "Zero-day: leave one attack family out of training; Isolation Forest, twin rules and hybrids; rule "
           "threshold sensitivity", "`results/e3_zero_day.csv`, `e3_rules_sensitivity.csv`"],
    ["e4", "Recall by attack intensity", "`results/e4_intensity.csv`"],
    ["e5", "Window length 2, 5 and 10 s: accuracy versus time-to-detect", "`results/e5_window.csv`, `e5_delays.csv`"],
    ["xai", "SHAP (XGBoost) and impurity importance (Random Forest)", "`results/feature_importance.csv`, `shap_per_class.csv`"],
    ["diag", "Spoofing diagnostic: share of impostor flows each model labels normal", "`results/spoofing_diagnostic.json`"],
], [1.6, 8.4, 6.0])

doc.add_heading("4.4 Real-time detection (E6)", level=2)
para("The real-time detector runs the twin live (or replays a stored packet log), extracts features every 5 s "
     "window, scores each flow with the deployed model and the Isolation Forest, updates per-device risk scores and "
     "writes alerts. Two modes are available:")
code(["# E6 evaluation runs used in the report (as fast as possible, no console output)",
      "python -m medtwin.realtime --replay test --minutes 180 --speed 0 --quiet --save-as replay",
      "python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --quiet --save-as live",
      "python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --twin-rules --quiet \\",
      "       --save-as live_rules",
      "",
      "# live demonstration (10 times faster than real time, colour alerts in the console)",
      "python -m medtwin.realtime --minutes 20 --speed 10",
      "",
      "# replay the independent test run's packet log",
      "python -m medtwin.realtime --replay test --minutes 30 --speed 20"])
table(["Option", "Meaning", "Default"], [
    ["`--minutes`", "Minutes of ward time to simulate or replay (sets the attack schedule)", "20"],
    ["`--speed`", "Time acceleration: 1 = real time, 10 = ten times faster, 0 = as fast as possible", "10"],
    ["`--seed`", "Seed of the live twin (use a seed not used for training)", "99"],
    ["`--campaigns-per-hour`", "Attack rate in the live twin", "12"],
    ["`--replay train|test`", "Replay a stored packet log instead of the live twin", "off"],
    ["`--no-hybrid`", "Disable the Isolation Forest safety net", "on"],
    ["`--twin-rules`", "Also raise alerts from the twin consistency rules (needs `models/twin_rules.json` from step 3)", "off"],
    ["`--stop-at`", "Stop after this many minutes but keep the --minutes attack schedule", "off"],
    ["`--save-as NAME`", "Also copy the run's outputs to `results/e6_NAME_*` (kept for the report)", "off"],
    ["`--quiet`", "Do not print alerts", "off"],
], [4.0, 9.5, 2.5])
para("Each console alert shows the ward clock, predicted attack, device, confidence, the device's new risk score and "
     "the three most abnormal features. Files written to `runtime/`:")
table(["File", "Content"], [
    ["`state.json`", "Latest KPIs, device risks, vitals and incidents (read by the dashboard every 2 s)"],
    ["`alerts.jsonl`", "Every alert, one JSON object per line"],
    ["`timeline.csv`", "Risk score of every device for every window"],
    ["`ground_truth.csv`", "Attack campaigns in the run (for evaluation only; never shown to the model)"],
    ["`live_summary.json`, `live_campaigns.csv`", "Detection rate, time-to-detect per campaign, latency"],
], [5.5, 10.5])
para("Risk levels: Low (below 30), Medium (30 to 59), High (60 to 79) and Critical (80 and above).")

doc.add_heading("4.5 Figures", level=2)
code(["python -m medtwin.figures"])
para("Writes all report figures as 300 dpi PNG files to `results/figures/`. The architecture diagram is compiled "
     "from `docs/diagrams/architecture.puml` with PlantUML: `java -jar plantuml.jar -tpng architecture.puml`.")

doc.add_heading("4.6 Dashboard", level=2)
code(["streamlit run dashboard/app.py"])
para("Streamlit prints a local address (normally http://localhost:8501) and opens it in the browser. Start the live "
     "detector (section 4.4, demonstration mode) in a second terminal to see the Live Ward Monitor update.")
table(["Page", "What it shows"], [
    ["Live Ward Monitor", "Ward clock, KPIs, a map of all beds with device tiles coloured by risk level and the latest "
                          "vitals, grouped incidents with explanations and recommended actions, risk timeline and the "
                          "hidden ground truth (for demonstration)"],
    ["Model Evaluation", "Comparison table, quality and cost charts, confusion matrices, ROC curves, per-class F1, "
                         "cross-validation spread, extra-ward results and bootstrap intervals"],
    ["Experiments", "Ablation, zero-day heatmap with twin rules, rule threshold sensitivity, intensity, window length "
                    "and real-time summary"],
    ["Dataset Explorer", "Run statistics, class counts, campaign timeline and feature distributions by class"],
    ["Explainability", "Global SHAP importance, importance by feature group and per-class SHAP heatmap"],
], [3.5, 12.5])
para("Suggested live demonstration (about 4 minutes):")
bullets([
    "Start the dashboard (`streamlit run dashboard/app.py`), then the live detector in a second terminal "
    "(`python -m medtwin.realtime --minutes 20 --speed 10`).",
    "On Live Ward Monitor, watch the ward clock and KPIs; when an attack starts, point out the device tiles that "
    "change colour and the risk timeline.",
    "Open the newest incident: show its class, confidence, the three most abnormal features and the recommended action.",
    "Switch to Model Evaluation (comparison table, extra wards, bootstrap intervals), then Experiments (ablation, "
    "zero-day heatmap with twin rules) and Explainability (SHAP).",
    "Optionally restart the detector with `--twin-rules` to show twin-rule alerts.",
])

doc.add_heading("4.7 Optional: benchmark on WUSTL-EHMS-2020", level=2)
para("Download `wustl-ehms-2020.csv` from https://www.cse.wustl.edu/~jain/ehms/index.html, copy it to `data/` and run:")
code(["python -m medtwin.benchmark_wustl --csv data/wustl-ehms-2020.csv"])
para("The script removes address, port, MAC and packet-number columns (they identify hosts rather than behaviour), "
     "evaluates the three models with five-fold stratified cross-validation on network-only, biometric-only and "
     "combined features, and writes `results/wustl_benchmark.csv` for comparison with Hady et al. (2020).")

doc.add_heading("4.8 Automated tests", level=2)
code(["python -m pytest -v"])
para(f"The suite has {N_TESTS} tests and takes about {t('tests', '1.5 min')}. It checks that the twin is "
     "deterministic and labels every attack family, that short runs keep campaigns inside the run, that the cleaner "
     "removes capture artefacts but keeps extreme vital signs, that flows are oriented by the SYN, that no identifier "
     "is used as a feature, that window-by-window extraction gives exactly the same 55 features as offline "
     "extraction, that a rogue pump command raises the pump-rate feature, that each twin rule fires only on its "
     "violation, that every model trains on a subset of classes, that the risk engine protects impersonated devices "
     "and tracks rogue hosts, that saved models load on any platform, that cross-validation never splits a campaign, "
     "that forged frames match benign sizes, that attacked pumps are restored, that explanations cite identity "
     "evidence, that the benchmark loader matches the WUSTL-EHMS-2020 schema and that the real-time "
     "detector runs end to end. Tests write only to temporary folders, so they never change the submitted results. "
     "The last test needs the trained models (step 2) and is skipped without them.")

# ------------------------------------------------------------------ 5
doc.add_heading("5. Reproducing the report", level=1)
table(["Report item", "Produced by", "Source file"], [
    ["Dataset summary table", "step 1", "`data/*_report.json`"],
    ["Model comparison table, per-class table, confusion figure", "step 2 + 5", "`results/model_comparison.csv`, `per_class_*.csv`, `confusion_*.csv`"],
    ["Bootstrap and extra-run rows of the comparison table", "step 3", "`results/e1_bootstrap.csv`, `e1_extra_runs.csv`"],
    ["Ablation figure", "step 3 + 5", "`results/e2_ablation.csv`"],
    ["Zero-day figure and twin-rules paragraph", "steps 3, 4c + 5", "`results/e3_zero_day.csv`, `e3_rules_sensitivity.csv`, `e6_live_rules_*`"],
    ["Intensity and window figures", "step 3 + 5", "`results/e4_intensity.csv`, `e5_window.csv`"],
    ["SHAP figure", "step 3 + 5", "`results/feature_importance.csv`"],
    ["Real-time figure and table", "step 4 + 5", "`results/e6_live_*`, `results/e6_replay_*`"],
    ["Dashboard screenshots", "steps 6 + 7", "browser"],
    ["Console screenshots and test count (appendix)", "steps 1, 2, 7 + 8", "console output"],
], [6.0, 2.5, 7.5])

# ------------------------------------------------------------------ 6
doc.add_heading("6. Configuration", level=1)
para("Edit `medtwin/config.py` to change the scenario, then rerun from step 1.")
table(["Setting", "Meaning"], [
    ["`N_BEDS`", "Number of beds (each adds a monitor, pump and wearable)"],
    ["`WINDOW_SECONDS`", "Flow window length used for training and live detection"],
    ["`PUBLISH_PERIOD`", "Telemetry period per device type"],
    ["`CRITICALITY`, `SEVERITY`", "Weights used by the risk engine"],
    ["`PLAYBOOK`", "Recommended actions shown with alerts"],
    ["`ScenarioConfig`", "Default rates of attack campaigns, clinical episodes, congestion, updates and reboots"],
], [4.5, 11.5])
para("Attack intensities are set in `INTENSITY_PARAMS` in `medtwin/attacks.py`.")

# ------------------------------------------------------------------ 7
doc.add_heading("7. Troubleshooting", level=1)
table(["Problem", "Fix"], [
    ["`ModuleNotFoundError: medtwin`", "Run commands from the `MedTwinIDS` folder, not from inside `medtwin/`."],
    ["`FileNotFoundError: ..._flows.csv`", "Run step 1 (simulate) first."],
    ["`model_card.json` not found", "Run step 2 (train) before experiments, real-time or the dashboard."],
    ["Dashboard says no live run", "Start `python -m medtwin.realtime` in a second terminal."],
    ["Parquet error", "Install pyarrow: `pip install pyarrow`."],
    ["Training is slow", "Close other programs; skip tuning with `python -m medtwin.train` (reuses saved settings)."],
    ["Port 8501 in use", "`streamlit run dashboard/app.py --server.port 8502`"],
], [5.5, 10.5])

# ------------------------------------------------------------------ 8
doc.add_heading("8. Safety and ethics", level=1)
para("All traffic is synthetic and generated inside the program. Addresses are from private (RFC 1918) and "
     "documentation (RFC 5737) ranges, and no packets are sent to any network. The attack generators only write "
     "labelled records that resemble the network footprint of each threat; they are not attack tools. No real patient "
     "data is used.")

OUT.mkdir(parents=True, exist_ok=True)
zoom = doc.settings.element.find(qn("w:zoom"))
if zoom is not None and zoom.get(qn("w:percent")) is None:
    zoom.set(qn("w:percent"), "100")  # required by the schema; python-docx's default template omits it
doc.save(str(OUT / "ICT3214_CW2_User_Manual.docx"))
finalise(OUT / "ICT3214_CW2_User_Manual.docx", "MedTwin-IDS User Manual",
         "ICT3214 Security Analytics, Coursework 2: User Manual")
print("saved manual")
