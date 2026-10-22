"""Build the Coursework 2 project proposal (IEEE template, .docx)."""
from common import (FOOTER, GROUP_ID, MEMBERS, OUT, PROJ, TEMPLATE, TITLE, comparison, f3, f4, report)
from finalise import finalise
from ieee_doc import IEEEDoc
from refs import Citer

c = Citer()
C = c.cite
tr, te = report("train"), report("test")
comp = comparison()
best = comp.index[comp["selected"]][0]
b = comp.loc[best]

d = IEEEDoc(str(TEMPLATE), footer_text=FOOTER)
d.title(TITLE, ["ICT3214 Security Analytics, Coursework 2: Project Proposal", f"Group ID: {GROUP_ID}"])
d.authors(MEMBERS)

d.abstract(
    "Hospitals depend on networked patient monitors, infusion pumps and wearable sensors that are hard to patch "
    "and rarely monitored, yet an attack on them can directly harm patients. This proposal describes MedTwin-IDS, "
    "a security analytics project that builds a digital twin of an eight-bed hospital ward and uses it to generate "
    "labelled medical Internet of Things (IoMT) traffic. The twin produces normal telemetry, realistic clinical and "
    "network anomalies, and six attack families (denial of service, device spoofing, in-transit data alteration, "
    "unauthorised access to infusion pumps, reconnaissance and data exfiltration) at three intensities. A parser "
    "turns the packet log into flow windows that combine network statistics, patient biometrics and twin-derived "
    "context features. Three machine learning models, Random Forest, XGBoost and a multilayer perceptron (MLP), will "
    "be compared on accuracy, precision, recall and processing time, with an Isolation Forest added for attacks never "
    "seen in training. The selected model will run in real time, assign a risk score to every device and explain each "
    f"alert on an analyst dashboard. A working prototype already exists: on an independent test run, the preliminary "
    f"{best} model reaches a binary F1-score of {f4(b['f1'])} and a macro F1-score of {f3(b['macro_f1'])} across seven "
    "classes.")
d.keywords("medical IoT, digital twin, intrusion detection, machine learning, XGBoost, risk scoring, healthcare security")

d.h1("Introduction and Problem Definition")
d.p(C("Connected medical devices now stream vital signs to nurse stations and receive commands such as infusion-rate "
      "changes over the hospital network. Many run legacy firmware with weak authentication, and attacks on medical "
      "devices have been demonstrated for well over a decade [@halperin]. Surveys of IoMT security show that the "
      "data in such systems must be protected while it is collected, transmitted and stored [@ghubaish]. A spoofed "
      "monitor that hides a patient's deterioration, or an unauthorised command that changes an infusion rate, is a "
      "patient-safety incident, not only a data breach."))
d.p(C("Machine learning intrusion detection for IoMT is limited by data. Public datasets such as WUSTL-EHMS-2020 "
      "[@hady] and ECU-IoHT [@ahmed] are valuable but fixed in size, attack mix and time span, so they cannot be used "
      "to study unseen attacks, varying attack intensity or real-time behaviour. Evaluations are also prone to "
      "leakage and over-optimism [@sommer,@arp]. This project therefore investigates the following questions:"))
d.bullets([
    "**RQ1.** Can a ward-level digital twin generate realistic, labelled IoMT data that supports training and fair "
    "evaluation of detectors?",
    "**RQ2.** Which of Random Forest, XGBoost and an MLP gives the best balance of detection quality and processing "
    "time, and why?",
    "**RQ3.** Do twin-derived context features (asset registry checks, message sequence integrity and cross-sensor "
    "physiological consistency) improve detection of integrity attacks beyond network and biometric features?",
    "**RQ4.** How well do the detectors generalise to attack families not seen in training, and can an unsupervised "
    "model close the gap?",
    "**RQ5.** Can the selected model run in real time and give analysts explainable, device-level risk scores?",
])
d.p("The project implements Idea 1 (AI-driven intrusion detection for medical IoT digital twins) and borrows two "
    "elements from related ideas: a continuously updated per-device risk score (Idea 7) and human-readable alert "
    "explanations (Idea 12).")

d.h1("Objectives and Scope")
d.bullets([
    "Build a digital-twin simulator of a hospital ward that generates labelled packet-level data (dataset Option 1).",
    "Develop a parser and cleaner that converts raw packets into flow-window features.",
    "Develop and compare three ML models on accuracy, precision, recall, F1-score and processing time, and justify "
    "the final choice.",
    "Evaluate generalisation to unseen attacks, attack intensity and window length, with leakage-aware protocols.",
    "Deliver real-time detection with per-device risk scores, grouped incidents and an analyst dashboard.",
    C("Benchmark against published results on the public WUSTL-EHMS-2020 dataset [@wustl_ds]."),
])
d.p("Out of scope: real hospital networks, live patient data and automatic blocking actions on devices. All traffic "
    "is synthetic and uses private or documentation address ranges.")

d.h1("Related Work")
d.p(C("Hady et al. [@hady] built a real-time enhanced healthcare monitoring testbed and released WUSTL-EHMS-2020, about "
      "16,000 records with spoofing and data-alteration attacks. They showed that combining network and biometric "
      "features improves detection over either alone. ECU-IoHT [@ahmed] provides IoHT traffic with ARP spoofing, DoS "
      "and scanning attacks. MQTTset [@vaccari] and MQTT-IoT-IDS2020 [@hindy] focus on the MQTT protocol used by many "
      "IoMT gateways, and the latter found that flow-level features matter for MQTT attacks. General-purpose datasets "
      "such as CIC-IDS2017 [@sharafaldin] and UNSW-NB15 [@moustafa] lack medical context. On the modelling side, "
      "N-BaIoT [@meidan] and Kitsune [@mirsky] use autoencoders to detect IoT attacks without labels, and Kitsune "
      "runs online. Digital twins are well established in industry [@tao], but we found little work that uses a "
      "healthcare twin both to generate security data and to supply detection features."))
d.p("Our work differs in four ways: the twin is controllable (attack type, intensity and timing), it yields "
    "physiological and network data together, it supplies consistency features that only a twin can compute, and "
    "the same pipeline is tested on unseen attacks and in real time.")

d.h1("Data: Digital Twin Simulation")
d.h2("Ward Model")
d.p("The twin models an eight-bed ward. Each bed has a patient monitor (1 Hz telemetry), an infusion pump (every 5 s) "
    "and a wearable sensor (every 2 s) that publish over MQTT to an IoMT gateway. A nurse station subscribes to all "
    "telemetry and sends pump commands, an HL7 interface sends batches to the EHR server every 30 s, and two "
    "clinical workstations open HTTPS sessions to the EHR and a vendor portal. An asset registry stores each "
    "device's address, MAC, expected TTL and clinical criticality.")
d.p("Each patient's vital signs follow an Ornstein-Uhlenbeck process around a personal baseline with a daily "
    "rhythm. Clinical episodes (tachycardia, bradycardia, desaturation, fever, hypertension and ST elevation) and "
    "benign network events (congestion, firmware updates, device reboots, nurse login retries and wearable motion "
    "artefacts) are added deliberately, so that a detector must separate cyber attacks from genuine clinical or "
    "operational anomalies.")
d.table("Attack Families Injected by the Twin",
        ["Class", "Footprint", "Intensity knob"],
        [["DoS flood", "SYN or MQTT PUBLISH flood on gateway", "25 / 80 / 250 pkt/s"],
         ["Spoofing", "Client-ID takeover or IP/MAC clone with false vitals", "1x / 2x / 4x rate"],
         ["Data alteration", "ARP poisoning, vitals modified in transit", "0.2 / 1 / 3 ARP/s"],
         ["Unauth. access", "MQTT credential guessing, rogue pump command", "0.5 / 2 / 8 tries/s"],
         ["Reconnaissance", "Host and port scanning of the ward", "0.5 / 4 / 20 probes/s"],
         ["Exfiltration", "Bulk upload of records to an external host", "15 / 120 / 400 kB/s"]],
        [0.8, 1.75, 0.85], size=7)
d.h2("Datasets")
d.p(f"Two independent runs are generated with different random seeds, patients and times of day: a training run "
    f"({tr['hours']:.0f} h from 08:00, {tr['campaigns']} campaigns, {tr['packets']:,} packets, {tr['flows']:,} flow "
    f"windows) and a test run ({te['hours']:.0f} h from 20:00, {te['campaigns']} campaigns, {te['packets']:,} packets, "
    f"{te['flows']:,} flow windows). Every packet carries ground-truth labels, so labels are exact. For comparison "
    "with published work, the same pipeline will also be run on WUSTL-EHMS-2020 [@wustl_ds].".replace(
        "[@wustl_ds]", C("[@wustl_ds]")))

d.h1("Proposed Method")
d.p("Fig. 1 shows the planned pipeline. The numbers give the order of processing: (1) twin simulation, (2) data "
    "engineering, (3) model training and selection, (4) real-time response and (5) the dashboard.")
d.figure(PROJ / "results" / "figures" / "fig_architecture_column.png", "MedTwin-IDS architecture.", width_in=3.3)
d.h2("Pre-processing and Features")
d.p("The cleaner removes truncated records and duplicated capture frames and nulls sensor values outside the "
    "devices' measurement ranges. Packets are then grouped into 5 s service-level bidirectional flows, oriented by "
    "the TCP SYN where visible. Each flow window has 55 features: 34 network features (volumes, rates, packet sizes, "
    "inter-arrival times, jitter, TCP flags and MQTT message counts), 10 biometric features and 11 twin features, "
    "such as an unknown source, MAC or TTL mismatch with the registry, duplicated or missing message sequence numbers, "
    "the difference between a monitor and the wearable on the same patient, and the size of a commanded pump-rate "
    "change.")
d.h2("Models")
d.p(C("**Random Forest** [@breiman] averages many decorrelated trees and is a robust, interpretable baseline. "
      "**XGBoost** [@chen] adds trees that correct the previous ensemble's errors and is strong on tabular flow data. "
      "A three-layer **MLP** represents deep learning and is the fastest at inference. An **Isolation Forest** "
      "[@liu] trained only on normal traffic acts as a safety net for unseen attacks. Class imbalance is handled by "
      "class weights or oversampling. Models are built with scikit-learn [@pedregosa] and xgboost."))
d.h2("Evaluation Protocol")
d.p(C("Following Arp et al. [@arp], we avoid leakage. Hyperparameters are tuned on a temporal hold-out, five-fold "
      "cross-validation uses 30-minute time blocks as groups, the model is selected on cross-validation alone, and "
      "results are reported once on the independent test run. Metrics are accuracy, precision, recall, F1-score, "
      "false positive rate, ROC AUC, macro F1-score across the seven classes, training time and per-flow latency. "
      "Planned experiments are: E1 model comparison, E2 feature-set ablation, E3 leave-one-attack-out zero-day "
      "tests, E4 attack intensity, E5 window length versus time-to-detect and E6 real-time operation. SHAP "
      "[@lundberg] will explain the selected model."))
d.h2("Real-time Detection and Dashboard")
d.p("A streaming loop drives the twin live, extracts features every window, scores flows, and updates a risk score "
    "per device that combines detection confidence, attack severity and asset criticality. Alerts are grouped into "
    "incidents with the most abnormal features and a recommended action. A Streamlit dashboard shows a live ward "
    "map, incidents, risk timelines and the evaluation results.")

d.h1("Expected Outcomes")
d.bullets([
    "A reusable digital-twin dataset generator with exact labels and adjustable attack intensity.",
    "Three trained and compared detectors, with a justified choice based on accuracy and processing time.",
    "Evidence on whether twin features help detect integrity attacks, and how well models handle unseen attacks.",
    "A real-time detector with device risk scores, explanations and a dashboard, plus a reproducible user manual.",
])

d.h1("Progress to Date")
d.p("The simulator, parser, three models, Isolation Forest, real-time detector and a first dashboard are working. "
    "Table II gives preliminary results of experiment E1 on the independent test run. All three models detect "
    f"attacks well at binary level, but the macro F1-score shows clear differences on the hardest classes, "
    f"spoofing and data alteration. {best} currently has the best cross-validation score and the best test macro "
    f"F1-score, narrowly ahead of the MLP. The remaining "
    "experiments (E2 to E6) and the WUSTL-EHMS-2020 benchmark will be completed before the final report.")
rows = []
for m in ["Random Forest", "XGBoost", "MLP"]:
    r = comp.loc[m]
    rows.append([m, f4(r["accuracy"]), f4(r["f1"]), f3(r["macro_f1"]), f3(r["cv_macro_f1"]), f"{r['latency_ms']:.2f}"])
d.table("Preliminary Results on the Independent Test Run",
        ["Model", "Accuracy", "Binary F1", "Macro F1", "CV macro F1", "Latency (ms)"], rows,
        [0.85, 0.55, 0.55, 0.5, 0.5, 0.5], size=7,
        footnote="Binary metrics treat all attacks as one class. Latency is the median time to score one flow on one CPU thread.")

d.h1("Project Plan and Task Allocation")
d.table("Timeline",
        ["Week", "Dates (2026)", "Milestones"],
        [["1", "19 to 25 Oct", "Topic, literature review, twin design"],
         ["2", "26 Oct to 1 Nov", "Simulator, parser, baseline models, proposal (1 Nov)"],
         ["3", "2 to 8 Nov", "Tuning, experiments E1 to E5, WUSTL benchmark"],
         ["4", "9 to 15 Nov", "Real-time detector, risk engine, dashboard, testing"],
         ["5", "16 to 22 Nov", "Report, slides, user manual, demo rehearsal"],
         ["6", "23 Nov", "Final submission and peer evaluation"]],
        [0.4, 0.95, 2.05], size=7)
d.table("Task Allocation",
        ["Member", "Main responsibilities"],
        [[MEMBERS[0]["name"], "Project lead; patient physiology and device layer of the twin; report integration"],
         [MEMBERS[1]["name"], "Network layer and attack campaign generator; dataset validation"],
         [MEMBERS[2]["name"], "Parser, cleaner and feature engineering; WUSTL-EHMS-2020 benchmark"],
         [MEMBERS[3]["name"], "Model training, tuning, evaluation protocol and experiments E1 to E5"],
         [MEMBERS[4]["name"], "Real-time detector, risk engine, dashboard and user manual"]],
        [1.2, 2.2], size=7, align=["left", "left"],
        footnote="All members contribute to the literature review, slides and presentation.")

d.h1("Risks and Mitigations")
d.bullets([
    "**Simulation realism.** Models may learn simulator artefacts. Mitigation: benign confounders, independent "
    "test run, ablation and zero-day tests, and the public WUSTL-EHMS-2020 benchmark.",
    "**Class imbalance.** Rare attacks may be ignored. Mitigation: class weighting, oversampling and macro metrics.",
    "**Time.** Mitigation: the core pipeline is already working, leaving three weeks for experiments and writing.",
])

d.references(c.reference_list())
OUT.mkdir(parents=True, exist_ok=True)
d.save(str(OUT / "ICT3214_CW2_Project_Proposal.docx"))
finalise(OUT / "ICT3214_CW2_Project_Proposal.docx", TITLE,
         "ICT3214 Security Analytics, Coursework 2: Project Proposal")
print("saved proposal with", len(c.order), "references")
