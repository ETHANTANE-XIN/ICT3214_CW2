"""Front matter, introduction, related work and methods of the final report."""
from common import PROJ, f3, f4, pct


def related_and_intro(d, C, ctx):
    comp, best = ctx["comp"], ctx["best"]
    b = comp.loc[best]
    d.h1("Introduction")
    d.p(C("Hospital wards now run on connected devices. Patient monitors stream vital signs to a nurse station, "
          "infusion pumps accept rate changes over the network, and wearable sensors relay data through IoMT "
          "gateways. Many of these devices run legacy firmware with weak authentication, and practical attacks on "
          "medical devices have been shown for more than a decade [@halperin]. IoMT security surveys stress that data "
          "must be protected during collection, transmission and storage [@ghubaish]. In this setting a cyber attack "
          "is a patient-safety event: a spoofed monitor can hide deterioration, altered vitals can trigger the wrong "
          "treatment, and an unauthorised pump command can over-infuse a drug."))
    d.p(C("Machine learning (ML) intrusion detection can learn these threats from data, but IoMT research is held "
          "back by its datasets. Public sets such as WUSTL-EHMS-2020 [@hady] and ECU-IoHT [@ahmed] are small and fixed, "
          "so they cannot test unseen attacks, varied attack intensity or real-time operation, and ML evaluations in "
          "security often suffer from leakage and over-optimism [@sommer,@arp]. We therefore ask: can a digital twin of "
          "a hospital ward generate realistic labelled data (RQ1); which of three ML models gives the best balance of "
          "accuracy and processing time (RQ2); do features that only a twin can compute improve detection of "
          "integrity attacks (RQ3); how well do detectors handle attack families never seen in training (RQ4); and can "
          "the chosen model run in real time with explainable, device-level risk scores (RQ5)?"))
    d.p("We built MedTwin-IDS, which implements Idea 1 of the coursework with per-device risk scoring (Idea 7) and "
        "alert explanations (Idea 12). Our contributions are:")
    d.bullets([
        "A digital twin of an eight-bed ward that couples patient physiology, 24 medical devices and packet-level "
        "network behaviour, with benign clinical and operational anomalies and six attack families at three "
        "intensities, giving exactly labelled data (bonus: simulated dataset).",
        "A parser that turns packets into 55 flow-window features, including 11 twin features that check traffic "
        "against the asset registry, message sequence integrity and cross-sensor physiological consistency.",
        "A leakage-aware comparison of Random Forest, XGBoost and an MLP, checked with bootstrap intervals and three "
        "extra test wards, plus an ablation, zero-day, intensity and window-length study and SHAP explanations.",
        "Label-free twin consistency rules for unseen integrity attacks, and a real-time detector with an Isolation "
        "Forest safety net, per-device risk scores, grouped incidents and a Streamlit dashboard (bonus: real-time "
        "detection).",
    ])
    d.p(f"On an independent test run, the selected {best} model reaches {pct(b['accuracy'], 2)} accuracy, a binary "
        f"F1-score of {f4(b['f1'])} and a macro F1-score of {f3(b['macro_f1'])} over seven classes, while scoring a flow "
        f"in {b['latency_ms']:.2f} ms. " + ctx["intro_findings"])

    d.h1("Related Work")
    d.p(C("**IoMT datasets.** Hady et al. [@hady] built a healthcare monitoring testbed and released WUSTL-EHMS-2020, "
          "about 16,000 records with spoofing and data-alteration attacks, and showed that combining network and "
          "biometric features improves detection. ECU-IoHT [@ahmed] provides IoHT traffic with ARP spoofing, DoS and "
          "scanning and found nearest-neighbour anomaly detectors effective. MQTTset [@vaccari] and MQTT-IoT-IDS2020 "
          "[@hindy] cover the MQTT protocol used by many IoMT gateways; the latter found flow-level features "
          "essential for MQTT attacks. General datasets such as CIC-IDS2017 [@sharafaldin] and UNSW-NB15 [@moustafa] "
          "have no medical context. All of these are fixed captures, so attack mix, intensity and timing cannot be "
          "varied."))
    d.p(C("**Detection methods.** Tree ensembles [@breiman,@chen] are strong baselines for flow data. N-BaIoT "
          "[@meidan] and Kitsune [@mirsky] use autoencoders to detect IoT attacks without labels, and Kitsune works "
          "online; Isolation Forest [@liu] is a light alternative for anomaly detection [@chandola]. Sommer and Paxson "
          "[@sommer] and Arp et al. [@arp] warn that closed-world assumptions, data snooping and spurious correlations "
          "inflate reported accuracy. **Digital twins** are well established in industry for design, production and "
          "prognostics [@tao], but we found little work that uses a healthcare twin both to generate security data "
          "and to supply detection features."))
    d.table("Comparison With Related Datasets and Systems",
            ["Work", "Patient vitals", "Controllable attacks", "Unseen-attack test", "Device risk score"],
            [[C("WUSTL-EHMS [@hady]"), "Yes", "No", "No", "No"],
             [C("ECU-IoHT [@ahmed]"), "No", "No", "No", "No"],
             [C("MQTT-IoT-IDS [@hindy]"), "No", "No", "No", "No"],
             [C("Kitsune [@mirsky]"), "No", "No", "Yes", "No"],
             ["MedTwin-IDS (ours)", "Yes", "Yes", "Yes", "Yes"]],
            [1.15, 0.55, 0.6, 0.6, 0.55], size=6.5,
            footnote="Controllable attacks: attack type, intensity and timing can be set when generating data. "
                     "Unseen-attack test: detection evaluated on attacks absent from training.")


def methods(d, C, ctx):
    tr, te = ctx["tr"], ctx["te"]
    d.h1("Methods")
    d.h2("System Overview")
    d.p("Fig. 1 shows the pipeline. The digital twin (1) produces packet events, which are stored with ground truth "
        "and parsed into flow features (2). Three supervised models are trained and one is selected by "
        "cross-validation (3); an Isolation Forest and twin consistency rules, both built from normal traffic only, "
        "act as safety nets for unseen attacks. The selected model scores the live stream, and a risk engine turns "
        "detections into device risk scores, incidents and recommended actions (4) shown on a dashboard (5).")
    d.figure(PROJ / "docs" / "diagrams" / "architecture.png",
             "MedTwin-IDS architecture (PlantUML). Labels (1a) to (5) give the order of processing.", width_in=5.6,
             wide=True)

    d.h2("Digital Twin and Data Generation")
    d.p("**Ward and devices.** The twin models an eight-bed ward. Each bed has a patient monitor publishing vitals at "
        "1 Hz, an infusion pump reporting its rate every 5 s and a wearable sensor publishing every 2 s, all over MQTT "
        "(QoS 1) to an IoMT gateway. A nurse station subscribes to all telemetry and sends pump commands, the gateway "
        "sends HL7 batches to the EHR server every 30 s, two workstations open HTTPS sessions to the EHR and a vendor "
        "portal, and the gateway refreshes ARP entries. The twin's asset registry records each device's IP, MAC, "
        "expected TTL and clinical criticality.")
    d.p("**Physiology.** Each patient's heart rate, SpO~2~, temperature, blood pressure, respiration rate and ST "
        "segment follow an Ornstein-Uhlenbeck process around a personal baseline with a daily rhythm. Clinical "
        "episodes (tachycardia, bradycardia, desaturation, fever, hypertension and ST elevation) ramp in and out over "
        "3 to 15 minutes. Monitors and wearables sample the same patient with different noise, so in normal "
        "operation they agree.")
    d.p("**Benign confounders.** To avoid an easy closed world [@sommer], the twin adds events that look anomalous "
        "but are legitimate: network congestion (higher jitter and retransmissions), 2 to 6 MB firmware downloads, "
        "device reboots (reconnects and sequence resets), nurse pump changes and pauses, nurse login retries, "
        "wearable motion artefacts and large EHR downloads. It also injects capture artefacts (duplicate frames, "
        "65535-style sensor sentinels and truncated records) for the cleaner to handle.".replace("[@sommer]", C("[@sommer]")))
    d.p("**Attacks.** Campaigns of 60 to 240 s are placed at random times, cycling through six families (Table II). "
        "Each is drawn at low, medium or high intensity and from a rogue host, a compromised workstation or a "
        "compromised device, so the same family appears with different footprints. During floods the gateway slows "
        "down, which also changes legitimate traffic.")
    d.table("Attack Families and Intensity Settings",
            ["Class", "Network and clinical footprint", "Low / medium / high"],
            [["DoS flood", "SYN or MQTT PUBLISH flood on the gateway from 1 to 3 sources", "25 / 80 / 250 pkt/s"],
             ["Spoofing", "MQTT client-ID takeover, or IP (and optionally MAC) clone; false or masked vitals",
              "1x / 2x / 4x rate"],
             ["Data alteration", "ARP poisoning; vitals offset in transit; extra delay, TTL may drop by one",
              "0.2 / 1 / 3 ARP/s"],
             ["Unauth. access", "MQTT credential guessing, then rogue infusion-rate command", "0.5 / 2 / 8 tries/s"],
             ["Reconnaissance", "Host and port scan of the ward subnet", "0.5 / 4 / 20 probes/s"],
             ["Exfiltration", "EHR records staged, then uploaded to an external host", "15 / 120 / 400 kB/s"]],
            [0.75, 1.9, 0.8], size=6.5)

    d.h2("Datasets")
    d.p(f"We generated two independent runs (Table III). The training run covers {tr['hours']:.0f} h from 08:00 with "
        f"seed 7; the test run covers {te['hours']:.0f} h from 20:00 with seed 2026, so patients, device ports, "
        "attack timing and attack sources all differ. Ground truth comes from the generator, so every packet is "
        "labelled exactly. Normal traffic dominates, as it does in practice.")
    classes = ["normal", "dos_flood", "spoofing", "data_alteration", "unauthorised_access", "reconnaissance",
               "exfiltration"]
    names = ["Normal", "DoS flood", "Spoofing", "Data alteration", "Unauth. access", "Reconnaissance", "Exfiltration"]
    rows = [["Packets", f"{tr['packets']:,}", f"{te['packets']:,}"],
            ["Attack campaigns", str(tr["campaigns"]), str(te["campaigns"])],
            ["Flow windows", f"{tr['flows']:,}", f"{te['flows']:,}"]]
    rows += [[f"  {n}", f"{tr['class_counts'].get(k, 0):,}", f"{te['class_counts'].get(k, 0):,}"]
             for k, n in zip(classes, names)]
    d.table("Generated Datasets", ["", "Training run", "Test run"], rows, [1.35, 0.95, 0.95], size=7)
    rz = ctx["realism"]
    d.p(f"**Realism checks (RQ1).** In normal training traffic the monitors report heart rates of {rz['hr'][0]:.0f} "
        f"to {rz['hr'][2]:.0f} bpm (5th to 95th percentile), SpO~2~ of {rz['spo2'][0]:.1f} to {rz['spo2'][2]:.1f}% and "
        f"temperatures of {rz['temp'][0]:.1f} to {rz['temp'][2]:.1f} °C, which are typical adult ward values. The "
        f"median telemetry interval is {rz['iat']:.2f} s, {rz['retrans']:.2f}% of packets are retransmissions, and the "
        f"monitor and wearable on the same patient agree within {rz['resid95']:.1f} bpm in 95% of normal windows. The "
        f"training run contains {rz['clinical']} clinical episodes, {rz['congestion']} congestion periods, "
        f"{rz['firmware']} firmware updates, {rz['reboot']} reboots and {rz['pump']} nurse pump commands, so benign "
        "anomalies are common rather than exceptional.")
    d.p(f"**Training and detecting data.** The initial training data is the training run's {tr['flows']:,} flow "
        "windows: its first 75% fit and its last 25% validate the hyperparameter search, and all of it is used for "
        "cross-validation and the final fit. The detecting data never touches training: the independent test run "
        f"({te['flows']:,} flows) for the one-time evaluation, three further 1 h runs (seeds 3001 to 3003) for "
        "robustness, and live streams (the test log replayed window by window and a new twin, seed 99) for real-time "
        "detection.")

    d.h2("Pre-processing and Feature Engineering")
    cl = ctx["clean"]
    d.p(f"**Cleaning.** The cleaner drops records missing essential fields ({cl['missing']} in the training run), "
        f"duplicate capture frames that are not TCP retransmissions ({cl['dups']}), and nulls vital-sign values "
        f"outside the sensors' measurement range ({cl['glitches']}). Clinically extreme but measurable values are "
        "kept, because deleting them would erase evidence of tampering.")
    d.p("**Flow windows.** Packets are grouped into 5 s service-level bidirectional flows keyed by initiator, "
        "responder, service port and protocol. The initiator is the sender of the TCP SYN where it is visible, "
        "otherwise the endpoint with the higher (ephemeral) port. Each flow window carries 55 features:")
    d.bullets([
        "**Network (34):** duration, packets, bytes, loads and rates per direction; packet-size extremes; mean "
        "inter-arrival time and jitter; SYN, RST, FIN and ARP counts; retransmissions; MQTT CONNECT, refused CONNACK, "
        "PUBLISH and command counts; host fan-out and port spread; service indicators.",
        "**Biometric (10):** mean temperature, SpO~2~, pulse, blood pressure, heart rate, respiration and ST from "
        "the flow's telemetry, infusion rate and a telemetry indicator, as in WUSTL-EHMS-2020 [@hady].".replace(
            "[@hady]", C("[@hady]")),
        "**Twin (11):** unknown source; MAC mismatch and TTL deviation against the registry; external destination; "
        "duplicated and missing sequence numbers per device; heart-rate and SpO~2~ residuals between a patient's "
        "monitor and wearable; change of vitals since the last window; within-window heart-rate spread; and the "
        "relative pump-rate change, including the change requested by a command.",
    ])
    d.p("Identifiers (addresses, ports, MACs, device names and time stamps) are never used as features, so models "
        "must learn behaviour rather than memorise hosts or schedules [@arp].".replace("[@arp]", C("[@arp]")))

    d.h2("Detection Models and Rationale")
    d.p(C("We compare three supervised models that represent different learning families, all built with "
          "scikit-learn [@pedregosa] and xgboost:"))
    d.bullets([
        C("**Random Forest (RF)** [@breiman]: bagged, decorrelated decision trees. It is robust to scale and noise, "
          "needs little tuning and is a standard IDS baseline. Class weights handle imbalance."),
        C("**XGBoost** [@chen]: gradient-boosted trees that add each tree to correct the residual errors of the "
          "ensemble, with regularisation and histogram-based splits. Boosting suits rare, hard classes, which we "
          "weight by the square root of inverse class frequency."),
        "**Multilayer perceptron (MLP)**: a deep feed-forward network with three ReLU hidden layers ("
        + "-".join(str(u) for u in ctx["params"]["MLP"]["hidden_layer_sizes"]) + " units after tuning), Adam and "
        "early stopping, on log-scaled, standardised features. Rare classes are randomly oversampled.",
    ])
    d.p(C("An **Isolation Forest** [@liu] is trained on normal training traffic only. It needs no attack labels, so "
          "it can flag behaviour that no supervised model has seen. Its thresholds are set at the 1st (standard) and "
          "0.1st (strict, live) percentiles of normal anomaly scores."))

    d.h2("Training and Evaluation Protocol")
    d.p(C("Following Arp et al. [@arp], we prevent leakage at every stage. (i) Hyperparameters are tuned on a "
          "temporal hold-out: the first 75% of the training run fits and the last 25% validates (20 settings in "
          "total). (ii) Five-fold cross-validation groups windows into 30-minute time blocks, so adjacent windows of "
          "one campaign never sit on both sides of a split. (iii) The deployed model is chosen on cross-validation "
          "macro F1-score alone, with near-ties (below 0.002) broken by latency. (iv) Only then is each model tested "
          "once on the independent test run."))
    d.p("We report accuracy, precision, recall, F1-score, false positive rate (FPR) and ROC AUC for attack versus "
        "normal, and macro F1-score over the seven classes, which weights rare attacks equally. Cost is measured as "
        "training time, batch inference time per flow and single-flow latency (median of 100 calls on one thread).")

    d.h2("Real-time Detection and Risk Scoring")
    d.p("The live detector steps the twin (or replays a log) one 5 s window at a time, applies the same cleaner and "
        "feature extractor, and scores each flow with the deployed model. A flow is an alert when "
        "P(attack) = 1 − P(normal) ≥ 0.5. Each flow's threat is T = P(attack)·s(c), where s(c) is the "
        "severity of the most likely attack class c (0.5 for reconnaissance up to 1.0 for data alteration and "
        "unauthorised access). Every device involved in the flow is updated as")
    d.equation("R~t~ = min(100, max(100·T·w, λ·R~t−1~))", 1)
    d.p("where w = 0.6 + 0.4k, k is the device's criticality in the registry (1.0 for pumps and the gateway, 0.6 for wearables) and "
        "λ = 0.9 per window, so risk jumps at once and decays with a half-life of about 33 s. Two rules from the "
        "twin keep blame accurate: when a frame's MAC does not match the registry, the device whose address is "
        "claimed is treated as a victim, and an unregistered host becomes its own tracked entity. Alerts for the "
        "same device and class within 30 s are grouped into one incident, labelled with its three most abnormal "
        "features (z-scores against normal traffic, weighted by model importance) and a playbook action. Flows "
        "that the supervised model passes but the strict Isolation Forest flags become “unknown anomaly” "
        "alerts; with the optional twin rules (E3), a passed flow that breaks a rule becomes a “twin rule” alert "
        "first.")

    d.h2("Implementation and Testing")
    d.p(C("The system is a Python 3 package of 13 modules (twin, attacks, simulator, parser, models, training, "
          "experiments, rules, risk engine, real-time detector, figures, benchmark and configuration) plus a "
          "Streamlit dashboard. We used pandas and NumPy for data handling, scikit-learn [@pedregosa] for RF, the MLP, "
          "the Isolation Forest and cross-validation, xgboost [@chen], SHAP [@lundberg] for explanations, Plotly for "
          "the dashboard and PlantUML for the system diagram. Every run is seeded, so datasets and results are "
          "reproducible.")
          + f" An automated suite of {ctx['n_tests']} pytest tests passes. It checks that the twin is deterministic "
          "and labels every packet, that the cleaner removes capture artefacts but keeps extreme vital signs, that "
          "flows are oriented by the SYN, that no identifier is used as a feature, that rogue pump commands raise the "
          "pump-rate feature, that the twin rules and risk attribution behave as designed, that models train on any "
          "subset of classes, that the benchmark loader matches the WUSTL-EHMS-2020 schema, and that the real-time "
          "detector runs end to end. One test confirms a key property: extracting features window by window, as the "
          "live detector does, gives exactly the same values for all 55 features as offline extraction, so the live "
          "model sees the inputs it was trained on.")
