"""Build the final report (IEEE template, .docx)."""
import json
import re

import pandas as pd

from common import (FOOTER, GROUP_ID, MEMBERS, OUT, PROJ, RES, ROOT, TEMPLATE, TITLE, comparison, f3, f4, pct, report)
from finalise import finalise
from ieee_doc import IEEEDoc
from refs import Citer
from report_appendix import appendix
from report_front import methods, related_and_intro
from report_results import conclusion, load, results

cit = Citer()
C = cit.cite
ctx = load()
comp = comparison()
best = comp.index[comp["selected"]][0]
tr, te = report("train"), report("test")
params = json.loads((RES / "best_params.json").read_text())
ctx.update(comp=comp, best=best, tr=tr, te=te, params=params)
xg, rf, ml = comp.loc["XGBoost"], comp.loc["Random Forest"], comp.loc["MLP"]
ab = ctx["ab"].set_index(["feature_set", "model"])
zd, ws, it, fi, spc = ctx["zd"], ctx["ws"], ctx["it"], ctx["fi"], ctx["spc"]
live, rep = ctx["live"], ctx["replay"]
diag = ctx["diag"]

# ------------------------------------------------------------------ realism checks (RQ1)
_pk = pd.read_parquet(PROJ / "data" / "train_packets.parquet",
                      columns=["ts", "device_id", "hr", "spo2", "temp", "mqtt", "label", "retrans"])
_n = _pk[_pk["label"] == 0]
_mon = _n[(_n["mqtt"] == "PUBLISH") & _n["hr"].notna() & _n["device_id"].str.startswith("MON")]
_ev = pd.read_csv(PROJ / "data" / "train_benign_events.csv")["kind"]
_fl = pd.read_csv(PROJ / "data" / "train_flows.csv", usecols=["attack", "HasVitals", "TwinResidualHR", "device_id"])
_res = _fl[(_fl["attack"] == "normal") & (_fl["HasVitals"] > 0) & _fl["device_id"].fillna("").str.match("MON|WRB")]
ctx["realism"] = dict(
    hr=_mon["hr"].quantile([.05, .5, .95]).tolist(), spo2=_mon["spo2"].quantile([.05, .5, .95]).tolist(),
    temp=_mon["temp"].quantile([.05, .5, .95]).tolist(),
    iat=float(_mon.sort_values("ts").groupby("device_id")["ts"].diff().median()),
    retrans=100 * float(_n["retrans"].mean()), resid95=float(_res["TwinResidualHR"].quantile(.95)),
    clinical=int(_ev.str.startswith("clinical_").sum()), congestion=int((_ev == "congestion").sum()),
    firmware=int((_ev == "firmware_update").sum()), reboot=int((_ev == "reboot").sum()),
    pump=int((_ev == "nurse_pump_command").sum()))

# ------------------------------------------------------------------ automated tests (captured console output)
_pt = (ROOT / "build" / "terminal" / "pytest.txt").read_text()
ctx["n_tests"] = int(re.search(r"(\d+) passed", _pt).group(1))
assert " failed" not in _pt and " error" not in _pt.lower(), "the captured test run must be clean"

# ------------------------------------------------------------------ cleaning numbers
ctx["clean"] = dict(missing=f"{tr['dropped_missing_essential']:,}", dups=f"{tr['dropped_exact_duplicates']:,}",
                    glitches=f"{tr['vital_glitches_nulled']:,}")

# ------------------------------------------------------------------ zero-day summary
zmean = zd.groupby("detector")[["recall", "fpr"]].mean()
sup_mean = zmean.loc[["Random Forest", "XGBoost", "MLP"], "recall"]
best_sup = sup_mean.idxmax()
hyb = f"Hybrid ({best} + IF)"
zrow = lambda det, att: float(zd[(zd["detector"] == det) & (zd["held_out"] == att)]["recall"].iloc[0])
NM = {"dos_flood": "DoS", "spoofing": "spoofing", "data_alteration": "data alteration",
      "unauthorised_access": "unauthorised access", "reconnaissance": "reconnaissance", "exfiltration": "exfiltration"}

# ------------------------------------------------------------------ window length
w2, w5, w10 = (ws[ws["window_s"] == w].iloc[0] for w in (2.0, 5.0, 10.0))

# ------------------------------------------------------------------ intensity
itb = it[it["model"] == best].set_index(["attack", "intensity"])["recall"]
itr = it[it["model"] == "Random Forest"].set_index(["attack", "intensity"])["recall"]

ctx["intro_findings"] = (
    f"Twin features raise the macro F1-score of {best} from {ab.loc[('network_biometric', best), 'macro_f1']:.3f} "
    f"(network and biometric features only) to {ab.loc[('full_twin', best), 'macro_f1']:.3f}, mainly by exposing "
    f"spoofing and data alteration. Supervised models detect few attacks from families they never saw (mean "
    f"recall {zmean.loc[best, 'recall']:.2f} for {best}); adding an Isolation Forest raises this to "
    f"{zmean.loc[hyb, 'recall']:.2f}, and label-free twin consistency rules raise it to "
    f"{zmean.loc[f'Hybrid ({best} + IF + twin rules)', 'recall']:.2f}, at the cost of more false positives. "
    f"In streaming operation every one of "
    f"{rep['campaigns']} test campaigns is detected, with a median time-to-detect of {rep['median_delay_s']} s.")

ctx["text_e2"] = (
    f"{{FIG}} removes feature groups and retrains every model. Biometric features alone are of little use (macro F1 "
    f"at most {max(ab.loc[('biometric_only', m), 'macro_f1'] for m in ['Random Forest', 'XGBoost', 'MLP']):.2f}), "
    f"because the falsified vitals in our attacks are plausible values. Network features alone give "
    f"{ab.loc[('network_only', best), 'macro_f1']:.3f} for {best} but recall only "
    f"{pct(ab.loc[('network_only', best), 'recall_spoofing'], 0)} of spoofing flows. Adding the biometric features, "
    f"which is the WUSTL-EHMS-2020 design, changes little ({ab.loc[('network_biometric', best), 'macro_f1']:.3f}). "
    f"Adding the 11 twin features lifts {best} to {ab.loc[('full_twin', best), 'macro_f1']:.3f}, spoofing recall "
    f"to {pct(ab.loc[('full_twin', best), 'recall_spoofing'], 0)} and data-alteration recall from "
    f"{pct(ab.loc[('network_biometric', best), 'recall_data_alteration'], 0)} to "
    f"{pct(ab.loc[('full_twin', best), 'recall_data_alteration'], 0)}, while the false positive rate falls from "
    f"{100 * ab.loc[('network_biometric', best), 'fpr']:.2f}% to {100 * ab.loc[('full_twin', best), 'fpr']:.2f}%. "
    "The insight is that the value of patient data lies less in the vital signs themselves than in whether they "
    "are consistent: with the patient's other sensor, with the device's message sequence and with what the "
    "registry expects. That consistency check is exactly what a digital twin can provide (RQ3).")

iso_r = zd[zd["detector"] == "Isolation Forest"].set_index("held_out")["recall"]
best_r = zd[zd["detector"] == best].set_index("held_out")["recall"]
ctx["text_e3"] = (
    f"{{FIG}} removes one attack family from training at a time. The supervised models largely fail on families "
    f"they have never seen: mean recall is {zmean.loc['Random Forest', 'recall']:.2f} for RF, "
    f"{zmean.loc['XGBoost', 'recall']:.2f} for XGBoost and {zmean.loc['MLP', 'recall']:.2f} for the MLP, because a "
    f"closed-world classifier assigns the unknown flows to the nearest class it knows, usually normal "
    f"[@sommer]. {best}'s best case is exfiltration ({best_r['exfiltration']:.2f}), whose volume resembles other "
    f"known attacks. The Isolation Forest, trained on normal traffic only, reaches "
    f"{zmean.loc['Isolation Forest', 'recall']:.2f} on its own and is strong on loud attacks (DoS "
    f"{iso_r['dos_flood']:.2f}, exfiltration {iso_r['exfiltration']:.2f}, reconnaissance "
    f"{iso_r['reconnaissance']:.2f}, unauthorised access {iso_r['unauthorised_access']:.2f}) but weak on the "
    f"integrity attacks (data alteration {iso_r['data_alteration']:.2f}, spoofing {iso_r['spoofing']:.2f}), whose "
    "flows look normal in volume and timing. The hybrid of the two reaches "
    f"{zmean.loc[hyb, 'recall']:.2f}, but its false positive rate rises from {100 * zmean.loc[best, 'fpr']:.2f}% to "
    f"{100 * zmean.loc[hyb, 'fpr']:.2f}%. Two lessons follow: an unsupervised safety net is essential for unknown "
    "attacks, and unknown integrity attacks need explicit twin consistency checks rather than generic anomaly "
    "scores. The live detector therefore uses a stricter Isolation Forest threshold and groups alerts (RQ4).")

# ------------------------------------------------------------------ twin consistency rules (designed fix for E3)
rul = zd[zd["detector"] == "Twin rules"].set_index("held_out")["recall"]
hyb_r, hyb_ir = f"Hybrid ({best} + twin rules)", f"Hybrid ({best} + IF + twin rules)"
sens = ctx["sens"].set_index("quantile")
lr = ctx["live_rules"]
# which rule fired for each twin-rule alert on a flow labelled normal in the live run with rules
rfp = dict(R1=0, R2=0, R3=0, R4=0, R5=0)
for _a in (json.loads(l) for l in open(RES / "e6_live_rules_alerts.jsonl")):
    if _a["attack"] == "twin_rule" and _a["truth"] == "normal":
        for _e in _a["explanation"]:
            if _e.get("normal_mean") == "rule":
                rfp[_e["feature"]] += 1
rfp_total = sum(1 for l in open(RES / "e6_live_rules_alerts.jsonl")
                if '"twin_rule"' in l and json.loads(l)["truth"] == "normal")
ctx["rules"] = dict(spoof=rul["spoofing"], alt=rul["data_alteration"], fpr=zmean.loc["Twin rules", "fpr"],
                    all_rec=zmean.loc[hyb_ir, "recall"], all_fpr=zmean.loc[hyb_ir, "fpr"])
ctx["text_e3_rules"] = (
    f"**A designed fix: twin consistency rules.** These lessons point to five invariants that the twin knows for "
    "every device: (R1) the sender must be registered; (R2) its MAC address and TTL must match the registry; (R3) "
    "its sequence numbers must be neither duplicated nor missing; (R4) a patient's monitor and wearable must agree; "
    "and (R5) a pump's rate must not change unexpectedly. We implemented them as label-free rules whose "
    "thresholds are the 99.9th percentile of each signal in normal training traffic, so they need no attack "
    f"examples and are the same whichever family is held out. On their own they catch {pct(rul['spoofing'], 0)} of "
    f"unseen spoofing and {pct(rul['data_alteration'], 0)} of unseen data alteration, exactly the integrity attacks "
    f"on which the Isolation Forest is weakest, at a false positive rate of "
    f"{100 * zmean.loc['Twin rules', 'fpr']:.2f}%. Combined with {best} and the Isolation Forest, mean recall on "
    f"held-out families rises from {zmean.loc[hyb, 'recall']:.2f} to {zmean.loc[hyb_ir, 'recall']:.2f}, with every "
    f"family at {zd[zd['detector'] == hyb_ir]['recall'].min():.2f} or above, for a false positive rate of "
    f"{100 * zmean.loc[hyb_ir, 'fpr']:.2f}%. The threshold sets the trade-off: at the 99.99th percentile the rules' "
    f"false positive rate falls to {100 * sens.loc[0.9999, 'fpr']:.2f}% but unseen-spoofing recall drops to "
    f"{pct(sens.loc[0.9999, 'spoofing'], 0)}. In the live run every family is known, so the rules cannot add "
    f"detections ({lr['detected']} of {lr['campaigns']} either way) and show only their cost: false alerts rise from "
    f"{live['false_alerts']} to {lr['false_alerts']} and incidents from {live['incidents']} to {lr['incidents']}. Of the "
    f"{rfp_total} rule alerts on normal flows, {rfp['R5']} are R5 firing on legitimate nurse pump commands (pump-rate "
    f"changes are so rare that its threshold is zero) and {rfp['R4']} are R4 on wearables that briefly disagree with "
    "their monitor. We therefore ship the rules as an option (--twin-rules) for wards that prioritise integrity "
    "attacks (RQ4).")


def iv(att, lv, s=itb):
    try:
        return f"{s.loc[(att, lv)]:.2f}"
    except KeyError:
        return "n/a"


ctx["text_e4"] = (
    f"**Intensity.** {{FIG}} shows that recall is 1.00 for every high-intensity campaign in the test run. Low intensity is "
    f"hardest for data alteration ({best}: {iv('data_alteration', 'low')} at low versus "
    f"{iv('data_alteration', 'high')} at high intensity), where the in-transit offset is only 6 to 10 beats per "
    f"minute and the ARP poisoning rate is one packet every five seconds. For RF, spoofing recall is "
    f"{iv('spoofing', 'low', itr)} (low) and {iv('spoofing', 'medium', itr)} (medium), against {iv('spoofing', 'low')} "
    f"and {iv('spoofing', 'medium')} for {best}. The random schedule placed no high-intensity spoofing or "
    "unauthorised-access campaign in the test run, so those cells are empty (Appendix C).")
ctx["text_e5"] = (
    f"**Window length.** Shorter windows react faster but see fewer packets per flow ({{FIG}}). Moving from 10 s to "
    f"5 s windows cuts the mean time-to-detect from {w10['mean_delay_s']:.1f} s to {w5['mean_delay_s']:.1f} s with "
    f"macro F1 of {w10['macro_f1']:.3f} and {w5['macro_f1']:.3f}; 2 s windows give {w2['mean_delay_s']:.1f} s and "
    f"{w2['macro_f1']:.3f}. We keep 5 s as the operating point. The means include exfiltration, which is only "
    "visible once the upload starts after the staging phase.")

top = fi.sort_values("XGBoost_SHAP", ascending=False)
share = fi.groupby("group")["XGBoost_SHAP"].sum()
pc_top = lambda c, k=2: " and ".join(spc[c].sort_values(ascending=False).head(k).index)
twin_rank = [i + 1 for i, f in enumerate(top.index[:10]) if top.loc[f, "group"] == "twin"]
ctx["text_xai"] = (
    f"SHAP [@lundberg] explains the selected model ({{FIG}}). Network features carry {pct(share['network'], 0)} of "
    f"the total attribution, twin features {pct(share['twin'], 0)} and biometric features "
    f"{pct(share['biometric'], 0)}; twin features are only 20% of the inputs but hold ranks "
    f"{', '.join(map(str, twin_rank))} of the top ten. Per class, the model's reasoning matches the attack "
    f"mechanics: reconnaissance is recognised by {pc_top('reconnaissance')}, DoS by {pc_top('dos_flood')}, "
    f"exfiltration by {pc_top('exfiltration')}, unauthorised access by {pc_top('unauthorised_access')}, spoofing "
    f"by {pc_top('spoofing')}, and data alteration by {pc_top('data_alteration')}. {{FIGNEXT}} shows why the residual "
    "works: in a real clinical episode the monitor and the wearable move together, while an in-transit alteration "
    "changes only the monitor. These explanations are what the dashboard shows an analyst with each incident.")

lc = ctx["live_c"]
# categorise live-run alerts on flows labelled normal
alerts = [json.loads(l) for l in open(RES / "e6_live_alerts.jsonl")]
gt = pd.read_csv(RES / "e6_live_ground_truth.csv")
partner = lambda dev: ("WRB-" + dev[4:]) if dev.startswith("MON-") else ("MON-" + dev[4:]) if dev.startswith("WRB-") else None
fp = dict(iforest=0, partner=0, pump=0, artefact=0, other=0)
for a in alerts:
    if a["truth"] != "normal":
        continue
    if a["attack"] == "anomaly_unknown":
        fp["iforest"] += 1
    elif partner(a["device"]) and any(partner(a["device"]) in str(c.target) and c.start - 5 <= a["t"] <= c.end + 10
                                      for c in gt.itertuples() if c.attack in ("spoofing", "data_alteration")):
        fp["partner"] += 1
    elif a["attack"] == "unauthorised_access" and a["device"].startswith("PMP"):
        fp["pump"] += 1
    elif a["attack"] == "data_alteration" and a["device"].startswith("WRB"):
        fp["artefact"] += 1
    else:
        fp["other"] += 1
ctx["live_fp"] = fp
print("live false-alert categories:", fp)
ctx["text_e6"] = (
    f"We streamed the 3 h test log through the live detector window by window, and ran a new twin (seed 99, 60 min, "
    f"{live['campaigns']} campaigns) end to end. {{TABNEXT}} gives the results. In replay, {rep['detected']} of "
    f"{rep['campaigns']} campaigns are detected with a median time-to-detect of {rep['median_delay_s']} s; in the "
    f"live run, {live['detected']} of {live['campaigns']} with a median of {live['median_delay_s']} s. Exfiltration "
    "is the slow case (about 40 s): its staging phase looks like a large but ordinary EHR download, and the alert "
    "fires once records start leaving for the external host. Processing a 5 s window takes "
    f"{live['avg_window_latency_ms']:.0f} ms on average (95th percentile {live['p95_window_latency_ms']:.0f} ms), "
    f"about {5000 / live['avg_window_latency_ms']:.0f} times faster than real time. Grouping turns "
    f"{live['alerts']:,} alerts into {live['incidents']} incidents. {{FIGNEXT}} shows device risk: each attacked pump or "
    "monitor and the rogue host jumps above 80 (Critical) as soon as its first alert fires and decays once its "
    "traffic is clean, while the two peaks of PMP-01 are false alarms caused by a nurse pausing and later restarting "
    "its infusion. Of the "
    f"{live['false_alerts']} alerts on flows labelled normal, {ctx['live_fp']['iforest']} came from the Isolation "
    f"Forest safety net, {ctx['live_fp']['partner']} were wearables whose paired monitor was being spoofed (the "
    "residual is symmetric, so the right bed is flagged even though that flow carries no attack packets), "
    f"{ctx['live_fp']['pump']} were nurse pump pauses or restarts and {ctx['live_fp']['artefact']} were wearable "
    "motion artefacts (RQ5).")

ctx["text_discussion"] = (
    f"**Implications.** First, the twin is useful twice: as a data source with exact labels and controllable "
    "attacks (RQ1), and as a source of features that make integrity attacks visible (RQ3). Second, binary accuracy "
    "is a poor guide in this domain; every model exceeds 99.7% accuracy, yet one of them misses most impostor "
    "flows, which only macro F1 and per-class analysis reveal. Third, model choice matters most for rare, subtle "
    "attacks: boosting and the neural network capture weak, combined evidence that bagging dilutes. Fourth, a "
    "supervised IDS alone is closed-world [@sommer]; pairing it with an unsupervised model raises recall on unseen "
    "attacks but needs alert grouping and context to keep false positives manageable. Fifth, the twin is useful a "
    "third time: its invariants become label-free rules that catch the unseen integrity attacks that both "
    "learners miss, so the three layers together reach a recall of at least "
    f"{zd[zd['detector'] == f'Hybrid ({best} + IF + twin rules)']['recall'].min():.2f} on every unseen family. For a "
    "hospital, the risk score turns detections into priorities: a compromised pump outranks a scanned workstation.")
ctx["text_limits"] = (
    "**Limitations.** The data is simulated: real devices, protocols and attackers are more varied, and our "
    "benign confounders cover only some sources of noise, so absolute scores are optimistic. Flow-level labels "
    "can disagree with bed-level reality, as the paired-sensor alerts show. Nurse pump pauses still trigger "
    "unauthorised-access alerts because the pump-side flow does not reveal who sent the command. The twin rules "
    "encode invariants of our simulated ward; real devices can break them for benign reasons, so their thresholds "
    "would need calibrating on each ward's own traffic. Some test cells (high-intensity spoofing) are empty, and we "
    "could not yet run the WUSTL-EHMS-2020 benchmark in our environment; the script is provided.")

ctx["text_conclusion"] = (
    f"We built MedTwin-IDS, a digital twin of an eight-bed ward that generates exactly labelled medical IoT traffic "
    f"and supplies consistency features for detection. Among Random Forest, XGBoost and an MLP, {best} gives the "
    f"best balance: macro F1-score {f3(xg['macro_f1'])} and binary F1-score {f4(xg['f1'])} on an independent run, "
    f"with {xg['latency_ms']:.2f} ms per flow; it ties with the MLP on that run but is more stable across folds and "
    "best on three extra wards. Twin features are what make spoofing and data alteration detectable, an Isolation "
    "Forest and label-free twin rules together extend coverage to unseen attacks (mean recall "
    f"{zmean.loc[f'Hybrid ({best} + IF + twin rules)', 'recall']:.2f}), and the streaming detector finds every test "
    "campaign within seconds while ranking devices by risk.")
ctx["text_future"] = (
    "Future work: (i) validate on real captures and on WUSTL-EHMS-2020 [@hady] and other public sets; (ii) pass the "
    "command source and clinical orders into the pump's features to remove false pump alerts; (iii) replace the "
    "Isolation Forest with an online sequence model such as an autoencoder ensemble [@mirsky] and calibrate the twin "
    "rules per device to cut their false alerts; (iv) correlate alerts at bed level; and (v) close "
    "the loop with automatic, clinician-approved isolation of high-risk devices.")

# claims made in the abstract and conclusion that depend on the robustness runs
_ex = ctx["extra"].pivot(index="seed", columns="model", values="macro_f1")
assert (_ex.idxmax(axis=1) == best).all(), "abstract says the selected model is best on every extra ward"
assert ctx["boot"].loc["Random Forest", "diff_vs_best_low"] > 0, "conclusion says the lead over RF is significant"
assert ctx["boot"].loc["MLP", "diff_vs_best_low"] < 0 < ctx["boot"].loc["MLP", "diff_vs_best_high"], "tie with MLP"

# ------------------------------------------------------------------ build
d = IEEEDoc(str(TEMPLATE), footer_text=FOOTER)
d.title(TITLE, ["ICT3214 Security Analytics, Coursework 2: Final Report", f"Group ID: {GROUP_ID}"])
d.authors(MEMBERS)
d.abstract(
    "Connected patient monitors, infusion pumps and wearables make hospital wards vulnerable to cyber attacks that "
    "can directly harm patients, yet public medical IoT datasets are too small and fixed to test detectors on "
    "unseen attacks or in real time. We present MedTwin-IDS, which builds a digital twin of an eight-bed ward to "
    "generate exactly labelled packet-level data with realistic clinical and network anomalies and six attack "
    "families at three intensities, and uses the twin to derive consistency features such as cross-sensor "
    "physiological residuals and message sequence integrity. Random Forest, XGBoost and a multilayer perceptron are "
    f"compared under a leakage-aware protocol. {best}, selected by cross-validation, reaches "
    f"{pct(xg['accuracy'], 2)} accuracy, a binary F1-score of {f4(xg['f1'])} and a macro F1-score of "
    f"{f3(xg['macro_f1'])} on an independent test run at {xg['latency_ms']:.2f} ms per flow, against "
    f"{f3(rf['macro_f1'])} for Random Forest, and is the best model on three further unseen wards. Twin features "
    f"raise macro F1 from {ab.loc[('network_biometric', best), 'macro_f1']:.3f} to "
    f"{ab.loc[('full_twin', best), 'macro_f1']:.3f} and spoofing recall from "
    f"{pct(ab.loc[('network_biometric', best), 'recall_spoofing'], 0)} to "
    f"{pct(ab.loc[('full_twin', best), 'recall_spoofing'], 0)}. On attack families never seen in training, an "
    f"Isolation Forest raises mean recall from {zmean.loc[best, 'recall']:.2f} to {zmean.loc[hyb, 'recall']:.2f}, "
    f"and label-free twin consistency rules raise it to {zmean.loc[f'Hybrid ({best} + IF + twin rules)', 'recall']:.2f}. "
    f"In streaming operation, all {rep['campaigns']} test campaigns are detected with a median delay of "
    f"{rep['median_delay_s']} s, and per-device risk scores and explanations are shown on an analyst dashboard.")
d.keywords("medical IoT, digital twin, intrusion detection, XGBoost, explainable AI, real-time detection, risk scoring")
related_and_intro(d, C, ctx)
methods(d, C, ctx)
results(d, C, ctx)
conclusion(d, C, ctx)
d.references(cit.reference_list())
d.one_column()
appendix(d, C, ctx)
OUT.mkdir(parents=True, exist_ok=True)
d.save(str(OUT / "ICT3214_CW2_Final_Report.docx"))
finalise(OUT / "ICT3214_CW2_Final_Report.docx", TITLE, "ICT3214 Security Analytics, Coursework 2: Final Report")
print("saved report with", len(cit.order), "references")
