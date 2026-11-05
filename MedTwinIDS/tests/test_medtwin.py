"""Automated tests for MedTwin-IDS. Run from the MedTwinIDS folder with: python -m pytest -q"""
import io
import json
import pickle
from argparse import Namespace
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from medtwin import config as C
from medtwin.attacks import WARD_SUBNET, campaign_table
from medtwin.models import MODEL_NAMES, PortableXGBClassifier, fit_model, full_proba, make_model
from medtwin.parser import FeatureExtractor, clean_packets, packets_to_features
from medtwin.risk import Explainer, RiskEngine
from medtwin.rules import TwinRules
from medtwin.simulate import run_simulation
from medtwin.train import campaign_blocks, campaign_safe_cut, time_blocks
from medtwin.twin import PacketBuffer, WardTwin, build_registry

REG = build_registry()
MAC = dict(zip(REG.device_id, REG.mac))
IP = dict(zip(REG.device_id, REG.ip))
VITALS = (36.8, 97.0, 80.0, 120.0, 80.0, 80.0, 16.0, 0.0)


@pytest.fixture(scope="module")
def small_run():
    """15 minutes of ward time with a high attack rate, so every family appears."""
    return run_simulation(C.ScenarioConfig(seed=21, hours=0.25, start_hour=9, campaigns_per_hour=24), verbose=False)


@pytest.fixture(scope="module")
def small_flows(small_run):
    """Batch flow features of the small run (as written to data/*_flows.csv)."""
    tw, packets = small_run
    return packets_to_features(packets, tw.registry)[0]


# ------------------------------------------------------------------ digital twin
def test_twin_is_deterministic():
    a = run_simulation(C.ScenarioConfig(seed=5, hours=0.05), verbose=False)[1]
    b = run_simulation(C.ScenarioConfig(seed=5, hours=0.05), verbose=False)[1]
    pd.testing.assert_frame_equal(a, b)


def test_every_family_is_scheduled_and_labelled(small_run):
    tw, packets = small_run
    camps = campaign_table(tw.campaigns)
    assert set(camps["attack"]) == set(C.ATTACKS)
    assert (camps["end"] <= tw.duration).all() and (camps["duration"] > 0).all()
    attack = packets[packets["label"] == 1]
    assert set(attack["attack"]) == set(C.ATTACKS)
    assert set(attack["campaign"]) == set(camps["campaign"])  # every campaign leaves labelled packets, none unlisted
    benign = packets[packets["label"] == 0]
    assert (benign["attack"] == C.NORMAL).all() and (benign["campaign"] == -1).all()


@pytest.mark.parametrize("minutes", [3, 8, 20])
def test_short_runs_keep_campaigns_inside_the_run(minutes):
    tw = WardTwin(C.ScenarioConfig(seed=7, hours=minutes / 60, campaigns_per_hour=12))
    camps = campaign_table(tw.campaigns)
    assert len(camps) >= 1
    assert (camps["end"] <= tw.duration).all() and (camps["duration"] > 0).all()


def test_campaign_table_lists_every_source(small_run):
    tw, packets = small_run
    camps = campaign_table(tw.campaigns).set_index("campaign")
    dos = [c for c in tw.campaigns if c["type"] == "dos_flood"]
    assert any(len(c["sources"]) > 1 for c in dos)  # seed 21 floods the gateway from three monitors
    for c in dos:
        ips = camps.loc[c["id"], "source_ip"].split(";")
        assert ips == [s["ip"] for s in c["sources"]]
        assert camps.loc[c["id"], "source_kind"].split(";") == [s["kind"] for s in c["sources"]]
        sent = packets[(packets["campaign"] == c["id"]) & (packets["src_ip"] != C.BROKER_IP)]
        assert set(sent["src_ip"]) == set(ips)  # the table names exactly the hosts seen flooding
    assert (camps.loc[camps["attack"] == "reconnaissance", "target"] == WARD_SUBNET).all()


def test_forged_arp_frames_match_benign_sizes(small_run):
    _, packets = small_run
    arp = packets[packets["proto"] == "ARP"]
    for from_gw in (True, False):  # gateway -> device (42 bytes) and padded device -> gateway (60 bytes)
        d = arp[(arp["src_ip"] == C.BROKER_IP) == from_gw]
        forged, benign = set(d.loc[d["label"] == 1, "length"]), set(d.loc[d["label"] == 0, "length"])
        assert forged and forged == benign, (from_gw, forged, benign)


def test_tampered_pump_is_restored_and_medium_setpoints_stay_below_high():
    # seed 6: three medium campaigns (bed 4 twice) whose restores all fall inside the run
    cfg = C.ScenarioConfig(seed=6, hours=0.25, campaigns_per_hour=12, attack_types=["unauthorised_access"],
                           intensities=("medium",))
    tw, W = WardTwin(cfg), C.WINDOW_SECONDS
    restores = {e["campaign"]: e for e in tw.benign_events if e["kind"] == "nurse_pump_restore"}
    assert sorted(restores) == [c["id"] for c in tw.campaigns]  # one restore per campaign
    frames, rate_after = [], {}
    for t0 in np.arange(0, tw.duration, W):
        t1 = min(t0 + W, tw.duration)
        frames.append(tw.step(t0, t1))
        rate_after |= {cid: tw.pump_rate[e["bed"]] for cid, e in restores.items() if t0 <= e["start"] < t1}
    packets = pd.concat(frames, ignore_index=True)
    assert rate_after == {c["id"]: c["base_rate"] for c in tw.campaigns}  # back at the pre-attack rate
    cmds = packets[packets["mqtt"] == "CMD"]
    assert (cmds.loc[cmds["label"] == 1, "pump_rate"] < 400).all()  # medium never reaches the high band
    for c in tw.campaigns:
        e = restores[c["id"]]
        assert c["end"] < e["start"] < tw.duration
        r = cmds[(cmds["ts"] >= e["start"]) & (cmds["ts"] < e["start"] + 0.1)
                 & (cmds["device_id"] == f"PMP-{c['bed']:02d}")]
        assert len(r) >= 2  # nurse station -> gateway and gateway -> pump
        assert (r["pump_rate"] == c["base_rate"]).all() and (r["label"] == 0).all()
        assert (r["attack"] == C.NORMAL).all() and (r["campaign"] == -1).all()
        assert r["src_ip"].iloc[0] == C.NURSE_STATION_IP


def test_takeover_victim_telemetry_is_labelled_spoofing():
    # seed 1: three client-ID takeovers (MON-01 twice, WRB-01)
    tw, packets = run_simulation(C.ScenarioConfig(seed=1, hours=0.25, campaigns_per_hour=12, attack_types=["spoofing"]),
                                 verbose=False)
    camps = [c for c in tw.campaigns if c["variant"] == "takeover"]
    assert camps
    flows, _ = packets_to_features(packets, tw.registry)
    W = C.WINDOW_SECONDS
    for c in camps:
        v = tw.by_id[c["victim"]]
        own = packets[(packets["src_ip"] == v.ip) & (packets["mqtt"] == "PUBLISH") & (packets["retrans"] == 0)]
        during = own[(own["ts"] >= c["start"]) & (own["ts"] < c["end"])]
        assert len(during) and (during["label"] == 1).all()
        assert (during["attack"] == "spoofing").all() and (during["campaign"] == c["id"]).all()
        acks = packets[(packets["dst_ip"] == v.ip) & (packets["mqtt"] == "PUBACK")
                       & (packets["ts"] >= c["start"] + 0.05) & (packets["ts"] < c["end"])]
        assert len(acks) and (acks["label"] == 1).all()
        inside = flows[(flows["init_ip"] == v.ip) & (flows["svc_port"] == C.MQTT_PORT)
                       & (flows["t_start"] >= c["start"]) & (flows["t_start"] + W <= c["end"])]
        assert len(inside) and (inside["attack"] == "spoofing").all()
        # outside every campaign on this device its own telemetry stays normal
        on_victim = [x for x in camps if x["victim"] == c["victim"]]
        out = own[~np.any([(own["ts"] >= x["start"]) & (own["ts"] < x["end"]) for x in on_victim], axis=0)]
        assert len(out) and (out["label"] == 0).all()


# ------------------------------------------------------------------ cleaner and parser
def _telemetry(buf, ts, vitals=VITALS, seq=1, dst=C.BROKER_IP):
    buf.add(ts, IP["MON-01"], dst, MAC["MON-01"], 40000, C.MQTT_PORT, "TCP", "PA", 190, 64, mqtt="PUBLISH",
            device_id="MON-01", seq=seq, vitals=vitals)


def test_cleaner_handles_capture_artefacts():
    buf = PacketBuffer()
    _telemetry(buf, 1.0)
    _telemetry(buf, 1.0)                                                 # duplicated capture frame
    _telemetry(buf, 2.0, vitals=VITALS[:5] + (65535.0,) + VITALS[6:])   # sensor sentinel
    _telemetry(buf, 3.0, vitals=VITALS[:5] + (25.0,) + VITALS[6:])      # extreme but measurable
    _telemetry(buf, 4.0, dst=None)                                       # truncated record
    clean, report = clean_packets(buf.to_frame())
    assert report["dropped_exact_duplicates"] == 1
    assert report["dropped_missing_essential"] == 1
    assert report["vital_glitches_nulled"] == 1
    assert 25.0 in clean["hr"].to_numpy()  # evidence of tampering must not be cleaned away


def test_scan_to_high_port_is_oriented_by_the_syn():
    buf = PacketBuffer()
    buf.add(1.0, C.ROGUE_HOST_IP, IP["MON-01"], "02:66:66:00:00:01", 40000, 50000, "TCP", "S", 60, 64)
    buf.add(1.001, IP["MON-01"], C.ROGUE_HOST_IP, MAC["MON-01"], 50000, 40000, "TCP", "RA", 54, 64)
    flows = FeatureExtractor(REG).transform(clean_packets(buf.to_frame())[0])
    assert len(flows) == 1
    row = flows.iloc[0]
    assert row["init_ip"] == C.ROGUE_HOST_IP and row["SynCnt"] == 1 and row["SrcKnown"] == 0


def test_features_are_complete_and_identifier_free(small_flows):
    flows = small_flows
    assert len(C.ALL_FEATURES) == 55 == len(set(C.ALL_FEATURES))
    assert not flows[C.ALL_FEATURES].isna().any().any()
    identifiers = {"src_ip", "dst_ip", "init_ip", "resp_ip", "sport", "dport", "svc_port", "device_id", "src_mac",
                   "ts", "window", "t_start", "campaign"}
    assert identifiers.isdisjoint(C.ALL_FEATURES)
    assert flows["label"].sum() > 0


def test_streaming_extraction_matches_batch(small_run, small_flows):
    """The live detector must see exactly the features the models were trained on. The streamed side
    repeats the loop of realtime.run: a fresh twin stepped one window at a time, packets stamped past
    the window end carried into the next window, and cleaning and extraction per chunk."""
    tw0, packets = small_run
    tw, W = WardTwin(tw0.cfg), C.WINDOW_SECONDS
    fx, parts, carry, carried, seen, t = FeatureExtractor(tw.registry, W), [], None, 0, 0, 0.0
    while t < tw.duration:
        t1 = min(t + W, tw.duration)
        pk = tw.step(t, t1)
        if carry is not None and len(carry):
            pk = pd.concat([carry, pk], ignore_index=True)
        carry, pk = pk[pk["ts"] >= t1], pk[pk["ts"] < t1]
        carried += len(carry)
        seen += len(pk)
        clean, _ = clean_packets(pk) if len(pk) else (pk, {})
        if len(clean):
            parts.append(fx.transform(clean))
        t = t1
    assert carried > 0  # late packets (acks, jittered telemetry) did cross window ends
    assert seen == (packets["ts"] < tw.duration).sum()  # the fresh twin replays the fixture packet for packet
    stream = pd.concat(parts, ignore_index=True)
    batch = small_flows[small_flows["t_start"] < tw.duration]  # the live loop stops at the end of the run
    key = ["window", "init_ip", "resp_ip", "svc_port", "proto"]
    merged = batch.merge(stream, on=key, suffixes=("_b", "_s"), how="outer", indicator=True)
    assert (merged["_merge"] == "both").all()
    for f in C.ALL_FEATURES:
        np.testing.assert_allclose(merged[f + "_b"], merged[f + "_s"], atol=1e-6, err_msg=f)
    assert (merged["attack_b"] == merged["attack_s"]).all() and (merged["campaign_b"] == merged["campaign_s"]).all()


def test_vitals_delta_spans_a_window_without_telemetry():
    buf = PacketBuffer()
    _telemetry(buf, 1.0, seq=1)                                                    # HR 80
    buf.add(6.0, IP["MON-01"], C.BROKER_IP, MAC["MON-01"], 40000, C.MQTT_PORT, "TCP", "PA", 90, 64,
            mqtt="CONNECT")                                                       # same flow, no vitals
    _telemetry(buf, 11.0, vitals=VITALS[:5] + (100.0,) + VITALS[6:], seq=2)     # HR 100
    clean = clean_packets(buf.to_frame())[0]
    batch = FeatureExtractor(REG).transform(clean)
    assert batch["VitalsDelta"].tolist() == [0.0, 0.0, 20.0]  # measured against the last reading, not zeroed
    for cuts in ([0, 5, 10, 15], [0, 10, 15], [0, 5, 15]):  # one window per chunk, then chunks of two windows
        fx = FeatureExtractor(REG)
        stream = pd.concat([fx.transform(clean[(clean["ts"] >= a) & (clean["ts"] < b)]) for a, b in zip(cuts, cuts[1:])],
                           ignore_index=True)
        pd.testing.assert_frame_equal(stream[C.ALL_FEATURES], batch[C.ALL_FEATURES])


def test_rogue_pump_command_raises_pump_rate_delta():
    buf = PacketBuffer()
    pump = "PMP-01"
    buf.add(1.0, IP[pump], C.BROKER_IP, MAC[pump], 41000, C.MQTT_PORT, "TCP", "PA", 120, 255, mqtt="PUBLISH",
            device_id=pump, seq=1, pump_rate=50.0)
    buf.add(6.0, C.BROKER_IP, IP[pump], MAC["GW-01"], C.MQTT_PORT, 41000, "TCP", "PA", 110, 64, mqtt="CMD",
            device_id=pump, pump_rate=500.0)
    flows = FeatureExtractor(REG).transform(clean_packets(buf.to_frame())[0])
    cmd = flows[flows["MqttCmd"] > 0].iloc[0]
    assert cmd["PumpRateDelta"] == pytest.approx(9.0)  # |500 - 50| / 50


# ------------------------------------------------------------------ rules, models, risk
def test_twin_rules_fire_on_violations_only():
    base = {f: 0.0 for f in C.ALL_FEATURES} | {"SrcKnown": 1.0, "attack": C.NORMAL}
    normal = pd.DataFrame([base] * 1000)
    rules = TwinRules().fit(normal)
    probes = pd.DataFrame([base,
                           base | {"SrcKnown": 0.0},
                           base | {"MacMismatch": 1.0},
                           base | {"SeqDup": 3.0},
                           base | {"TwinResidualHR": 30.0},
                           base | {"PumpRateDelta": 4.0}])
    hits = rules.evaluate(probes)
    assert not hits.iloc[0].any()
    assert [hits.columns[hits.iloc[i]].tolist() for i in range(1, 6)] == [["R1"], ["R2"], ["R3"], ["R4"], ["R5"]]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_train_on_a_subset_of_classes(name):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 2, 5], 120)  # deliberately non-contiguous labels (as in the zero-day experiment)
    X = rng.normal(size=(len(y), 6)).astype(np.float32) + y[:, None]
    model = fit_model(name, make_model(name, fast=True, n_jobs=1) if name != "MLP" else make_model(name, fast=True), X, y)
    proba = full_proba(model, X)
    assert proba.shape == (len(y), len(C.CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert proba[:, [1, 3, 4, 6]].max() == 0.0
    assert (proba.argmax(1) == y).mean() > 0.9


def test_xgboost_model_pickles_portably():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(len(C.CLASSES)), 40)
    X = rng.normal(size=(len(y), 6)).astype(np.float32) + y[:, None]
    model = fit_model("XGBoost", make_model("XGBoost", fast=True, n_jobs=1, n_estimators=20), X, y)
    assert isinstance(model, PortableXGBClassifier)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    raw = buf.getvalue()
    # the booster's memory snapshot (with its platform-specific RNG state) must not be pickled
    assert b"rng_state" not in raw and b"rng_state" not in pickle.dumps(model)
    loaded = joblib.load(io.BytesIO(raw))
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(full_proba(pickle.loads(pickle.dumps(model)), X), full_proba(model, X))


def test_cv_groups_never_split_a_campaign(small_flows):
    flows = small_flows
    att = (flows["campaign"] >= 0).to_numpy()
    # 1-minute blocks, so the 100 s campaigns of the small run cross block boundaries
    g = pd.DataFrame(dict(campaign=flows["campaign"], group=campaign_blocks(flows, 1.0), block=time_blocks(flows, 1.0)))
    per = g[att].groupby("campaign")[["group", "block"]].nunique()
    assert (per["group"] == 1).all()
    assert (per["block"] > 1).any()  # plain time blocks would have split them
    assert (g.loc[~att, "group"] == g.loc[~att, "block"]).all()  # normal flows keep their own block
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, va in cv.split(flows, flows["attack"], g["group"]):
        fit_c, val_c = set(flows["campaign"].iloc[tr]) - {-1}, set(flows["campaign"].iloc[va]) - {-1}
        assert fit_c.isdisjoint(val_c)


def test_tuning_cut_keeps_campaigns_whole():
    df = pd.DataFrame(dict(t_start=[0.0, 50.0, 100.0, 150.0, 200.0, 300.0], campaign=[0, 1, 0, 1, 1, -1]))
    cut = campaign_safe_cut(df, 75.0)  # splits campaign 0 and, once moved, overlapping campaign 1
    assert 200.0 < cut <= 300.0
    assert campaign_safe_cut(df, 250.0) == 250.0  # nothing to split: unchanged


def _flows(rows):
    df = pd.DataFrame([{f: 0.0 for f in C.ALL_FEATURES} | {"svc_port": 0, "proto": "TCP", "device_id": "",
                                                            "attack": C.NORMAL} | r for r in rows])
    return df


def test_risk_engine_protects_impersonated_devices_and_tracks_rogue_hosts():
    engine = RiskEngine(REG)
    poison = {"init_ip": C.BROKER_IP, "resp_ip": IP["WRB-01"], "MacMismatch": 1.0}   # ARP reply claiming the gateway
    flood = {"init_ip": C.ROGUE_HOST_IP, "resp_ip": C.BROKER_IP, "svc_port": C.MQTT_PORT}
    proba = np.zeros((2, len(C.CLASSES)))
    proba[0, C.CLASSES.index("data_alteration")] = 1.0
    proba[1, C.CLASSES.index("dos_flood")] = 1.0
    alerts = engine.update(_flows([poison, flood]), proba, None, 5.0, "00:00:05")
    assert len(alerts) == 2
    assert engine.risk["WRB-01"] >= 60                     # the poisoned device
    assert engine.risk[f"UNKNOWN {C.ROGUE_HOST_IP}"] >= 60  # unregistered attacker is tracked
    alone = RiskEngine(REG)
    alone.update(_flows([poison]), proba[:1], None, 5.0, "00:00:05")
    assert alone.risk["GW-01"] == 0.0                      # impersonated gateway is not blamed


def test_risk_engine_groups_alerts_into_incidents():
    engine = RiskEngine(REG)
    flood = _flows([{"init_ip": C.ROGUE_HOST_IP, "resp_ip": C.BROKER_IP, "svc_port": C.MQTT_PORT}])
    for t, cls in [(5.0, "dos_flood"), (35.0, "dos_flood"), (60.0, "dos_flood"),  # gaps of 30 s or less
                   (95.0, "dos_flood"),                                          # 35 s of silence
                   (100.0, "reconnaissance"),                                    # same host, other attack
                   (105.0, "dos_flood")]:                                        # back to the open flood
        proba = np.zeros((1, len(C.CLASSES)))
        proba[0, C.CLASSES.index(cls)] = 1.0
        engine.update(flood, proba, None, t, f"t={t}")
    got = [(i["id"], i["attack"], i["first_t"], i["last_t"], i["alerts"]) for i in engine.incidents]
    assert got == [(1, "dos_flood", 5.0, 60.0, 3), (2, "dos_flood", 95.0, 105.0, 2), (3, "reconnaissance", 100.0, 100.0, 1)]
    assert {i["device"] for i in engine.incidents} == {f"UNKNOWN {C.ROGUE_HOST_IP}"}


def test_explainer_cites_identity_evidence():
    stats = pd.DataFrame({"mean": [1.0, 0.0, 0.2, 10.0, 0.5], "std": [0.0, 0.0, 0.5, 4.0, 0.5]},
                         index=["SrcKnown", "TTLDev", "SynCnt", "Rate", "HasVitals"])
    ex = Explainer(stats, {"SynCnt": 1.0, "Rate": 1.0}, top_k=2)
    assert sorted(ex.constant) == ["SrcKnown", "TTLDev"]  # zero spread in normal traffic
    base = {"SrcKnown": 1.0, "TTLDev": 0.0, "SynCnt": 1.2, "Rate": 14.0, "HasVitals": 1.0}

    def cited(**change):
        return [e["feature"] for e in ex.explain(pd.Series(base | change))]

    assert cited() == ["SynCnt", "Rate"]                     # constant features at their normal value stay quiet
    assert cited(SrcKnown=0.0) == ["SrcKnown", "SynCnt"]     # unregistered sender
    assert cited(TTLDev=64.0) == ["TTLDev", "SynCnt"]        # TTL does not match the registry


def test_benchmark_loader_matches_wustl_schema(tmp_path):
    from medtwin.benchmark_wustl import load
    net = ["Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "Dport", "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
           "SrcGap", "DstGap", "SIntPkt", "DIntPkt", "SIntPktAct", "DIntPktAct", "SrcJitter", "DstJitter",
           "sMaxPktSz", "dMaxPktSz", "sMinPktSz", "dMinPktSz", "Dur", "Trans", "TotPkts", "TotBytes", "Load",
           "Loss", "pLoss", "pSrcLoss", "pDstLoss", "Rate", "SrcMac", "DstMac", "Packet_num"]
    bio = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.random((50, len(net) + len(bio))), columns=net + bio)
    df[["Dir", "Flgs", "SrcAddr", "SrcMac"]] = "x"
    df["Attack Category"], df["Label"] = "normal", rng.integers(0, 2, 50)
    path = tmp_path / "wustl.csv"
    df.to_csv(path, index=False)
    X, y, sets = load(str(path))
    assert len(sets["biometric_only"]) == 8
    assert len(sets["network_only"]) == 26  # 35 network columns minus 9 identifier columns
    assert not {"SrcAddr", "Sport", "SrcMac", "Packet_num", "Label"} & set(X.columns)


# ------------------------------------------------------------------ end to end
def test_live_summary_handles_no_campaigns_and_measures_delay(tmp_path, monkeypatch):
    from medtwin import realtime
    monkeypatch.setattr(C, "RUNTIME_DIR", tmp_path)
    engine = SimpleNamespace(incidents=[])
    truth = pd.DataFrame(dict(campaign=[0, 1], attack=["spoofing", "dos_flood"], intensity=["low", "high"],
                              start=[100.0, 300.0], end=[200.0, 400.0]))
    # stopped before the first campaign (e.g. --stop-at 1): nothing to score, but the outputs are still written
    s = realtime.summarise(engine, truth[truth["start"] < 60.0], [], 0, 1.0, 5.0)
    assert s["per_campaign"] == [] and s["overall"]["campaigns"] == 0 and s["overall"]["detected"] == 0
    assert s["overall"]["detection_rate"] is None and s["overall"]["mean_delay_s"] is None
    assert s["overall"]["avg_window_latency_ms"] is None
    assert list(pd.read_csv(tmp_path / "live_campaigns.csv").columns) == realtime.CAMPAIGN_COLS
    json.dumps(s)
    alerts = [dict(t=50.0, attack="dos_flood", truth=C.NORMAL),            # false alert
              dict(t=112.0, attack="data_alteration", truth="spoofing"),   # first hit, wrong class
              dict(t=115.0, attack="spoofing", truth="spoofing")]
    (tmp_path / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in alerts))
    s = realtime.summarise(engine, truth, [2.0, 4.0], 10, 1.0, 5.0)
    per = {r["campaign"]: r for r in s["per_campaign"]}
    assert per[0]["detected"] and per[0]["delay_s"] == 12.0 and per[0]["alerts"] == 2 and per[0]["class_correct"] == 0.5
    assert not per[1]["detected"] and per[1]["delay_s"] is None
    o = s["overall"]
    assert (o["campaigns"], o["detected"], o["detection_rate"]) == (2, 1, 0.5)
    assert o["mean_delay_s"] == o["median_delay_s"] == 12.0
    assert (o["alerts"], o["false_alerts"], o["avg_window_latency_ms"]) == (3, 1, 3.0)
    assert len(pd.read_csv(tmp_path / "live_campaigns.csv")) == 2


def test_realtime_detector_end_to_end(tmp_path, monkeypatch):
    if not (C.MODEL_DIR / "model_card.json").exists():
        pytest.skip("train the models first (python -m medtwin.train)")
    from medtwin import realtime
    monkeypatch.setattr(C, "RUNTIME_DIR", tmp_path)
    args = Namespace(replay=None, minutes=8.0, speed=0.0, seed=7, start_hour=14.0, campaigns_per_hour=12.0,
                     hybrid=True, twin_rules=True, stop_at=None, save_as=None, quiet=True)
    summary = realtime.run(args)
    assert summary["overall"]["campaigns"] >= 1
    assert summary["overall"]["detection_rate"] == 1.0
    for name in ("state.json", "alerts.jsonl", "timeline.csv", "ground_truth.csv", "live_summary.json"):
        assert (tmp_path / name).exists()
