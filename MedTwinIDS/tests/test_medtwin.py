"""Automated tests for MedTwin-IDS. Run from the MedTwinIDS folder with: python -m pytest -q"""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from medtwin import config as C
from medtwin.attacks import campaign_table
from medtwin.models import MODEL_NAMES, fit_model, full_proba, make_model
from medtwin.parser import FeatureExtractor, clean_packets, packets_to_features
from medtwin.risk import RiskEngine
from medtwin.rules import TwinRules
from medtwin.simulate import run_simulation
from medtwin.twin import PacketBuffer, WardTwin, build_registry

REG = build_registry()
MAC = dict(zip(REG.device_id, REG.mac))
IP = dict(zip(REG.device_id, REG.ip))
VITALS = (36.8, 97.0, 80.0, 120.0, 80.0, 80.0, 16.0, 0.0)


@pytest.fixture(scope="module")
def small_run():
    """15 minutes of ward time with a high attack rate, so every family appears."""
    return run_simulation(C.ScenarioConfig(seed=21, hours=0.25, start_hour=9, campaigns_per_hour=24), verbose=False)


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
    assert set(attack["attack"]) <= set(C.ATTACKS)
    assert (attack["campaign"] >= 0).all()
    assert (packets.loc[packets["label"] == 0, "attack"] == C.NORMAL).all()


@pytest.mark.parametrize("minutes", [3, 8, 20])
def test_short_runs_keep_campaigns_inside_the_run(minutes):
    tw = WardTwin(C.ScenarioConfig(seed=7, hours=minutes / 60, campaigns_per_hour=12))
    camps = campaign_table(tw.campaigns)
    assert len(camps) >= 1
    assert (camps["end"] <= tw.duration).all() and (camps["duration"] > 0).all()


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


def test_features_are_complete_and_identifier_free(small_run):
    tw, packets = small_run
    flows, _ = packets_to_features(packets, tw.registry)
    assert len(C.ALL_FEATURES) == 55 == len(set(C.ALL_FEATURES))
    assert not flows[C.ALL_FEATURES].isna().any().any()
    identifiers = {"src_ip", "dst_ip", "init_ip", "resp_ip", "sport", "dport", "svc_port", "device_id", "src_mac",
                   "ts", "window", "t_start", "campaign"}
    assert identifiers.isdisjoint(C.ALL_FEATURES)
    assert flows["label"].sum() > 0


def test_streaming_extraction_matches_batch(small_run):
    """The live detector must see exactly the features the models were trained on."""
    tw, packets = small_run
    clean, _ = clean_packets(packets)
    batch = FeatureExtractor(tw.registry).transform(clean)
    fx, parts, carry = FeatureExtractor(tw.registry), [], None
    for t0 in np.arange(0, tw.duration, C.WINDOW_SECONDS):
        chunk = clean[(clean["ts"] >= t0) & (clean["ts"] < t0 + C.WINDOW_SECONDS)]
        if carry is not None:
            chunk = pd.concat([carry, chunk])
        carry, chunk = chunk[chunk["ts"] >= t0 + C.WINDOW_SECONDS], chunk[chunk["ts"] < t0 + C.WINDOW_SECONDS]
        if len(chunk):
            parts.append(fx.transform(chunk))
    stream = pd.concat(parts, ignore_index=True)
    key = ["window", "init_ip", "resp_ip", "svc_port", "proto"]
    merged = batch.merge(stream, on=key, suffixes=("_b", "_s"), how="outer", indicator=True)
    assert (merged["_merge"] == "both").all()
    for f in C.ALL_FEATURES:
        np.testing.assert_allclose(merged[f + "_b"], merged[f + "_s"], atol=1e-6, err_msg=f)
    assert (merged["attack_b"] == merged["attack_s"]).all()


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
