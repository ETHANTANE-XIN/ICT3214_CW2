"""Attack campaign scheduler for the ward digital twin.

Each campaign is a labelled, synthetic traffic pattern written into the packet
log. Nothing here interacts with a real network; the module only produces
records that look like the network footprint of each threat so that detectors
can be trained and evaluated safely.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

INTENSITY_PARAMS = {
    "dos_flood": {"low": 25.0, "medium": 80.0, "high": 250.0},          # packets/s
    "spoofing": {"low": 1.0, "medium": 2.0, "high": 4.0},               # publish-rate multiplier
    "data_alteration": {"low": 0.2, "medium": 1.0, "high": 3.0},        # ARP replies/s
    "unauthorised_access": {"low": 0.5, "medium": 2.0, "high": 8.0},    # login attempts/s
    "reconnaissance": {"low": 0.5, "medium": 4.0, "high": 20.0},        # probes/s
    "exfiltration": {"low": 15e3, "medium": 1.2e5, "high": 4e5},        # bytes/s
}
HR_OFFSET = {"low": (6, 10), "medium": (15, 25), "high": (40, 60)}
SPO2_OFFSET = {"low": (1, 2), "medium": (3, 5), "high": (8, 12)}
SCAN_PORTS = [21, 22, 23, 80, 104, 161, 443, 502, 1883, 2575, 3389, 5900, 8080, 8883]
UNUSED_IPS = [f"10.10.1.{i}" for i in range(100, 120)] + [f"10.10.2.{i}" for i in range(100, 110)]


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------
def _rogue(rng):
    return dict(ip=C.ROGUE_HOST_IP, mac="02:66:66:00:00:%02x" % int(rng.integers(1, 255)),
                ttl=int(rng.choice([64, 128])), kind="rogue")


def _asset(row):
    return dict(ip=row.ip, mac=row.mac, ttl=int(row.ttl), kind=row.type, device_id=row.device_id)


def _pick(tw, kinds_probs):
    rng = tw.rng
    kinds, probs = zip(*kinds_probs)
    k = str(rng.choice(kinds, p=np.array(probs) / sum(probs)))
    if k == "rogue":
        return _rogue(rng)
    if k == "workstation":
        ws = [r for r in tw.registry.itertuples() if r.type == "workstation"]
        return _asset(ws[int(rng.integers(len(ws)))])
    if k == "gateway":
        return _asset(tw.by_id["GW-01"])
    devs = [d for d in tw.telemetry_devices if d.type != "infusion_pump"]
    return _asset(devs[int(rng.integers(len(devs)))])


def schedule_campaigns(tw) -> list[dict]:
    rng, cfg = tw.rng, tw.cfg
    types = list(cfg.attack_types)
    warmup = min(180.0, 0.2 * tw.duration)  # attack-free start so the twin settles
    n = max(1, int(round(cfg.campaigns_per_hour * cfg.hours)))
    if (tw.duration - warmup) / len(types) >= 80:  # enough time to show every family at least once
        n = max(n, len(types))
    seq = (types * (n // len(types) + 1))[:n]
    rng.shuffle(seq)
    slot = (tw.duration - warmup) / n
    camps = []
    sensors = [d for d in tw.telemetry_devices if d.type != "infusion_pump"]
    for i, typ in enumerate(seq):
        dur = float(max(10.0, min(rng.uniform(60, 240), slot - 20)))
        start = warmup + i * slot + float(rng.uniform(0, max(1.0, slot - dur - 10)))
        inten = str(rng.choice(cfg.intensities))
        c = dict(id=i, type=typ, start=start, end=start + dur, intensity=inten,
                 rate=INTENSITY_PARAMS[typ][inten], port=int(rng.integers(32768, 61000)))
        if typ == "dos_flood":
            if rng.random() < 0.4:
                c["sources"] = [_rogue(rng)]
                c["variant"] = str(rng.choice(["syn", "publish"]))
            else:
                k = int(rng.integers(1, 4))
                picks = rng.choice(len(sensors), size=k, replace=False)
                c["sources"] = [_asset(sensors[j]) for j in picks]
                c["variant"] = "publish" if rng.random() < 0.7 else "syn"
        elif typ == "spoofing":
            v = sensors[int(rng.integers(len(sensors)))]
            c["victim"] = v.device_id
            c["variant"] = "takeover" if rng.random() < 0.5 else "clone"
            c["attacker"] = _pick(tw, [("rogue", 0.6), ("workstation", 0.4)])
            c["clone_mac"] = bool(rng.random() < 0.6)
            c["clone_ttl"] = bool(rng.random() < 0.5)
            c["mode"] = "mask" if rng.random() < 0.3 else "offset"
            c["hr_off"] = float(rng.uniform(*HR_OFFSET[inten]) * rng.choice([-1, 1]))
            c["spo2_off"] = -float(rng.uniform(*SPO2_OFFSET[inten]))
            c["seq_mode"] = "sniffed" if rng.random() < 0.6 else "naive"
            c["seq_base"] = int(rng.integers(0, 60000))
            c["kick_period"] = float(rng.uniform(8, 15))
        elif typ == "data_alteration":
            k = int(rng.integers(1, 3))
            picks = rng.choice(len(sensors), size=k, replace=False)
            c["victims"] = [sensors[j].device_id for j in picks]
            c["attacker"] = _pick(tw, [("rogue", 0.5), ("workstation", 0.5)])
            c["ttl_drop"] = int(rng.random() < 0.5)
            c["delay"] = {"low": (0.001, 0.003), "medium": (0.003, 0.008), "high": (0.008, 0.02)}[inten]
            c["jitter_sd"] = {"low": 0.002, "medium": 0.005, "high": 0.01}[inten]
            c["hr_off"] = float(rng.uniform(*HR_OFFSET[inten]) * rng.choice([-1, 1]))
            c["spo2_off"] = float(rng.uniform(*SPO2_OFFSET[inten]) * rng.choice([-1, 1]))
        elif typ == "unauthorised_access":
            c["attacker"] = _pick(tw, [("rogue", 0.5), ("workstation", 0.5)])
            c["bed"] = int(rng.integers(1, C.N_BEDS + 1))
            c["success_t"] = start + 0.7 * dur
            ncmd = int(rng.integers(1, 4))
            c["cmd_times"] = sorted(float(c["success_t"] + rng.uniform(2, max(3.0, 0.28 * dur))) for _ in range(ncmd))
            c["cmd_done"] = []
        elif typ == "reconnaissance":
            c["attacker"] = _pick(tw, [("rogue", 0.4), ("workstation", 0.3), ("device", 0.3)])
        elif typ == "exfiltration":
            c["attacker"] = _pick(tw, [("workstation", 0.7), ("gateway", 0.3)])
            c["dst"] = str(rng.choice(C.EXTERNAL_IPS[:2]))
        camps.append(c)
    return camps


def campaign_table(camps) -> pd.DataFrame:
    rows = []
    for c in camps:
        who = c.get("attacker") or (c.get("sources") or [{}])[0]
        rows.append(dict(campaign=c["id"], attack=c["type"], intensity=c["intensity"], start=round(c["start"], 2),
                         end=round(c["end"], 2), duration=round(c["end"] - c["start"], 1),
                         source_ip=who.get("ip", ""), source_kind=who.get("kind", ""),
                         variant=c.get("variant", c.get("mode", "")),
                         target=c.get("victim") or ",".join(c.get("victims", [])) or (f"PMP-{c['bed']:02d}" if "bed" in c else c.get("dst", C.BROKER_IP))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Effects applied to otherwise-normal telemetry
# --------------------------------------------------------------------------
def broker_load_delay(camps) -> float:
    """Extra gateway latency suffered by legitimate traffic during flooding."""
    return sum(0.001 * c["rate"] / 50.0 for c in camps if c["type"] == "dos_flood")


def _offset_vitals(vitals, hr_off, spo2_off):
    temp, spo2, pulse, sys_, dia, hr, rr, st = vitals
    hr = round(hr + hr_off, 1)
    pulse = pulse if math.isnan(pulse) else round(pulse + hr_off, 1)
    spo2 = round(float(min(100.0, max(60.0, spo2 + spo2_off))), 1)
    return (temp, spo2, pulse, sys_, dia, hr, rr, st)


def in_path_effect(tw, camps, d, ts):
    for c in camps:
        if c["type"] == "data_alteration" and d.device_id in c["victims"] and c["start"] <= ts < c["end"]:
            return dict(
                alter=lambda v, c=c: _offset_vitals(v, c["hr_off"], c["spo2_off"]),
                ttl_drop=c["ttl_drop"], delay=float(tw.rng.uniform(*c["delay"])),
                jitter=float(abs(tw.rng.normal(0, c["jitter_sd"]))),
                attack="data_alteration", campaign=c["id"], intensity=c["intensity"])
    return None


# --------------------------------------------------------------------------
# Campaign emitters
# --------------------------------------------------------------------------
def _poisson_times(rng, a, b, rate):
    if b <= a:
        return np.array([])
    return np.sort(rng.uniform(a, b, rng.poisson(rate * (b - a))))


def emit(tw, c, t0, t1, buf, states):
    a, b = max(t0, c["start"]), min(t1, c["end"])
    if b <= a:
        return
    kw = dict(label=1, attack=c["type"], campaign=c["id"], intensity=c["intensity"])
    fn = {"dos_flood": _dos, "spoofing": _spoof, "data_alteration": _mitm, "unauthorised_access": _brute,
          "reconnaissance": _recon, "exfiltration": _exfil}[c["type"]]
    fn(tw, c, a, b, t0, t1, buf, states, kw)


def _dos(tw, c, a, b, t0, t1, buf, states, kw):
    rng = tw.rng
    gw_mac = tw.by_id["GW-01"].mac
    per_src = c["rate"] / len(c["sources"])
    for s in c["sources"]:
        port = tw.port.get(s.get("device_id", ""), c["port"])
        if c["variant"] == "publish" and s["kind"] == "rogue" and t0 <= c["start"] < t1:
            tw._emit_connect(buf, c["start"], s["ip"], s["mac"], s["ttl"], port, **kw)
        for t in _poisson_times(rng, a, b, per_src):
            t = float(t)
            if c["variant"] == "syn":
                sp = int(rng.integers(1024, 65535))
                buf.add(t, s["ip"], C.BROKER_IP, s["mac"], sp, C.MQTT_PORT, "TCP", "S", 60, s["ttl"], **kw)
                if rng.random() < 0.85:
                    buf.add(t + 0.001, C.BROKER_IP, s["ip"], gw_mac, C.MQTT_PORT, sp, "TCP", "SA", 58, 64, **kw)
            else:
                buf.add(t, s["ip"], C.BROKER_IP, s["mac"], port, C.MQTT_PORT, "TCP", "PA",
                        int(rng.integers(60, 1200)), s["ttl"], mqtt="PUBLISH", **kw)
                if rng.random() < 0.6:
                    buf.add(t + 0.002 + rng.exponential(0.003), C.BROKER_IP, s["ip"], gw_mac, C.MQTT_PORT, port,
                            "TCP", "PA", 58, 64, mqtt="PUBACK", **kw)


def _mask_vitals(tw, d, rng):
    i = d.bed - 1
    base = tw.patients.base
    hr = round(float(base["hr"][i] + rng.normal(0, 0.3)), 1)
    spo2 = round(float(min(100, base["spo2"][i] + rng.normal(0, 0.2))), 1)
    temp = round(float(base["temp"][i] + rng.normal(0, 0.02)), 2)
    rr = round(float(base["rr"][i] + rng.normal(0, 0.2)), 1)
    if d.type == "patient_monitor":
        return (temp, spo2, hr, round(float(base["sys"][i]), 1), round(float(base["dia"][i]), 1), hr, rr,
                round(float(base["st"][i]), 3))
    return (round(temp - 0.4, 2), spo2, float("nan"), float("nan"), float("nan"), hr, rr, float("nan"))


def _spoof(tw, c, a, b, t0, t1, buf, states, kw):
    rng = tw.rng
    v = tw.by_id[c["victim"]]
    att = c["attacker"]
    period = C.PUBLISH_PERIOD[v.type] / c["rate"]
    if c["variant"] == "clone":
        src_ip, mac = v.ip, (v.mac if c["clone_mac"] else att["mac"])
        ttl = v.ttl if c["clone_ttl"] else att["ttl"]
    else:
        src_ip, mac, ttl = att["ip"], att["mac"], att["ttl"]
        if t0 <= c["start"] < t1:
            tw._emit_connect(buf, c["start"], src_ip, mac, ttl, c["port"], **kw)
        # legitimate device keeps being disconnected by the client-ID takeover and reconnects
        k0 = math.ceil((a - c["start"]) / c["kick_period"])
        k1 = math.floor((b - c["start"]) / c["kick_period"] - 1e-9)
        for k in range(max(1, k0), k1 + 1):
            tk = c["start"] + k * c["kick_period"]
            tw.port[v.device_id] = int(rng.integers(32768, 61000))
            tw._emit_connect(buf, tk, v.ip, v.mac, v.ttl, tw.port[v.device_id], **kw)
    k0 = math.ceil((a - c["start"]) / period)
    k1 = math.floor((b - c["start"]) / period - 1e-9)
    gw_mac = tw.by_id["GW-01"].mac
    for k in range(max(0, k0), k1 + 1):
        t = float(c["start"] + k * period + abs(rng.normal(0, 0.01)))
        if c["mode"] == "mask":
            vit = _mask_vitals(tw, v, rng)
        else:
            vit = _offset_vitals(tw.sense(v, min(t, b - 1e-6), states, set()), c["hr_off"], c["spo2_off"])
        seq = tw.seq[v.device_id] + int(rng.integers(-2, 3)) if c["seq_mode"] == "sniffed" else c["seq_base"] + k
        length = int({"patient_monitor": 190, "wearable": 150}[v.type] + rng.integers(-6, 7))
        buf.add(t, src_ip, C.BROKER_IP, mac, c["port"], C.MQTT_PORT, "TCP", "PA", length, ttl, mqtt="PUBLISH",
                device_id=v.device_id, seq=seq, vitals=vit, **kw)
        buf.add(t + 0.002 + rng.exponential(0.001), C.BROKER_IP, src_ip, gw_mac, C.MQTT_PORT, c["port"], "TCP", "PA",
                58, 64, mqtt="PUBACK", **kw)


def _mitm(tw, c, a, b, t0, t1, buf, states, kw):
    rng = tw.rng
    att = c["attacker"]
    for t in _poisson_times(rng, a, b, c["rate"]):
        for vid in c["victims"]:
            v = tw.by_id[vid]
            buf.add(float(t), C.BROKER_IP, v.ip, att["mac"], 0, 0, "ARP", "", 42, 0, **kw)
            buf.add(float(t) + 0.001, v.ip, C.BROKER_IP, att["mac"], 0, 0, "ARP", "", 42, 0, **kw)


def _brute(tw, c, a, b, t0, t1, buf, states, kw):
    rng = tw.rng
    att = c["attacker"]
    for t in _poisson_times(rng, a, min(b, c["success_t"]), c["rate"]):
        tw._emit_connect(buf, float(t), att["ip"], att["mac"], att["ttl"], int(rng.integers(32768, 61000)), rc=5, **kw)
    if t0 <= c["success_t"] < t1:
        tw._emit_connect(buf, c["success_t"], att["ip"], att["mac"], att["ttl"], c["port"], rc=0, **kw)
    for tc in c["cmd_times"]:
        if a <= tc < b and tc not in c["cmd_done"]:
            c["cmd_done"].append(tc)
            cur = tw.pump_rate[c["bed"]]
            if c["intensity"] == "low":
                new = round(max(cur, 20.0) * float(rng.uniform(1.4, 1.8)), 1)
            elif c["intensity"] == "medium":
                new = round(max(cur, 20.0) * float(rng.uniform(2.5, 4.0)), 1)
            else:
                new = float(rng.choice([0.0, float(rng.uniform(400, 999))]))
            tw._emit_pump_command(buf, tc, c["bed"], malicious=True, src_ip=att["ip"], src_mac=att["mac"],
                                  src_ttl=att["ttl"], src_port=c["port"], new_rate=new,
                                  campaign=c["id"], intensity=c["intensity"])


def _recon(tw, c, a, b, t0, t1, buf, states, kw):
    rng = tw.rng
    att = c["attacker"]
    regs = [r for r in tw.registry.itertuples() if r.ip != att["ip"]]
    open_ports = {(C.BROKER_IP, 1883), (C.EHR_IP, 443), (C.EHR_IP, 2575), (C.UPDATE_SERVER_IP, 443)}
    open_ports |= {(r.ip, 80) for r in regs if r.type == "patient_monitor"}
    for t in _poisson_times(rng, a, b, c["rate"]):
        t = float(t)
        if rng.random() < 0.8:
            r = regs[int(rng.integers(len(regs)))]
            ip, mac, ttl = r.ip, r.mac, r.ttl
        else:
            ip, mac, ttl = UNUSED_IPS[int(rng.integers(len(UNUSED_IPS)))], None, None
        port = int(rng.choice(SCAN_PORTS)) if rng.random() < 0.7 else int(rng.integers(1, 65535))
        sp = int(rng.integers(32768, 61000))
        buf.add(t, att["ip"], ip, att["mac"], sp, port, "TCP", "S", 60, att["ttl"], **kw)
        if mac is None:
            continue
        if (ip, port) in open_ports:
            buf.add(t + 0.001, ip, att["ip"], mac, port, sp, "TCP", "SA", 58, ttl, **kw)
            buf.add(t + 0.0015, att["ip"], ip, att["mac"], sp, port, "TCP", "R", 54, att["ttl"], **kw)
        elif rng.random() < 0.85:
            buf.add(t + 0.001, ip, att["ip"], mac, port, sp, "TCP", "RA", 54, ttl, **kw)


def _exfil(tw, c, a, b, t0, t1, buf, states, kw):
    att = c["attacker"]
    dur = c["end"] - c["start"]
    if att["kind"] == "workstation":
        stage = dict(start=c["start"], end=c["start"] + 0.25 * dur)
        tw._emit_stream(buf, stage, t0, t1, src=att["ip"], src_mac=att["mac"], src_ttl=att["ttl"], dst=C.EHR_IP,
                        dport=C.HTTPS_PORT, sport=c["port"] + 1, rate_bytes=3 * c["rate"], direction="down", **kw)
    up = dict(start=c["start"] + 0.2 * dur, end=c["end"])
    if t0 <= up["start"] < t1:
        s = up["start"]
        buf.add(s, att["ip"], c["dst"], att["mac"], c["port"], C.HTTPS_PORT, "TCP", "S", 60, att["ttl"], **kw)
        buf.add(s + 0.06, c["dst"], att["ip"], "02:ee:ee:00:00:03", C.HTTPS_PORT, c["port"], "TCP", "SA", 58, 52, **kw)
        buf.add(s + 0.061, att["ip"], c["dst"], att["mac"], c["port"], C.HTTPS_PORT, "TCP", "A", 54, att["ttl"], **kw)
    tw._emit_stream(buf, up, t0, t1, src=att["ip"], src_mac=att["mac"], src_ttl=att["ttl"], dst=c["dst"],
                    dport=C.HTTPS_PORT, sport=c["port"], rate_bytes=c["rate"], direction="up", **kw)
