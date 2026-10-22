"""Digital twin of a hospital ward with connected medical IoT devices.

The twin keeps three synchronised layers:
  1. Physical layer   - a physiological model of every patient (vital signs).
  2. Device layer     - patient monitors, infusion pumps and wearables that sample
                        the patients and publish telemetry over MQTT.
  3. Network layer    - packet-level events between devices, the IoMT gateway
                        (MQTT broker), the nurse station, the EHR server and
                        clinical workstations.

``WardTwin.step(t0, t1)`` advances the twin and returns every packet event in the
interval, so the same engine drives offline dataset generation and the real-time
stream. Attack campaigns are injected by ``medtwin.attacks``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Packet buffer
# --------------------------------------------------------------------------
PACKET_COLUMNS = [
    "ts", "src_ip", "dst_ip", "src_mac", "sport", "dport", "proto", "flags",
    "length", "ttl", "mqtt", "mqtt_rc", "retrans", "device_id", "seq",
    "temp", "spo2", "pulse", "sys", "dia", "hr", "rr", "st", "pump_rate",
    "label", "attack", "campaign", "intensity",
]
NAN = float("nan")
NO_VITALS = (NAN,) * 8


class PacketBuffer:
    """Column-oriented packet store (much faster than a list of dicts)."""

    def __init__(self):
        self.cols = {c: [] for c in PACKET_COLUMNS}

    def add(self, ts, src_ip, dst_ip, src_mac, sport, dport, proto, flags, length, ttl,
            mqtt="", mqtt_rc=-1, retrans=0, device_id="", seq=-1, vitals=NO_VITALS,
            pump_rate=NAN, label=0, attack=C.NORMAL, campaign=-1, intensity=""):
        c = self.cols
        c["ts"].append(ts); c["src_ip"].append(src_ip); c["dst_ip"].append(dst_ip)
        c["src_mac"].append(src_mac); c["sport"].append(sport); c["dport"].append(dport)
        c["proto"].append(proto); c["flags"].append(flags); c["length"].append(length)
        c["ttl"].append(ttl); c["mqtt"].append(mqtt); c["mqtt_rc"].append(mqtt_rc)
        c["retrans"].append(retrans); c["device_id"].append(device_id); c["seq"].append(seq)
        c["temp"].append(vitals[0]); c["spo2"].append(vitals[1]); c["pulse"].append(vitals[2])
        c["sys"].append(vitals[3]); c["dia"].append(vitals[4]); c["hr"].append(vitals[5])
        c["rr"].append(vitals[6]); c["st"].append(vitals[7]); c["pump_rate"].append(pump_rate)
        c["label"].append(label); c["attack"].append(attack); c["campaign"].append(campaign)
        c["intensity"].append(intensity)

    def __len__(self):
        return len(self.cols["ts"])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.cols)
        if len(df):
            df = df.sort_values("ts", kind="stable").reset_index(drop=True)
        return df


# --------------------------------------------------------------------------
# Asset registry (the twin's inventory of what *should* be on the network)
# --------------------------------------------------------------------------
def _mac(prefix: int, idx: int) -> str:
    return "02:1a:%02x:00:%02x:%02x" % (prefix, idx // 256, idx % 256)


def build_registry() -> pd.DataFrame:
    rows = [
        dict(device_id="GW-01", type="gateway", ip=C.BROKER_IP, mac=_mac(0, 10), bed=0),
        dict(device_id="NS-01", type="nurse_station", ip=C.NURSE_STATION_IP, mac=_mac(0, 20), bed=0),
        dict(device_id="EHR-01", type="ehr_server", ip=C.EHR_IP, mac=_mac(0, 30), bed=0),
        dict(device_id="UPD-01", type="update_server", ip=C.UPDATE_SERVER_IP, mac=_mac(0, 40), bed=0),
    ]
    for i, ip in enumerate(C.WORKSTATION_IPS, start=1):
        rows.append(dict(device_id=f"WS-{i:02d}", type="workstation", ip=ip, mac=_mac(0, 50 + i), bed=0))
    for b in range(1, C.N_BEDS + 1):
        rows.append(dict(device_id=f"MON-{b:02d}", type="patient_monitor", ip=f"10.10.1.{10 + b}", mac=_mac(1, b), bed=b))
        rows.append(dict(device_id=f"PMP-{b:02d}", type="infusion_pump", ip=f"10.10.2.{10 + b}", mac=_mac(2, b), bed=b))
        rows.append(dict(device_id=f"WRB-{b:02d}", type="wearable", ip=f"10.10.3.{10 + b}", mac=_mac(3, b), bed=b))
    reg = pd.DataFrame(rows)
    reg["ttl"] = reg["type"].map(C.DEVICE_TTL).astype(int)
    reg["criticality"] = reg["type"].map(C.CRITICALITY)
    return reg


# --------------------------------------------------------------------------
# Physiological model
# --------------------------------------------------------------------------
VITAL_KEYS = ["hr", "spo2", "temp", "sys", "dia", "rr", "st"]
EPISODE_EFFECTS = {
    "tachycardia": {"hr": 35.0, "rr": 5.0},
    "bradycardia": {"hr": -25.0},
    "desaturation": {"spo2": -8.0, "hr": 12.0, "rr": 6.0},
    "fever": {"temp": 1.6, "hr": 15.0},
    "hypertension": {"sys": 35.0, "dia": 15.0},
    "st_elevation": {"st": 0.15, "hr": 10.0},
}
OU_THETA = {"hr": 0.05, "spo2": 0.08, "temp": 0.01, "sys": 0.02, "dia": 0.02, "rr": 0.05, "st": 0.05}
OU_SIGMA = {"hr": 0.9, "spo2": 0.25, "temp": 0.01, "sys": 0.6, "dia": 0.4, "rr": 0.3, "st": 0.004}


class PatientModel:
    """Ornstein-Uhlenbeck vital signs around a personal baseline with clinical episodes."""

    def __init__(self, n_beds: int, rng: np.random.Generator):
        self.n = n_beds
        self.rng = rng
        self.base = {
            "hr": rng.normal(78, 9, n_beds),
            "spo2": np.clip(rng.normal(97, 1.0, n_beds), 93.5, 99.5),
            "temp": rng.normal(36.9, 0.25, n_beds),
            "sys": rng.normal(122, 12, n_beds),
            "dia": rng.normal(78, 7, n_beds),
            "rr": rng.normal(16, 2, n_beds),
            "st": rng.normal(0.0, 0.04, n_beds),
        }
        self.state = {k: v.copy() for k, v in self.base.items()}
        self.episodes: list[dict] = []  # dict(bed, kind, start, end)

    def target(self, t: float, hour: float) -> dict:
        circ = math.sin(2 * math.pi * (hour - 10.0) / 24.0)
        tgt = {k: v.copy() for k, v in self.base.items()}
        tgt["hr"] += 3.0 * circ
        tgt["sys"] += 4.0 * circ
        for ep in self.episodes:
            if ep["start"] <= t < ep["end"]:
                ramp = min(1.0, (t - ep["start"]) / 90.0, (ep["end"] - t) / 90.0)
                for k, delta in EPISODE_EFFECTS[ep["kind"]].items():
                    tgt[k][ep["bed"] - 1] += ramp * delta
        return tgt

    def step(self, t: float, hour: float) -> dict:
        tgt = self.target(t, hour)
        for k in VITAL_KEYS:
            x = self.state[k]
            x += OU_THETA[k] * (tgt[k] - x) + OU_SIGMA[k] * self.rng.standard_normal(self.n)
        self.state["spo2"] = np.clip(self.state["spo2"], 70, 100)
        return {k: v.copy() for k, v in self.state.items()}


# --------------------------------------------------------------------------
# Ward digital twin
# --------------------------------------------------------------------------
class WardTwin:
    def __init__(self, cfg: C.ScenarioConfig):
        from . import attacks  # local import to avoid a cycle

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.rng_quality = np.random.default_rng(cfg.seed + 100_000)
        self.registry = build_registry()
        self.by_ip = {r.ip: r for r in self.registry.itertuples()}
        self.by_id = {r.device_id: r for r in self.registry.itertuples()}
        self.duration = cfg.hours * 3600.0
        self.patients = PatientModel(C.N_BEDS, self.rng)
        self.telemetry_devices = [r for r in self.registry.itertuples() if r.type in C.PUBLISH_PERIOD]

        rng = self.rng
        self.port = {r.device_id: int(rng.integers(32768, 61000)) for r in self.registry.itertuples()}
        self.next_pub = {d.device_id: float(rng.uniform(0, C.PUBLISH_PERIOD[d.type])) for d in self.telemetry_devices}
        self.seq = {d.device_id: int(rng.integers(0, 5000)) for d in self.telemetry_devices}
        self.offline_until = {d.device_id: -1.0 for d in self.telemetry_devices}
        self.pending_reconnect: dict[str, float] = {}
        self.pump_rate = {b: float(rng.choice([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])) for b in range(1, C.N_BEDS + 1)}
        self.nibp = {b: (float(self.patients.base["sys"][b - 1]), float(self.patients.base["dia"][b - 1])) for b in range(1, C.N_BEDS + 1)}
        self.next_nibp = {b: float(rng.uniform(0, 300)) for b in range(1, C.N_BEDS + 1)}
        self.last_forward = 0.0
        self.next_hl7 = float(rng.uniform(0, 30))
        self.next_arp = float(rng.uniform(0, 30))
        self.msgs_since_forward = 0

        self.benign_events = self._schedule_benign()
        self.campaigns = attacks.schedule_campaigns(self)
        self.attacks = attacks

    # ------------------------------------------------------------------ utils
    def hour_at(self, t: float) -> float:
        return (self.cfg.start_hour + t / 3600.0) % 24

    def _poisson_times(self, rate_per_hour: float, start: float = 60.0) -> np.ndarray:
        n = self.rng.poisson(rate_per_hour * self.cfg.hours)
        return np.sort(self.rng.uniform(start, max(start + 1, self.duration - 60), n))

    def _schedule_benign(self) -> list[dict]:
        rng, cfg, ev = self.rng, self.cfg, []
        for t in self._poisson_times(cfg.clinical_events_per_hour):
            kind = str(rng.choice(list(EPISODE_EFFECTS)))
            bed = int(rng.integers(1, C.N_BEDS + 1))
            e = dict(kind="clinical_" + kind, bed=bed, start=float(t), end=float(t + rng.uniform(180, 900)))
            self.patients.episodes.append(dict(bed=bed, kind=kind, start=e["start"], end=e["end"]))
            ev.append(e)
        for t in self._poisson_times(cfg.congestion_events_per_hour):
            ev.append(dict(kind="congestion", start=float(t), end=float(t + rng.uniform(60, 300))))
        for t in self._poisson_times(2.0):
            bed = int(rng.integers(1, C.N_BEDS + 1))
            ev.append(dict(kind="motion_artefact", bed=bed, start=float(t), end=float(t + rng.uniform(30, 150))))
        for t in self._poisson_times(cfg.firmware_updates_per_hour):
            dev = self.telemetry_devices[int(rng.integers(len(self.telemetry_devices)))]
            size = float(rng.uniform(2e6, 6e6))
            dur = float(rng.uniform(40, 120))
            ev.append(dict(kind="firmware_update", device=dev.device_id, start=float(t), end=float(t + dur), rate=size / dur))
        for t in self._poisson_times(cfg.reboots_per_hour):
            dev = self.telemetry_devices[int(rng.integers(len(self.telemetry_devices)))]
            ev.append(dict(kind="reboot", device=dev.device_id, start=float(t), end=float(t + rng.uniform(15, 60))))
        for t in self._poisson_times(6.0):
            ev.append(dict(kind="nurse_pump_command", bed=int(rng.integers(1, C.N_BEDS + 1)), start=float(t), end=float(t) + 0.01))
        for t in self._poisson_times(1.0):
            ev.append(dict(kind="nurse_login_retry", start=float(t), end=float(t) + 6.0))
        for ws in C.WORKSTATION_IPS:
            for t in self._poisson_times(30.0, start=5.0):
                ev.append(dict(kind="ehr_session", ws=ws, start=float(t), end=float(t + rng.uniform(1, 6)),
                               down=float(rng.uniform(2e4, 3e5)), up=float(rng.uniform(1e3, 4e3)), server=C.EHR_IP))
            for t in self._poisson_times(10.0, start=5.0):
                ev.append(dict(kind="vendor_portal", ws=ws, start=float(t), end=float(t + rng.uniform(2, 10)),
                               down=float(rng.uniform(5e4, 8e5)), up=float(rng.uniform(2e3, 2e4)), server=C.EXTERNAL_IPS[2]))
        return sorted(ev, key=lambda e: e["start"])

    def active(self, events, t0, t1):
        return [e for e in events if e["start"] < t1 and e["end"] > t0]

    # ------------------------------------------------------------- main step
    def step(self, t0: float, t1: float) -> pd.DataFrame:
        buf = PacketBuffer()
        rng = self.rng
        benign = self.active(self.benign_events, t0, t1)
        camps = self.active(self.campaigns, t0, t1)
        congested = any(e["kind"] == "congestion" for e in benign)
        artefact_beds = {e["bed"] for e in benign if e["kind"] == "motion_artefact"}
        rtt_extra = self.attacks.broker_load_delay(camps)

        # Physiology: one state per integer second in the window
        secs = np.arange(math.floor(t0), math.ceil(t1))
        states = {int(s): self.patients.step(float(s), self.hour_at(float(s))) for s in secs}

        for e in benign:
            if e["kind"] == "reboot" and e["start"] < t1 and e["start"] >= t0:
                self.offline_until[e["device"]] = e["end"]
                self.pending_reconnect[e["device"]] = e["end"]

        # ---- telemetry from every medical device
        for d in self.telemetry_devices:
            dev_id, period = d.device_id, C.PUBLISH_PERIOD[d.type]
            # reconnect after reboot
            rc_t = self.pending_reconnect.get(dev_id)
            if rc_t is not None and t0 <= rc_t < t1:
                self.port[dev_id] = int(rng.integers(32768, 61000))
                self.seq[dev_id] = 0
                self._emit_connect(buf, rc_t, d.ip, d.mac, d.ttl, self.port[dev_id])
                del self.pending_reconnect[dev_id]
            while self.next_pub[dev_id] < t1:
                ts = self.next_pub[dev_id]
                jitter_sd = 0.008 * (6 if congested else 1)
                self.next_pub[dev_id] = ts + period + rng.normal(0, jitter_sd)
                if ts < t0:
                    continue
                if ts < self.offline_until[dev_id]:
                    continue
                self._emit_telemetry(buf, d, ts, states, congested, artefact_beds, rtt_extra, camps)

        # ---- broker forwards to the nurse station once a second
        t = math.floor(t0)
        while t < t1:
            if t >= t0:
                n = max(1, self.msgs_since_forward)
                self.msgs_since_forward = 0
                ns_port = self.port["NS-01"]
                ts = t + 0.05 + rng.uniform(0, 0.01)
                buf.add(ts, C.BROKER_IP, C.NURSE_STATION_IP, self.by_id["GW-01"].mac, C.MQTT_PORT, ns_port, "TCP", "PA",
                        int(60 + 38 * n + rng.integers(0, 20)), 64, mqtt="PUBLISH")
                buf.add(ts + 0.001 + rng.exponential(0.001), C.NURSE_STATION_IP, C.BROKER_IP, self.by_id["NS-01"].mac,
                        ns_port, C.MQTT_PORT, "TCP", "A", 54, 128)
            t += 1

        # ---- HL7 batches from gateway to EHR every 30 s
        while self.next_hl7 < t1:
            ts = self.next_hl7
            self.next_hl7 += 30 + rng.normal(0, 0.5)
            if ts >= t0:
                self._emit_bulk(buf, ts, C.BROKER_IP, C.EHR_IP, self.port["GW-01"], C.HL7_PORT,
                                up=float(rng.uniform(3e3, 8e3)), down=120.0, dur=0.15)

        # ---- ARP housekeeping from the gateway
        while self.next_arp < t1:
            ts = self.next_arp
            self.next_arp += rng.exponential(30)
            if ts >= t0:
                tgt = self.registry.iloc[int(rng.integers(len(self.registry)))]
                if tgt.ip != C.BROKER_IP:
                    buf.add(ts, C.BROKER_IP, tgt.ip, self.by_id["GW-01"].mac, 0, 0, "ARP", "", 42, 0)
                    buf.add(ts + 0.002, tgt.ip, C.BROKER_IP, tgt.mac, 0, 0, "ARP", "", 60, 0)

        # ---- other benign activity
        for e in benign:
            k = e["kind"]
            if k in ("ehr_session", "vendor_portal"):
                ws = self.by_ip[e["ws"]]
                self._emit_session(buf, e, t0, t1, ws.ip, ws.mac, ws.ttl, e["server"])
            elif k == "firmware_update":
                dev = self.by_id[e["device"]]
                self._emit_stream(buf, e, t0, t1, src=dev.ip, src_mac=dev.mac, src_ttl=dev.ttl,
                                  dst=C.UPDATE_SERVER_IP, dport=C.HTTPS_PORT, sport=self.port[dev.device_id] + 1,
                                  rate_bytes=e["rate"], direction="down")
            elif k == "nurse_pump_command" and t0 <= e["start"] < t1:
                self._emit_pump_command(buf, e["start"], e["bed"], malicious=False)
            elif k == "nurse_login_retry" and t0 <= e["start"] < t1:
                ns = self.by_id["NS-01"]
                self._emit_connect(buf, e["start"], ns.ip, ns.mac, ns.ttl, int(rng.integers(32768, 61000)), rc=5)
                self._emit_connect(buf, e["start"] + rng.uniform(3, 6), ns.ip, ns.mac, ns.ttl, int(rng.integers(32768, 61000)), rc=0)

        # ---- attack campaigns
        for c in camps:
            self.attacks.emit(self, c, t0, t1, buf, states)

        return self._capture_artefacts(buf.to_frame())

    def _capture_artefacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data-quality problems seen in real captures, for the cleaner to handle.

        Uses its own random stream so that the simulated ward itself is unaffected.
        """
        if df.empty:
            return df
        rng = self.rng_quality
        n = len(df)
        # 1) the same frame captured twice by a mirrored (SPAN) port
        dup = df[rng.random(n) < 0.001]
        # 2) sensor sentinel values in telemetry
        tel = np.flatnonzero(df["hr"].notna().to_numpy() & (rng.random(n) < 0.0005))
        for i in tel:
            col, val = [("hr", 65535.0), ("spo2", 127.0), ("temp", -273.0)][int(rng.integers(3))]
            df.iat[i, df.columns.get_loc(col)] = val
        # 3) truncated capture records with no destination address
        cut = rng.random(n) < 0.00005
        df.loc[cut, "dst_ip"] = None
        if len(dup):
            df = pd.concat([df, dup], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
        return df

    # --------------------------------------------------------- emit helpers
    def sense(self, d, ts: float, states: dict, artefact_beds: set):
        """Sensor reading of the patient by device ``d`` (adds device-specific noise)."""
        rng = self.rng
        s = states.get(int(math.floor(ts))) or states[max(states)]
        i = d.bed - 1
        if d.type == "patient_monitor":
            if ts >= self.next_nibp[d.bed]:
                self.nibp[d.bed] = (float(s["sys"][i] + rng.normal(0, 3)), float(s["dia"][i] + rng.normal(0, 2)))
                self.next_nibp[d.bed] = ts + 300
            sys_, dia_ = self.nibp[d.bed]
            hr = s["hr"][i] + rng.normal(0, 1.0)
            return (round(float(s["temp"][i] + rng.normal(0, 0.05)), 2),
                    round(float(min(100, s["spo2"][i] + rng.normal(0, 0.5))), 1),
                    round(float(s["hr"][i] + rng.normal(0, 1.5)), 1),
                    round(sys_, 1), round(dia_, 1), round(float(hr), 1),
                    round(float(s["rr"][i] + rng.normal(0, 0.7)), 1),
                    round(float(s["st"][i] + rng.normal(0, 0.02)), 3))
        if d.type == "wearable":
            hr_sd = 8.0 if d.bed in artefact_beds else 2.5
            return (round(float(s["temp"][i] - 0.4 + rng.normal(0, 0.1)), 2),
                    round(float(min(100, s["spo2"][i] + rng.normal(0, 1.0))), 1),
                    NAN, NAN, NAN,
                    round(float(s["hr"][i] + rng.normal(0, hr_sd)), 1),
                    round(float(s["rr"][i] + rng.normal(0, 1.0)), 1), NAN)
        return NO_VITALS

    def _emit_telemetry(self, buf, d, ts, states, congested, artefact_beds, rtt_extra, camps):
        rng = self.rng
        dev_id = d.device_id
        vitals = self.sense(d, ts, states, artefact_beds)
        pump = self.pump_rate[d.bed] if d.type == "infusion_pump" else NAN
        base_len = {"patient_monitor": 190, "wearable": 150, "infusion_pump": 120}[d.type]
        ttl, extra_delay, label, attack, camp, inten = d.ttl, 0.0, 0, C.NORMAL, -1, ""
        # in-path manipulation (data alteration) is applied by the attack module
        mod = self.attacks.in_path_effect(self, camps, d, ts)
        if mod is not None:
            vitals = mod["alter"](vitals)
            ttl = ttl - mod["ttl_drop"]
            extra_delay = mod["delay"]
            ts = ts + mod["jitter"]
            label, attack, camp, inten = 1, mod["attack"], mod["campaign"], mod["intensity"]
        seq = self.seq[dev_id]
        self.seq[dev_id] = seq + 1
        self.msgs_since_forward += 1
        length = int(base_len + rng.integers(-6, 7))
        port = self.port[dev_id]
        buf.add(ts, d.ip, C.BROKER_IP, d.mac, port, C.MQTT_PORT, "TCP", "PA", length, ttl, mqtt="PUBLISH",
                device_id=dev_id, seq=seq, vitals=vitals, pump_rate=pump,
                label=label, attack=attack, campaign=camp, intensity=inten)
        if rng.random() < (0.04 if congested else 0.002):
            buf.add(ts + 0.2 + rng.uniform(0, 0.1), d.ip, C.BROKER_IP, d.mac, port, C.MQTT_PORT, "TCP", "PA", length, ttl,
                    mqtt="PUBLISH", retrans=1, device_id=dev_id, seq=seq, vitals=vitals, pump_rate=pump,
                    label=label, attack=attack, campaign=camp, intensity=inten)
        rtt = 0.002 + rng.exponential(0.001) + (0.02 * rng.random() if congested else 0.0) + rtt_extra + extra_delay
        buf.add(ts + rtt, C.BROKER_IP, d.ip, self.by_id["GW-01"].mac, C.MQTT_PORT, port, "TCP", "PA", 58, 64,
                mqtt="PUBACK", label=label, attack=attack, campaign=camp, intensity=inten)

    def _emit_connect(self, buf, ts, ip, mac, ttl, sport, rc=0, dst=C.BROKER_IP, label=0, attack=C.NORMAL,
                      campaign=-1, intensity="", fin=False):
        rng = self.rng
        gw_mac = self.by_id["GW-01"].mac
        kw = dict(label=label, attack=attack, campaign=campaign, intensity=intensity)
        t = ts
        buf.add(t, ip, dst, mac, sport, C.MQTT_PORT, "TCP", "S", 60, ttl, **kw)
        t += 0.001 + rng.exponential(0.0005)
        buf.add(t, dst, ip, gw_mac, C.MQTT_PORT, sport, "TCP", "SA", 58, 64, **kw)
        t += 0.001 + rng.exponential(0.0005)
        buf.add(t, ip, dst, mac, sport, C.MQTT_PORT, "TCP", "A", 54, ttl, **kw)
        buf.add(t + 0.0005, ip, dst, mac, sport, C.MQTT_PORT, "TCP", "PA", int(70 + rng.integers(0, 40)), ttl,
                mqtt="CONNECT", **kw)
        t += 0.002 + rng.exponential(0.001)
        buf.add(t, dst, ip, gw_mac, C.MQTT_PORT, sport, "TCP", "PA", 58, 64, mqtt="CONNACK", mqtt_rc=rc, **kw)
        if rc != 0 or fin:
            buf.add(t + 0.001, dst, ip, gw_mac, C.MQTT_PORT, sport, "TCP", "FA", 54, 64, **kw)
        return t

    def _emit_pump_command(self, buf, ts, bed, malicious=False, src_ip=C.NURSE_STATION_IP, src_mac=None,
                           src_ttl=128, src_port=None, new_rate=None, campaign=-1, intensity=""):
        rng = self.rng
        pump = self.by_id[f"PMP-{bed:02d}"]
        kw = dict(label=int(malicious), attack="unauthorised_access" if malicious else C.NORMAL,
                  campaign=campaign, intensity=intensity)
        src_mac = src_mac or self.by_id["NS-01"].mac
        src_port = src_port or self.port["NS-01"]
        gw_mac = self.by_id["GW-01"].mac
        if new_rate is None:
            cur = self.pump_rate[bed]
            if rng.random() < 0.1:
                new_rate = 0.0 if cur > 0 else float(rng.choice([25.0, 50.0, 75.0]))
            else:
                new_rate = max(5.0, round(cur * (1 + rng.uniform(-0.3, 0.3)) if cur > 0 else float(rng.choice([25.0, 50.0])), 1))
        # command messages carry the target pump and the requested setpoint
        cmd = dict(device_id=pump.device_id, pump_rate=float(new_rate))
        buf.add(ts, src_ip, C.BROKER_IP, src_mac, src_port, C.MQTT_PORT, "TCP", "PA", int(105 + rng.integers(0, 12)),
                src_ttl, mqtt="CMD", **cmd, **kw)
        buf.add(ts + 0.002, C.BROKER_IP, src_ip, gw_mac, C.MQTT_PORT, src_port, "TCP", "PA", 58, 64, mqtt="PUBACK", **kw)
        t2 = ts + 0.003 + rng.exponential(0.001)
        buf.add(t2, C.BROKER_IP, pump.ip, gw_mac, C.MQTT_PORT, self.port[pump.device_id], "TCP", "PA",
                int(105 + rng.integers(0, 12)), 64, mqtt="CMD", **cmd, **kw)
        buf.add(t2 + 0.003, pump.ip, C.BROKER_IP, pump.mac, self.port[pump.device_id], C.MQTT_PORT, "TCP", "PA", 58,
                pump.ttl, mqtt="PUBACK", **kw)
        self.pump_rate[bed] = float(new_rate)

    def _emit_bulk(self, buf, ts, src, dst, sport, dport, up, down, dur, label=0, attack=C.NORMAL, campaign=-1, intensity=""):
        """Short request/response transfer of ``up`` bytes upstream and ``down`` bytes downstream."""
        rng = self.rng
        s, d = self.by_ip.get(src), self.by_ip.get(dst)
        smac = s.mac if s is not None else "02:66:66:00:00:01"
        sttl = s.ttl if s is not None else 64
        dmac = d.mac if d is not None else "02:ee:ee:00:00:01"
        dttl = d.ttl if d is not None else 52
        kw = dict(label=label, attack=attack, campaign=campaign, intensity=intensity)
        n_up = max(1, int(math.ceil(up / 1460)))
        n_dn = max(1, int(math.ceil(down / 1460)))
        times_up = ts + np.sort(rng.uniform(0, dur, n_up))
        for i, t in enumerate(times_up):
            size = 1514 if i < n_up - 1 else int(54 + up - 1460 * (n_up - 1))
            buf.add(float(t), src, dst, smac, sport, dport, "TCP", "PA", min(1514, max(60, size)), sttl, **kw)
        times_dn = ts + dur * 0.3 + np.sort(rng.uniform(0, dur, n_dn))
        for i, t in enumerate(times_dn):
            size = 1514 if i < n_dn - 1 else int(54 + down - 1460 * (n_dn - 1))
            buf.add(float(t), dst, src, dmac, dport, sport, "TCP", "PA", min(1514, max(60, size)), dttl, **kw)
            if i % 2 == 1:
                buf.add(float(t) + 0.0005, src, dst, smac, sport, dport, "TCP", "A", 54, sttl, **kw)

    def _emit_session(self, buf, e, t0, t1, src, src_mac, src_ttl, server):
        """HTTPS session (handshake at the start, then a download and a small upload)."""
        rng = self.rng
        sport = e.setdefault("sport", int(rng.integers(32768, 61000)))
        d = self.by_ip.get(server)
        dmac = d.mac if d is not None else "02:ee:ee:00:00:02"
        dttl = d.ttl if d is not None else 52
        a, b = max(t0, e["start"]), min(t1, e["end"])
        if b <= a:
            return
        if t0 <= e["start"] < t1:
            buf.add(e["start"], src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "S", 60, src_ttl)
            buf.add(e["start"] + 0.002, server, src, dmac, C.HTTPS_PORT, sport, "TCP", "SA", 58, dttl)
            buf.add(e["start"] + 0.003, src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "A", 54, src_ttl)
            buf.add(e["start"] + 0.004, src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "PA", 571, src_ttl)
        span = e["end"] - e["start"]
        frac = (b - a) / span
        n_dn = int(rng.poisson(frac * e["down"] / 1460))
        n_up = int(rng.poisson(frac * e["up"] / 600)) + 1
        for t in np.sort(rng.uniform(a, b, n_dn)):
            buf.add(float(t), server, src, dmac, C.HTTPS_PORT, sport, "TCP", "PA", 1514, dttl)
        for t in np.sort(rng.uniform(a, b, n_dn // 2)):
            buf.add(float(t), src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "A", 54, src_ttl)
        for t in np.sort(rng.uniform(a, b, n_up)):
            buf.add(float(t), src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "PA", int(rng.integers(200, 1514)), src_ttl)
        if t0 <= e["end"] < t1:
            buf.add(e["end"], src, server, src_mac, sport, C.HTTPS_PORT, "TCP", "FA", 54, src_ttl)
            buf.add(e["end"] + 0.002, server, src, dmac, C.HTTPS_PORT, sport, "TCP", "FA", 54, dttl)

    def _emit_stream(self, buf, e, t0, t1, src, src_mac, src_ttl, dst, dport, sport, rate_bytes, direction,
                     label=0, attack=C.NORMAL, campaign=-1, intensity=""):
        """Sustained transfer at ``rate_bytes`` per second; ``direction`` is 'up' (src->dst) or 'down'."""
        rng = self.rng
        a, b = max(t0, e["start"]), min(t1, e["end"])
        if b <= a:
            return
        d = self.by_ip.get(dst)
        dmac = d.mac if d is not None else "02:ee:ee:00:00:03"
        dttl = d.ttl if d is not None else 52
        kw = dict(label=label, attack=attack, campaign=campaign, intensity=intensity)
        n = int(rng.poisson((b - a) * rate_bytes / 1460))
        times = np.sort(rng.uniform(a, b, n))
        if direction == "down":
            for i, t in enumerate(times):
                buf.add(float(t), dst, src, dmac, dport, sport, "TCP", "PA", 1514, dttl, **kw)
                if i % 2 == 1:
                    buf.add(float(t) + 0.0004, src, dst, src_mac, sport, dport, "TCP", "A", 54, src_ttl, **kw)
        else:
            for i, t in enumerate(times):
                buf.add(float(t), src, dst, src_mac, sport, dport, "TCP", "PA", 1514, src_ttl, **kw)
                if i % 2 == 1:
                    buf.add(float(t) + 0.03 + rng.exponential(0.01), dst, src, dmac, dport, sport, "TCP", "A", 54, dttl, **kw)
