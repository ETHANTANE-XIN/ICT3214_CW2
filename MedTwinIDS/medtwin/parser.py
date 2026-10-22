"""Packet-log parser, cleaner and feature extractor.

Raw packet events (from the digital twin or a replayed capture) are cleaned and
aggregated into *service-level bidirectional flow windows*: one record per
(time window, initiator IP, responder IP, service port, protocol). Each record
carries three groups of features:

* network   - volume, timing, size, TCP flag and MQTT statistics (Argus-like);
* biometric - patient vital signs carried in the flow's telemetry;
* twin      - context only a digital twin can supply: asset registry checks
              (unknown host, MAC/TTL deviation), sequence integrity and
              cross-sensor physiological consistency.

The same ``FeatureExtractor`` is used for offline datasets and for the real-time
detector, so training and live inference see identical features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ESSENTIAL = ["ts", "src_ip", "dst_ip", "proto", "length"]
VITAL_MAP = {"temp": "Temp", "spo2": "SpO2", "pulse": "Pulse_Rate", "sys": "SYS", "dia": "DIA",
             "hr": "Heart_rate", "rr": "Resp_Rate", "st": "ST"}
META_COLS = ["window", "t_start", "init_ip", "resp_ip", "svc_port", "proto", "device_id", "bed",
             "label", "attack", "campaign", "intensity"]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_packets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sanitise a raw packet log and return (clean_df, report)."""
    report = {"rows_in": int(len(df))}
    df = df.copy()
    missing = df[ESSENTIAL].isna().any(axis=1)
    report["dropped_missing_essential"] = int(missing.sum())
    df = df[~missing]
    # exact duplicates that are not flagged as TCP retransmissions are capture artefacts
    dup = df.duplicated(keep="first") & (df["retrans"] == 0)
    report["dropped_exact_duplicates"] = int(dup.sum())
    df = df[~dup]
    neg = df["ts"] < 0
    report["dropped_negative_time"] = int(neg.sum())
    df = df[~neg]
    df["length"] = df["length"].clip(42, 1514).astype(np.int32)
    for col in ("sport", "dport", "ttl", "mqtt_rc", "retrans", "seq", "label", "campaign"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1 if col in ("seq", "campaign", "mqtt_rc") else 0).astype(np.int64)
    for col in ("src_mac", "flags", "mqtt", "device_id", "attack", "intensity"):
        df[col] = df[col].fillna("").astype(str)
    # values outside the sensors' measurement range are glitches (e.g. 65535 sentinels). Clinically
    # extreme but measurable values are kept: removing them would erase evidence of tampering.
    bounds = {"hr": (0, 300), "spo2": (0, 100), "temp": (25, 45), "sys": (0, 300), "dia": (0, 250),
              "rr": (0, 100), "pulse": (0, 300)}
    glitches = 0
    for col, (lo, hi) in bounds.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        glitches += int(bad.sum())
        df.loc[bad, col] = np.nan
    report["vital_glitches_nulled"] = glitches
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
class FeatureExtractor:
    def __init__(self, registry: pd.DataFrame, window: float = C.WINDOW_SECONDS):
        self.window = float(window)
        self.reg_mac = dict(zip(registry.ip, registry.mac))
        self.reg_ttl = dict(zip(registry.ip, registry.ttl))
        self.dev_bed = dict(zip(registry.device_id, registry.bed))
        self.dev_type = dict(zip(registry.device_id, registry.type))
        self.state: dict = {}  # flow key -> last (hr, spo2, pump_rate)
        self.pump_last: dict = {}  # pump device id -> last reported infusion rate

    def transform(self, pk: pd.DataFrame) -> pd.DataFrame:
        if pk.empty:
            return pd.DataFrame(columns=META_COLS + C.ALL_FEATURES)
        W = self.window
        pk = pk.copy()
        pk["window"] = np.floor(pk["ts"] / W).astype(np.int64)
        fwd = self._orientation(pk)
        pk["is_src"] = fwd
        pk["init_ip"] = np.where(fwd, pk["src_ip"], pk["dst_ip"])
        pk["resp_ip"] = np.where(fwd, pk["dst_ip"], pk["src_ip"])
        pk["svc_port"] = np.where(fwd, pk["dport"], pk["sport"])
        keys = ["window", "init_ip", "resp_ip", "svc_port", "proto"]
        pk["flow_id"] = pk.groupby(keys, sort=False).ngroup()

        src = pk["is_src"]
        L = pk["length"]
        pk["s_len"] = L.where(src)
        pk["d_len"] = L.where(~src)
        pk["is_syn"] = (pk["flags"] == "S").astype(np.int8)
        pk["is_rst"] = pk["flags"].isin(["R", "RA"]).astype(np.int8)
        pk["is_fin"] = pk["flags"].isin(["F", "FA"]).astype(np.int8)
        pk["is_arp"] = (pk["proto"] == "ARP").astype(np.int8)
        pk["is_connect"] = (pk["mqtt"] == "CONNECT").astype(np.int8)
        pk["is_connfail"] = ((pk["mqtt"] == "CONNACK") & (pk["mqtt_rc"] > 0)).astype(np.int8)
        pk["is_pub"] = ((pk["mqtt"] == "PUBLISH") & src).astype(np.int8)
        pk["is_cmd"] = (pk["mqtt"] == "CMD").astype(np.int8)
        reg_mac = pk["src_ip"].map(self.reg_mac)
        pk["mac_bad"] = (reg_mac.notna() & (reg_mac != pk["src_mac"]) & src).astype(np.int8)
        reg_ttl = pk["src_ip"].map(self.reg_ttl)
        pk["ttl_dev"] = ((pk["ttl"] - reg_ttl).abs()).where(src & reg_ttl.notna() & (pk["proto"] != "ARP"))
        telem = src & (pk["mqtt"] == "PUBLISH") & (pk["device_id"] != "")
        for raw in list(VITAL_MAP) + ["pump_rate"]:
            pk["v_" + raw] = pk[raw].where(telem)
        pk["dev_t"] = pk["device_id"].where(telem)
        pk["cmd_delta"] = self._command_deltas(pk, telem)

        # inter-arrival times per direction
        pk = pk.sort_values(["flow_id", "is_src", "ts"], kind="stable")
        pk["iat"] = pk.groupby(["flow_id", "is_src"], sort=False)["ts"].diff() * 1000.0
        pk["s_iat"] = pk["iat"].where(pk["is_src"])
        pk["d_iat"] = pk["iat"].where(~pk["is_src"])

        g = pk.groupby("flow_id", sort=True)
        f = g.agg(
            window=("window", "first"), init_ip=("init_ip", "first"), resp_ip=("resp_ip", "first"),
            svc_port=("svc_port", "first"), proto=("proto", "first"),
            first_ts=("ts", "min"), last_ts=("ts", "max"),
            SrcPkts=("is_src", "sum"), TotPkts=("ts", "size"),
            SrcBytes=("s_len", "sum"), DstBytes=("d_len", "sum"),
            sMaxPktSz=("s_len", "max"), sMinPktSz=("s_len", "min"), sMeanPktSz=("s_len", "mean"),
            dMaxPktSz=("d_len", "max"), dMinPktSz=("d_len", "min"),
            SIntPkt=("s_iat", "mean"), DIntPkt=("d_iat", "mean"),
            SrcJitter=("s_iat", "std"), DstJitter=("d_iat", "std"),
            SynCnt=("is_syn", "sum"), RstCnt=("is_rst", "sum"), FinCnt=("is_fin", "sum"), ArpCnt=("is_arp", "sum"),
            Retrans=("retrans", "sum"), MqttConnect=("is_connect", "sum"), MqttConnFail=("is_connfail", "sum"),
            MqttPublish=("is_pub", "sum"), MqttCmd=("is_cmd", "sum"),
            MacMismatch=("mac_bad", "max"), TTLDev=("ttl_dev", "mean"),
            Temp=("v_temp", "mean"), SpO2=("v_spo2", "mean"), Pulse_Rate=("v_pulse", "mean"), SYS=("v_sys", "mean"),
            DIA=("v_dia", "mean"), Heart_rate=("v_hr", "mean"), Resp_Rate=("v_rr", "mean"), ST=("v_st", "mean"),
            PumpRate=("v_pump_rate", "mean"), VitalsStd=("v_hr", "std"), CmdDelta=("cmd_delta", "max"),
            device_id=("dev_t", "first"), label=("label", "max"),
        )
        f["SrcPkts"] = f["SrcPkts"].astype(np.int64)
        f["DstPkts"] = f["TotPkts"] - f["SrcPkts"]
        f["TotBytes"] = f["SrcBytes"] + f["DstBytes"]
        f["Dur"] = f["last_ts"] - f["first_ts"]
        f["Rate"] = f["TotPkts"] / W
        f["SrcLoad"] = f["SrcBytes"] * 8 / W
        f["DstLoad"] = f["DstBytes"] * 8 / W
        f["t_start"] = f["window"] * W

        # ground truth (dominant attack inside the flow window)
        att = pk[pk["label"] == 1]
        if len(att):
            ga = att.groupby("flow_id")
            dom = ga["attack"].agg(lambda s: s.value_counts().index[0])
            f["attack"] = dom.reindex(f.index).fillna(C.NORMAL)
            f["campaign"] = ga["campaign"].first().reindex(f.index).fillna(-1).astype(int)
            f["intensity"] = ga["intensity"].first().reindex(f.index).fillna("")
        else:
            f["attack"], f["campaign"], f["intensity"] = C.NORMAL, -1, ""

        # host-level context: fan-out and port spread of each initiator per window
        host = f.groupby(["window", "init_ip"]).agg(SrcFanOut=("resp_ip", "nunique"), SrcPortSpread=("svc_port", "nunique"))
        f = f.join(host, on=["window", "init_ip"])

        # sequence integrity per claimed device id
        pubs = pk[(pk["mqtt"] == "PUBLISH") & pk["is_src"] & (pk["seq"] >= 0) & (pk["device_id"] != "")]
        if len(pubs):
            gs = pubs.groupby(["window", "device_id"])["seq"]
            seq_stats = pd.DataFrame({"SeqDup": gs.size() - gs.nunique()})
            u = pubs[["window", "device_id", "seq"]].drop_duplicates().sort_values(["window", "device_id", "seq"])
            d = u.groupby(["window", "device_id"])["seq"].diff()
            seq_stats["SeqGap"] = (d.notna() & (d != 1)).groupby([u["window"], u["device_id"]]).sum()
            f = f.join(seq_stats, on=["window", "device_id"])
        else:
            f["SeqDup"], f["SeqGap"] = 0, 0

        # cross-sensor consistency (digital twin residual): monitor vs wearable on the same bed
        dev_means = pubs.groupby(["window", "device_id"])[["hr", "spo2"]].mean() if len(pubs) else pd.DataFrame()
        f["bed"] = f["device_id"].map(self.dev_bed).fillna(0).astype(int)
        partner = f["device_id"].map(self._partner)
        if len(dev_means):
            idx = pd.MultiIndex.from_arrays([f["window"], partner.fillna("")])
            p_hr = dev_means["hr"].reindex(idx).to_numpy()
            p_sp = dev_means["spo2"].reindex(idx).to_numpy()
            f["TwinResidualHR"] = np.abs(f["Heart_rate"].to_numpy() - p_hr)
            f["TwinResidualSpO2"] = np.abs(f["SpO2"].to_numpy() - p_sp)
        else:
            f["TwinResidualHR"], f["TwinResidualSpO2"] = np.nan, np.nan

        # temporal deltas per flow key (state carries across streaming chunks)
        f = self._deltas(f)

        f["SrcKnown"] = f["init_ip"].isin(self.reg_mac.keys()).astype(np.int8)
        f["ExternalDst"] = (~f["resp_ip"].astype(str).str.startswith(C.INTERNAL_PREFIX)).astype(np.int8)
        f["HasVitals"] = f["Heart_rate"].notna().astype(np.int8) | f["PumpRate"].notna().astype(np.int8)
        sp = f["svc_port"]
        f["SvcMqtt"] = (sp == C.MQTT_PORT).astype(np.int8)
        f["SvcHl7"] = (sp == C.HL7_PORT).astype(np.int8)
        f["SvcHttps"] = (sp == C.HTTPS_PORT).astype(np.int8)
        f["SvcOther"] = (~sp.isin(list(C.SERVICE_PORTS))).astype(np.int8)

        f[C.ALL_FEATURES] = f[C.ALL_FEATURES].astype(float).fillna(0.0)
        f["device_id"] = f["device_id"].fillna("")
        out = f[META_COLS + C.ALL_FEATURES].sort_values(["window", "init_ip", "resp_ip", "svc_port"]).reset_index(drop=True)
        return out.copy()  # consolidated frame (cheaper for callers that add or merge columns)

    # ------------------------------------------------------------------
    def _orientation(self, pk: pd.DataFrame) -> np.ndarray:
        """True where the packet travels from the conversation initiator to the responder.

        The initiator of a TCP conversation is the sender of its first SYN; when the SYN is
        not in view (long-lived MQTT sessions, chunk boundaries) the endpoint with the lower
        port is taken as the server, which holds for every service in the ward.
        """
        ep_s = pk["src_ip"] + ":" + pk["sport"].astype(str)
        ep_d = pk["dst_ip"] + ":" + pk["dport"].astype(str)
        conv = np.where(ep_s < ep_d, ep_s + "|" + ep_d, ep_d + "|" + ep_s)
        conv = pd.Series(conv, index=pk.index)
        fwd = (pk["dport"] <= pk["sport"]).to_numpy()
        syn = pk["flags"] == "S"
        if syn.any():
            init = pd.Series(ep_s[syn].to_numpy(), index=conv[syn].to_numpy())
            init = init[~init.index.duplicated(keep="first")]
            known = conv.map(init)
            has = known.notna().to_numpy()
            fwd = np.where(has, (ep_s == known).to_numpy(), fwd)
        return fwd

    def _command_deltas(self, pk: pd.DataFrame, telem: pd.Series) -> pd.Series:
        """Relative change requested by each pump command versus the pump's last reported rate."""
        out = pd.Series(np.nan, index=pk.index)
        tel = pk.loc[telem & pk["pump_rate"].notna(), ["ts", "device_id", "pump_rate"]]
        cmd = pk.loc[(pk["mqtt"] == "CMD") & pk["pump_rate"].notna() & (pk["device_id"] != ""), ["ts", "device_id", "pump_rate"]]
        if len(cmd):
            m = pd.merge_asof(cmd.reset_index().sort_values("ts"),
                              tel.rename(columns={"pump_rate": "last_rate"}).sort_values("ts"),
                              on="ts", by="device_id", direction="backward")
            fallback = m["device_id"].map(self.pump_last)
            last = m["last_rate"].fillna(fallback)
            delta = (m["pump_rate"] - last).abs() / np.maximum(last.fillna(0), 10.0)
            out.loc[m["index"].to_numpy()] = delta.fillna(0).to_numpy()
        if len(tel):
            for dev, rate in tel.sort_values("ts").groupby("device_id")["pump_rate"].last().items():
                self.pump_last[dev] = float(rate)
        return out

    def _partner(self, dev):
        if not isinstance(dev, str) or "-" not in dev:
            return None
        kind, num = dev.split("-", 1)
        return {"MON": "WRB-", "WRB": "MON-"}.get(kind, "") + num if kind in ("MON", "WRB") else None

    def _deltas(self, f: pd.DataFrame) -> pd.DataFrame:
        key_cols = ["init_ip", "resp_ip", "svc_port", "proto"]
        f = f.sort_values(key_cols + ["window"])
        grp = f.groupby(key_cols, sort=False)
        prev_hr = grp["Heart_rate"].shift()
        prev_sp = grp["SpO2"].shift()
        pr_ff = grp["PumpRate"].ffill()
        prev_pr = pr_ff.groupby([f[c] for c in key_cols], sort=False).shift()
        first = grp.cumcount() == 0
        if self.state and first.any():
            keys = list(zip(*(f.loc[first, c] for c in key_cols)))
            st = [self.state.get(k, (np.nan, np.nan, np.nan)) for k in keys]
            st = np.array(st, dtype=float).reshape(-1, 3)
            prev_hr.loc[first] = st[:, 0]
            prev_sp.loc[first] = st[:, 1]
            prev_pr.loc[first] = st[:, 2]
        f["VitalsDelta"] = ((f["Heart_rate"] - prev_hr).abs().fillna(0) + (f["SpO2"] - prev_sp).abs().fillna(0))
        rel = (f["PumpRate"] - prev_pr).abs() / np.maximum(prev_pr.fillna(0), 10.0)
        rel = rel.where(f["PumpRate"].notna(), 0).fillna(0)
        # a command's requested change counts as soon as the command is seen
        f["PumpRateDelta"] = np.maximum(rel, f["CmdDelta"].fillna(0))
        # update state with the last observation of each key
        last = f.assign(_pr=pr_ff).groupby(key_cols, sort=False).agg(
            hr=("Heart_rate", "last"), sp=("SpO2", "last"), pr=("_pr", "last"))
        for k, row in zip(last.index, last.itertuples(index=False)):
            old = self.state.get(k, (np.nan, np.nan, np.nan))
            self.state[k] = (row.hr if pd.notna(row.hr) else old[0], row.sp if pd.notna(row.sp) else old[1],
                             row.pr if pd.notna(row.pr) else old[2])
        return f


def packets_to_features(packets: pd.DataFrame, registry: pd.DataFrame, window: float = C.WINDOW_SECONDS):
    clean, report = clean_packets(packets)
    fx = FeatureExtractor(registry, window)
    feats = fx.transform(clean)
    report["flows"] = int(len(feats))
    report["attack_flows"] = int(feats["label"].sum())
    return feats, report
