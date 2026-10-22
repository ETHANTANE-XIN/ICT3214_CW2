"""Per-device risk scoring, alert grouping and explanations for the live detector.

Risk model
----------
For every flow window the detector gives P(attack) and the most likely class c.
The instantaneous threat of a flow is T = P(attack) * severity(c). Each device
involved in the flow (initiator, responder and the device whose identity the
telemetry claims) receives

    R_t = min(100, max(100 * T * w, lambda * R_{t-1}))

where w = 0.6 + 0.4 * criticality (from the digital twin asset registry) and
lambda = 0.9 per window. Risk therefore jumps immediately on strong evidence and
decays with a half-life of roughly 6.5 windows (about 33 s) once traffic is clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .rules import RULES

LEVELS = [(80, "Critical"), (60, "High"), (30, "Medium"), (0, "Low")]


def level(score: float) -> str:
    for thr, name in LEVELS:
        if score >= thr:
            return name
    return "Low"


class Explainer:
    """Explains a flagged flow by its most abnormal features relative to normal training traffic."""

    def __init__(self, normal_stats: pd.DataFrame, importance: dict | None = None, top_k: int = 3):
        self.mu = normal_stats["mean"]
        self.sd = normal_stats["std"].replace(0, np.nan)
        self.importance = importance or {}
        self.top_k = top_k

    # indicators that only describe the flow type and carry no evidence on their own
    UNINFORMATIVE = {"HasVitals", "SvcMqtt", "SvcHl7", "SvcHttps"}

    def explain(self, row: pd.Series) -> list[dict]:
        feats = [f for f in self.mu.index if f not in self.UNINFORMATIVE]
        z = ((row[feats].astype(float) - self.mu[feats]) / self.sd[feats]).abs().fillna(0)
        # binary indicators deviate by definition; give them a fixed strong score when set
        for f in ("MacMismatch", "ExternalDst", "SvcOther"):
            if f in z and row[f] > 0 and self.mu[f] < 0.05:
                z[f] = 10.0
        if self.importance:
            z = z * pd.Series({f: 0.5 + self.importance.get(f, 0.0) for f in feats})
        top = z.sort_values(ascending=False).head(self.top_k)
        return [dict(feature=f, value=round(float(row[f]), 3), normal_mean=round(float(self.mu[f]), 3)) for f in top.index]


class RiskEngine:
    def __init__(self, registry: pd.DataFrame, decay: float = 0.9, threshold: float = 0.5,
                 explainer: Explainer | None = None, incident_gap: float = 30.0):
        self.registry = registry.set_index("device_id")
        self.ip_to_dev = dict(zip(registry.ip, registry.device_id))
        self.decay, self.threshold, self.explainer, self.incident_gap = decay, threshold, explainer, incident_gap
        self.risk = {d: 0.0 for d in self.registry.index}
        self.last_class = {d: C.NORMAL for d in self.registry.index}
        self.incidents: list[dict] = []
        self.alert_count = 0

    def _devices(self, row, cls: str) -> set[str]:
        init_dev = self.ip_to_dev.get(row["init_ip"])
        devs = {init_dev, self.ip_to_dev.get(row["resp_ip"])}
        if isinstance(row.get("device_id"), str) and row["device_id"]:
            devs.add(row["device_id"])
        if init_dev is None:
            # unregistered host: track it as its own entity so analysts can see it
            devs.add(self._unknown(row["init_ip"]))
        devs.discard(None)
        # a MAC that does not match the registry means the claimed identity is being impersonated:
        # the owner of that IP is a victim, not the culprit, so do not raise its risk for this flow
        if row.get("MacMismatch", 0) > 0 and init_dev is not None:
            devs.discard(init_dev)
        # the gateway relays all telemetry; only raise its risk when it is the initiator or the target
        if row["init_ip"] != C.BROKER_IP and cls not in ("dos_flood", "reconnaissance", "unauthorised_access"):
            devs.discard("GW-01")
        return devs

    def _unknown(self, ip: str) -> str:
        dev = f"UNKNOWN {ip}"
        if dev not in self.risk:
            self.registry.loc[dev] = dict(type="unknown", ip=ip, mac="", bed=0, ttl=0,
                                          criticality=C.CRITICALITY["unknown"])
            self.risk[dev] = 0.0
            self.last_class[dev] = C.NORMAL
            self.ip_to_dev[ip] = dev
        return dev

    def update(self, flows: pd.DataFrame, proba: np.ndarray, iso_flag: np.ndarray | None, t_now: float, clock: str,
               rule_hits: pd.DataFrame | None = None):
        p_attack = 1.0 - proba[:, 0]
        rule_names = list(rule_hits.columns) if rule_hits is not None else []
        rule_arr = rule_hits.to_numpy() if rule_hits is not None else None
        cls_idx = proba[:, 1:].argmax(1) + 1  # most likely attack class
        threat_by_dev: dict[str, tuple[float, str]] = {}
        new_alerts = []
        for i, row in enumerate(flows.to_dict("records")):
            cls = C.CLASSES[int(cls_idx[i])]
            p = float(p_attack[i])
            anomaly = bool(iso_flag[i]) if iso_flag is not None else False
            fired = [r for j, r in enumerate(rule_names) if rule_arr[i, j]] if rule_arr is not None else []
            if p < self.threshold and fired:
                # a twin consistency rule caught what the supervised model passed (possibly an unseen attack)
                cls, p = "twin_rule", max(p, 0.6)
            elif p < self.threshold and anomaly:
                # unknown behaviour: flagged only by the unsupervised model
                cls, p = "anomaly_unknown", max(p, 0.5)
            threat = p * C.SEVERITY.get(cls, 0.6)
            for dev in self._devices(row, cls):
                if threat > threat_by_dev.get(dev, (0.0, ""))[0]:
                    threat_by_dev[dev] = (threat, cls)
            if p >= self.threshold:
                expl = self.explainer.explain(pd.Series(row)) if self.explainer else []
                if cls == "twin_rule":
                    expl = [dict(feature=r, value=RULES[r], normal_mean="rule") for r in fired] + expl[:1]
                culprit = self.ip_to_dev.get(row["init_ip"], row["init_ip"])
                if row.get("MacMismatch", 0) > 0:
                    culprit = self.ip_to_dev.get(row["resp_ip"], row["resp_ip"])
                new_alerts.append(dict(t=round(t_now, 1), clock=clock, src=row["init_ip"], dst=row["resp_ip"],
                                       port=int(row["svc_port"]), device=row.get("device_id") or culprit,
                                       attack=cls, confidence=round(p, 3), explanation=expl,
                                       action=C.PLAYBOOK.get(cls, "Investigate the flagged host and correlate with other logs."),
                                       truth=row.get("attack", "")))
        for dev in self.risk:
            crit = float(self.registry.loc[dev, "criticality"])
            t, cls = threat_by_dev.get(dev, (0.0, C.NORMAL))
            inst = 100.0 * t * (0.6 + 0.4 * crit)
            self.risk[dev] = float(min(100.0, max(inst, self.decay * self.risk[dev])))
            if t > 0.25:
                self.last_class[dev] = cls
        self._group(new_alerts)
        self.alert_count += len(new_alerts)
        return new_alerts

    def _group(self, alerts):
        """Merge alerts into incidents keyed by (device, attack) to reduce alert fatigue."""
        for a in alerts:
            key = (a["device"], a["attack"])
            inc = next((x for x in reversed(self.incidents) if (x["device"], x["attack"]) == key
                        and a["t"] - x["last_t"] <= self.incident_gap), None)
            if inc is None:
                inc = dict(id=len(self.incidents) + 1, device=a["device"], attack=a["attack"], first_t=a["t"],
                           first_clock=a["clock"], last_t=a["t"], last_clock=a["clock"], alerts=0, max_conf=0.0,
                           src=a["src"], dst=a["dst"], explanation=a["explanation"], action=a["action"], truth=set())
                self.incidents.append(inc)
            inc["alerts"] += 1
            inc["last_t"], inc["last_clock"] = a["t"], a["clock"]
            if a["confidence"] >= inc["max_conf"]:
                inc["max_conf"], inc["explanation"] = a["confidence"], a["explanation"]
            if a["truth"]:
                inc["truth"].add(a["truth"])

    def device_table(self) -> list[dict]:
        rows = []
        for dev, r in self.registry.iterrows():
            score = round(self.risk[dev], 1)
            rows.append(dict(device_id=dev, type=r["type"], bed=int(r["bed"]), ip=r["ip"], risk=score,
                             level=level(score), last_class=self.last_class[dev] if score >= 30 else C.NORMAL))
        return rows
