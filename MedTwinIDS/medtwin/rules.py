"""Label-free digital-twin consistency rules.

The rules encode what the twin knows must hold in a healthy ward: every sender is in the asset
registry, a device's MAC and TTL match the registry, message sequence numbers advance without
duplicates or gaps, the monitor and the wearable on the same patient agree, and pump rates only
change by small steps. Thresholds are the 99.9th percentile of each signal in *normal* training
traffic, so no attack labels are needed and the rules apply to attacks never seen before.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

QUANTILE = 0.999
RULES = {
    "R1": "unregistered sender",
    "R2": "MAC or TTL differs from registry",
    "R3": "sequence numbers duplicated or missing",
    "R4": "monitor and wearable disagree",
    "R5": "unusual pump-rate change",
}
SIGNALS = ["SeqDup", "SeqGap", "TwinResidualHR", "TwinResidualSpO2", "PumpRateDelta"]


class TwinRules:
    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or {}

    def fit(self, flows: pd.DataFrame) -> "TwinRules":
        normal = flows[flows["attack"] == C.NORMAL] if "attack" in flows else flows
        self.thresholds = {f: float(normal[f].quantile(QUANTILE)) for f in SIGNALS}
        return self

    def evaluate(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Boolean frame with one column per rule."""
        t = self.thresholds
        return pd.DataFrame({
            "R1": flows["SrcKnown"].to_numpy() == 0,
            "R2": (flows["MacMismatch"].to_numpy() > 0) | (flows["TTLDev"].to_numpy() > 0),
            "R3": (flows["SeqDup"].to_numpy() > t["SeqDup"]) | (flows["SeqGap"].to_numpy() > t["SeqGap"]),
            "R4": (flows["TwinResidualHR"].to_numpy() > t["TwinResidualHR"])
                  | (flows["TwinResidualSpO2"].to_numpy() > t["TwinResidualSpO2"]),
            "R5": flows["PumpRateDelta"].to_numpy() > t["PumpRateDelta"],
        }, index=flows.index)

    def flag(self, flows: pd.DataFrame) -> np.ndarray:
        return self.evaluate(flows).any(axis=1).to_numpy()

    def save(self, path):
        path.write_text(json.dumps({"quantile": QUANTILE, "thresholds": self.thresholds, "rules": RULES}, indent=2))

    @classmethod
    def load(cls, path) -> "TwinRules":
        return cls(json.loads(path.read_text())["thresholds"])
