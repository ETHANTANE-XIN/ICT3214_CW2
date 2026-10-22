"""Shared paths, group details and result loaders for the document build scripts."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "MedTwinIDS"
RES = PROJ / "results"
FIG = RES / "figures"
DATA = PROJ / "data"
OUT = ROOT / "deliverables"
TEMPLATE = ROOT / "build" / "template_transitional.docx"

TITLE = "MedTwin-IDS: AI-Driven Intrusion Detection and Device Risk Scoring for a Medical IoT Digital Twin"
GROUP_ID = "[Group ID]"
# Replace the placeholders below with your group's details before submission.
MEMBERS = [
    dict(name="[Member 1 Full Name]", sid="[Student ID]", email="[SIT email]"),
    dict(name="[Member 2 Full Name]", sid="[Student ID]", email="[SIT email]"),
    dict(name="[Member 3 Full Name]", sid="[Student ID]", email="[SIT email]"),
    dict(name="[Member 4 Full Name]", sid="[Student ID]", email="[SIT email]"),
    dict(name="[Member 5 Full Name]", sid="[Student ID]", email="[SIT email]"),
]
FOOTER = f"ICT3214 Security Analytics, Coursework 2, Group {GROUP_ID}"


def report(name):
    return json.loads((DATA / f"{name}_report.json").read_text())


def comparison():
    return pd.read_csv(RES / "model_comparison.csv").set_index("model")


def card():
    return json.loads((PROJ / "models" / "model_card.json").read_text())


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"
