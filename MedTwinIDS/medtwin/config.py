"""Central configuration for the MedTwin-IDS digital twin and analytics pipeline.

All addresses use private (RFC 1918) or documentation (RFC 5737) ranges so that
the simulated data can never be confused with real hospital traffic.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RUNTIME_DIR = ROOT / "runtime"

# --------------------------------------------------------------------------
# Ward digital twin layout
# --------------------------------------------------------------------------
N_BEDS = 8
WINDOW_SECONDS = 5.0  # default flow aggregation window

BROKER_IP = "10.10.0.10"        # IoMT gateway running the MQTT broker
NURSE_STATION_IP = "10.10.0.20"  # central monitoring station (subscriber, pump commands)
EHR_IP = "10.10.0.30"            # electronic health record / HL7 interface engine
UPDATE_SERVER_IP = "10.10.0.40"  # vendor firmware repository mirror
WORKSTATION_IPS = ["10.10.0.51", "10.10.0.52"]
ROGUE_HOST_IP = "10.10.9.66"     # unregistered host used by some attack campaigns
EXTERNAL_IPS = ["203.0.113.45", "203.0.113.77", "198.51.100.23"]  # RFC 5737 ranges

INTERNAL_PREFIX = "10.10."

# Service ports
MQTT_PORT = 1883
HL7_PORT = 2575
HTTPS_PORT = 443
SERVICE_PORTS = {MQTT_PORT: "mqtt", HL7_PORT: "hl7", HTTPS_PORT: "https"}

# Asset criticality used by the risk engine (0..1)
CRITICALITY = {
    "infusion_pump": 1.00,
    "patient_monitor": 0.90,
    "gateway": 1.00,
    "nurse_station": 0.85,
    "ehr_server": 0.90,
    "wearable": 0.60,
    "workstation": 0.50,
    "update_server": 0.60,
    "unknown": 0.70,
}

# Telemetry publish period (seconds) per device class
PUBLISH_PERIOD = {"patient_monitor": 1.0, "wearable": 2.0, "infusion_pump": 5.0}

# Registered IP TTL per device class (the twin stores the expected value)
DEVICE_TTL = {
    "patient_monitor": 64,
    "wearable": 64,
    "infusion_pump": 255,
    "gateway": 64,
    "nurse_station": 128,
    "ehr_server": 64,
    "workstation": 128,
    "update_server": 64,
}

# --------------------------------------------------------------------------
# Attack catalogue (label names used throughout the project)
# --------------------------------------------------------------------------
NORMAL = "normal"
ATTACKS = [
    "dos_flood",            # MQTT/TCP flooding of the gateway
    "spoofing",             # device impersonation with falsified telemetry
    "data_alteration",      # man-in-the-middle modification of vitals in transit
    "unauthorised_access",  # credential guessing then unauthorised pump command
    "reconnaissance",       # host and port scanning of the ward subnet
    "exfiltration",         # bulk transfer of patient data to an external host
]
CLASSES = [NORMAL] + ATTACKS

# Severity weights for the risk engine
SEVERITY = {
    NORMAL: 0.0,
    "dos_flood": 0.80,
    "spoofing": 0.90,
    "data_alteration": 1.00,
    "unauthorised_access": 1.00,
    "reconnaissance": 0.50,
    "exfiltration": 0.90,
    "twin_rule": 0.70,        # a label-free twin consistency rule fired (possibly an unseen attack)
    "anomaly_unknown": 0.60,  # only the Isolation Forest flagged the flow
}

# Analyst playbook shown with each alert
PLAYBOOK = {
    "dos_flood": "Rate-limit the source at the gateway, move the source device to a quarantine VLAN and confirm bedside monitoring is still reaching the nurse station.",
    "spoofing": "Verify vitals at the bedside, block the duplicate publisher, re-key the device certificate and enforce MQTT client authentication per device.",
    "data_alteration": "Treat displayed vitals as untrusted, check bedside readings, inspect ARP tables for poisoning and enable TLS/integrity protection on the telemetry path.",
    "unauthorised_access": "Lock the targeted pump, confirm the infusion rate manually, rotate broker credentials and restrict command topics to the nurse station.",
    "reconnaissance": "Identify and isolate the scanning host, review firewall rules for the device VLAN and watch for follow-on exploitation.",
    "exfiltration": "Block the external destination, isolate the source host, preserve logs for forensics and start the data breach response process.",
    "twin_rule": "The device broke a digital-twin consistency rule (identity, sequence or physiology). Verify the device and the patient at the bedside before trusting its data.",
    "anomaly_unknown": "Unusual behaviour not matching a known attack. Review the flow and the device's recent activity.",
}

# --------------------------------------------------------------------------
# Feature groups (produced by medtwin.parser)
# --------------------------------------------------------------------------
NETWORK_FEATURES = [
    "Dur", "SrcPkts", "DstPkts", "TotPkts", "SrcBytes", "DstBytes", "TotBytes",
    "SrcLoad", "DstLoad", "Rate", "SIntPkt", "DIntPkt", "SrcJitter", "DstJitter",
    "sMaxPktSz", "sMinPktSz", "sMeanPktSz", "dMaxPktSz", "dMinPktSz",
    "SynCnt", "RstCnt", "FinCnt", "ArpCnt", "Retrans",
    "MqttConnect", "MqttConnFail", "MqttPublish", "MqttCmd",
    "SrcFanOut", "SrcPortSpread", "SvcMqtt", "SvcHl7", "SvcHttps", "SvcOther",
]
BIOMETRIC_FEATURES = [
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST",
    "PumpRate", "HasVitals",
]
TWIN_FEATURES = [
    "SrcKnown", "MacMismatch", "TTLDev", "ExternalDst", "SeqDup", "SeqGap",
    "TwinResidualHR", "TwinResidualSpO2", "VitalsDelta", "VitalsStd", "PumpRateDelta",
]
ALL_FEATURES = NETWORK_FEATURES + BIOMETRIC_FEATURES + TWIN_FEATURES

FEATURE_SETS = {
    "network_only": NETWORK_FEATURES,
    "biometric_only": BIOMETRIC_FEATURES,
    "network_biometric": NETWORK_FEATURES + BIOMETRIC_FEATURES,
    "full_twin": ALL_FEATURES,
}


@dataclass
class ScenarioConfig:
    """Parameters for one simulation run."""
    seed: int = 7
    hours: float = 4.0
    start_hour: float = 8.0          # wall-clock hour at simulation start
    campaigns_per_hour: float = 7.0   # attack campaigns per simulated hour
    clinical_events_per_hour: float = 3.0
    congestion_events_per_hour: float = 1.5
    firmware_updates_per_hour: float = 0.6
    reboots_per_hour: float = 1.0
    attack_types: list = field(default_factory=lambda: list(ATTACKS))
    intensities: tuple = ("low", "medium", "high")
