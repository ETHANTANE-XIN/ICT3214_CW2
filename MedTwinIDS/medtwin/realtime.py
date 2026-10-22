"""Real-time detection over a live digital-twin stream (or a replayed packet log).

Every window (default 5 s of ward time) the loop:
  1. pulls new packet events from the twin (or the replay file),
  2. cleans them and extracts flow features with the same FeatureExtractor used offline,
  3. scores each flow with the selected model and the Isolation Forest safety net,
  4. updates per-device risk scores and incidents,
  5. writes runtime/state.json, runtime/alerts.jsonl and runtime/timeline.csv for the dashboard.

Examples
--------
python -m medtwin.realtime --minutes 20 --speed 10            # live twin, 10x faster than real time
python -m medtwin.realtime --replay test --speed 50           # replay the independent test run
python -m medtwin.realtime --replay test --minutes 180 --speed 0 --quiet --save-as replay        # E6 (report)
python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --quiet --save-as live  # E6 (report)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .attacks import campaign_table
from .models import full_proba
from .parser import FeatureExtractor, clean_packets
from .risk import Explainer, RiskEngine
from .rules import TwinRules
from .twin import WardTwin

COLOURS = {"Critical": "\033[91m", "High": "\033[93m", "Medium": "\033[96m", "Low": "\033[92m"}


def _atomic_write(path, text):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def load_detector():
    card = json.loads((C.MODEL_DIR / "model_card.json").read_text())
    model = joblib.load(C.MODEL_DIR / "best_model.joblib")
    iso, iso_thr = None, None
    if (C.MODEL_DIR / "isolation_forest.joblib").exists():
        iso = joblib.load(C.MODEL_DIR / "isolation_forest.joblib")
        thr_file = C.MODEL_DIR / "iforest_threshold.json"
        if thr_file.exists():
            iso_thr = json.loads(thr_file.read_text())["strict_threshold"]
    stats_file = C.MODEL_DIR / "normal_stats.csv"
    explainer = None
    if stats_file.exists():
        stats = pd.read_csv(stats_file, index_col=0)
        imp = {}
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(est, "feature_importances_"):
            v = np.asarray(est.feature_importances_, dtype=float)
            imp = dict(zip(card["features"], v / v.max()))
        explainer = Explainer(stats, imp)
    return card, model, iso, iso_thr, explainer


def clock_str(start_hour: float, t: float) -> str:
    s = int(start_hour * 3600 + t) % 86400
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def run(args):
    C.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    card, model, iso, iso_thr, explainer = load_detector()
    features, W = card["features"], float(card["window_seconds"])
    rules = None
    if args.twin_rules and (C.MODEL_DIR / "twin_rules.json").exists():
        rules = TwinRules.load(C.MODEL_DIR / "twin_rules.json")

    if args.replay:
        packets = pd.read_parquet(C.DATA_DIR / f"{args.replay}_packets.parquet")
        registry = pd.read_csv(C.DATA_DIR / "devices.csv")
        truth = pd.read_csv(C.DATA_DIR / f"{args.replay}_campaigns.csv")
        start_hour = {"train": 8.0, "test": 20.0}.get(args.replay, 0.0)
        duration = min(float(packets["ts"].max()), args.minutes * 60 if args.minutes else 1e12)
        twin = None
    else:
        cfg = C.ScenarioConfig(seed=args.seed, hours=args.minutes / 60.0, start_hour=args.start_hour,
                               campaigns_per_hour=args.campaigns_per_hour)
        twin = WardTwin(cfg)
        registry = twin.registry
        truth = campaign_table(twin.campaigns)
        start_hour, duration = args.start_hour, twin.duration

    if args.stop_at:
        duration = min(duration, args.stop_at * 60.0)  # keep the full schedule, stop early (e.g. for screenshots)
    truth.to_csv(C.RUNTIME_DIR / "ground_truth.csv", index=False)
    fx = FeatureExtractor(registry, W)
    engine = RiskEngine(registry, threshold=card.get("alert_threshold", 0.5), explainer=explainer)
    alerts_path = C.RUNTIME_DIR / "alerts.jsonl"
    alerts_path.write_text("")
    timeline_rows, latencies, flows_total = [], [], 0
    vitals_latest: dict = {}
    carry = None
    t, t_wall0 = 0.0, time.time()
    print(f"MedTwin-IDS live detector | model={card['selected_model']} | window={W:.0f}s | speed x{args.speed}")
    while t < duration:
        loop_start = time.time()
        t1 = min(t + W, duration)
        if twin is not None:
            pk = twin.step(t, t1)
        else:
            pk = packets[(packets["ts"] >= t) & (packets["ts"] < t1)]
        if carry is not None and len(carry):
            pk = pd.concat([carry, pk], ignore_index=True)
        carry = pk[pk["ts"] >= t1]
        pk = pk[pk["ts"] < t1]

        proc_start = time.perf_counter()
        clean, _ = clean_packets(pk) if len(pk) else (pk, {})
        flows = fx.transform(clean) if len(clean) else pd.DataFrame()
        clock = clock_str(start_hour, t1)
        new_alerts = []
        if len(flows):
            X = flows[features].to_numpy(dtype=np.float32)
            proba = full_proba(model, X)
            iso_flag = None
            if iso is not None and args.hybrid:
                if iso_thr is not None:
                    iso_flag = (iso.score_samples(X) < iso_thr).astype(int)
                else:
                    iso_flag = (iso.predict(X) == -1).astype(int)
            rule_hits = rules.evaluate(flows) if rules is not None else None
            new_alerts = engine.update(flows, proba, iso_flag, t1, clock, rule_hits)
            flows_total += len(flows)
            # latest vitals per bed for the dashboard
            tel = flows[flows["device_id"].str.startswith("MON") & (flows["Heart_rate"] > 0)]
            for r in tel.itertuples():
                vitals_latest[int(r.bed)] = dict(hr=round(r.Heart_rate, 1), spo2=round(r.SpO2, 1), temp=round(r.Temp, 2),
                                                 sys=round(r.SYS, 1), dia=round(r.DIA, 1), rr=round(r.Resp_Rate, 1))
            pumps = flows[flows["device_id"].str.startswith("PMP") & (flows["HasVitals"] > 0)]
            for r in pumps.itertuples():
                vitals_latest.setdefault(int(r.bed), {})["pump_rate"] = round(r.PumpRate, 1)
        else:
            engine.update(pd.DataFrame(columns=["init_ip", "resp_ip", "svc_port", "device_id"]), np.zeros((0, len(C.CLASSES))), None, t1, clock)
        proc_ms = (time.perf_counter() - proc_start) * 1000
        latencies.append(proc_ms)

        with open(alerts_path, "a") as fh:
            for a in new_alerts:
                fh.write(json.dumps(a) + "\n")
        devices = engine.device_table()
        for d in devices:
            timeline_rows.append(dict(t=round(t1, 1), clock=clock, device_id=d["device_id"], risk=d["risk"]))
        active_truth = truth[(truth["start"] <= t1) & (truth["end"] >= t)]
        incidents = [dict({k: v for k, v in inc.items() if k != "truth"}, truth=sorted(inc["truth"]))
                     for inc in engine.incidents[-60:]]
        elapsed = max(1e-6, time.time() - t_wall0)
        state = dict(
            updated=datetime.now().isoformat(timespec="seconds"), sim_time=round(t1, 1), clock=clock,
            duration=duration, model=card["selected_model"], window=W, speed=args.speed,
            kpis=dict(flows_processed=flows_total, alerts_total=engine.alert_count, incidents=len(engine.incidents),
                      max_risk=max(d["risk"] for d in devices),
                      devices_at_risk=sum(d["risk"] >= 60 for d in devices),
                      avg_latency_ms=round(float(np.mean(latencies)), 1),
                      p95_latency_ms=round(float(np.percentile(latencies, 95)), 1),
                      throughput_fps=round(flows_total / elapsed, 1)),
            devices=devices, vitals={str(k): v for k, v in vitals_latest.items()}, incidents=incidents,
            ground_truth=active_truth.to_dict("records"),
        )
        _atomic_write(C.RUNTIME_DIR / "state.json", json.dumps(state, default=str))
        if len(timeline_rows) > 0 and (int(t1 / W) % 6 == 0 or t1 >= duration):
            pd.DataFrame(timeline_rows).to_csv(C.RUNTIME_DIR / "timeline.csv", index=False)

        if not args.quiet:
            for a in new_alerts[:3]:
                risk = engine.risk.get(a["device"], 0.0)
                col = COLOURS["Critical" if risk >= 80 else "High" if risk >= 60 else "Medium"]
                why = ", ".join(f"{e['feature']}={e['value']}" for e in a["explanation"])
                print(f"{col}[{clock}] ALERT {a['attack']:<20s} dev={a['device']:<8s} p={a['confidence']:.2f} "
                      f"risk={risk:5.1f} | {why}\033[0m")
        t = t1
        if args.speed > 0:
            time.sleep(max(0.0, W / args.speed - (time.time() - loop_start)))

    pd.DataFrame(timeline_rows).to_csv(C.RUNTIME_DIR / "timeline.csv", index=False)
    # only campaigns that started before the run ended can be detected (relevant with --stop-at)
    summary = summarise(engine, truth[truth["start"] < duration], latencies, flows_total, time.time() - t_wall0, W)
    (C.RUNTIME_DIR / "live_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if args.save_as:
        # keep an evaluation run: runtime/ is overwritten by every later run (e.g. the dashboard demo)
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("live_summary.json", "live_campaigns.csv", "timeline.csv", "ground_truth.csv", "alerts.jsonl"):
            shutil.copy(C.RUNTIME_DIR / name, C.RESULTS_DIR / f"e6_{args.save_as}_{name}")
    print(json.dumps(summary["overall"], indent=2))
    return summary


def summarise(engine, truth, latencies, flows_total, wall_s, W):
    """Detection delay per campaign (first alert carrying that campaign's true class) and runtime stats."""
    alerts = [json.loads(l) for l in open(C.RUNTIME_DIR / "alerts.jsonl")] if (C.RUNTIME_DIR / "alerts.jsonl").exists() else []
    rows = []
    for c in truth.itertuples():
        hits = [a for a in alerts if c.start - W <= a["t"] <= c.end + 2 * W and a["truth"] == c.attack]
        first = min((a["t"] for a in hits), default=None)
        correct = [a for a in hits if a["attack"] == c.attack]
        rows.append(dict(campaign=c.campaign, attack=c.attack, intensity=c.intensity, detected=first is not None,
                         delay_s=round(first - c.start, 1) if first is not None else None,
                         alerts=len(hits), class_correct=round(len(correct) / len(hits), 3) if hits else None))
    per = pd.DataFrame(rows)
    per.to_csv(C.RUNTIME_DIR / "live_campaigns.csv", index=False)
    false_alerts = sum(1 for a in alerts if a["truth"] == C.NORMAL)
    overall = dict(campaigns=len(per), detected=int(per["detected"].sum()) if len(per) else 0,
                   detection_rate=round(float(per["detected"].mean()), 3) if len(per) else None,
                   mean_delay_s=round(float(per["delay_s"].dropna().mean()), 1) if per["delay_s"].notna().any() else None,
                   median_delay_s=round(float(per["delay_s"].dropna().median()), 1) if per["delay_s"].notna().any() else None,
                   alerts=len(alerts), false_alerts=false_alerts, incidents=len(engine.incidents),
                   flows=flows_total, avg_window_latency_ms=round(float(np.mean(latencies)), 1),
                   p95_window_latency_ms=round(float(np.percentile(latencies, 95)), 1),
                   wall_seconds=round(wall_s, 1))
    return dict(overall=overall, per_campaign=rows)


def main():
    ap = argparse.ArgumentParser(description="MedTwin-IDS real-time detector")
    ap.add_argument("--minutes", type=float, default=20.0, help="minutes of ward time to simulate/replay")
    ap.add_argument("--speed", type=float, default=10.0, help="time acceleration (1 = real time, 0 = as fast as possible)")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--start-hour", type=float, default=14.0)
    ap.add_argument("--campaigns-per-hour", type=float, default=12.0)
    ap.add_argument("--replay", choices=["train", "test"], help="replay a stored packet log instead of the live twin")
    ap.add_argument("--no-hybrid", dest="hybrid", action="store_false", help="disable the Isolation Forest safety net")
    ap.add_argument("--twin-rules", action="store_true", help="also raise alerts from the label-free twin consistency rules")
    ap.add_argument("--stop-at", type=float, help="stop after this many minutes but keep the --minutes schedule")
    ap.add_argument("--save-as", help="also copy the run's outputs to results/e6_<name>_* (e.g. replay, live)")
    ap.add_argument("--quiet", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
