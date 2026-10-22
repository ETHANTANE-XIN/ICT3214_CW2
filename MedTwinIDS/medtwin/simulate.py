"""Generate datasets from the ward digital twin.

Examples
--------
python -m medtwin.simulate --preset all            # train + test runs used in the report
python -m medtwin.simulate --name demo --hours 0.5 --seed 11
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import config as C
from .attacks import campaign_table
from .parser import packets_to_features
from .twin import WardTwin

PRESETS = {
    # independent runs: different seed (patients, devices, attack timing) and time of day
    "train": dict(seed=7, hours=6.0, start_hour=8.0, campaigns_per_hour=10.0),
    "test": dict(seed=2026, hours=3.0, start_hour=20.0, campaigns_per_hour=10.0),
}


def run_simulation(cfg: C.ScenarioConfig, chunk: float = 5.0, verbose: bool = True):
    tw = WardTwin(cfg)
    frames, t, t_wall = [], 0.0, time.time()
    n_chunks = int(tw.duration // chunk) + 1
    for i in range(n_chunks):
        if t >= tw.duration:
            break
        frames.append(tw.step(t, min(t + chunk, tw.duration)))
        t += chunk
        if verbose and i % 720 == 0:
            print(f"  simulated {t / 3600:5.2f} h of {cfg.hours:.2f} h ({time.time() - t_wall:5.1f} s)")
    packets = pd.concat(frames, ignore_index=True)
    return tw, packets


def generate(name: str, cfg: C.ScenarioConfig, window: float = C.WINDOW_SECONDS, out_dir=C.DATA_DIR,
             save_packets: bool = True, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[{name}] seed={cfg.seed} hours={cfg.hours} start={cfg.start_hour:02.0f}:00")
    tw, packets = run_simulation(cfg, verbose=verbose)
    tw.registry.to_csv(out_dir / "devices.csv", index=False)
    campaigns = campaign_table(tw.campaigns)
    campaigns.to_csv(out_dir / f"{name}_campaigns.csv", index=False)
    pd.DataFrame(tw.benign_events).drop(columns=["sport"], errors="ignore").to_csv(
        out_dir / f"{name}_benign_events.csv", index=False)
    if save_packets:
        packets.to_parquet(out_dir / f"{name}_packets.parquet", index=False)
    feats, report = packets_to_features(packets, tw.registry, window)
    feats.to_csv(out_dir / f"{name}_flows.csv", index=False)
    report.update(name=name, seed=cfg.seed, hours=cfg.hours, window=window, packets=int(len(packets)),
                  attack_packets=int(packets["label"].sum()), campaigns=int(len(campaigns)))
    report["class_counts"] = feats["attack"].value_counts().to_dict()
    (out_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"[{name}] packets={len(packets):,} flows={len(feats):,} attack flows={int(feats['label'].sum()):,}")
        print(feats["attack"].value_counts().to_string())
    return packets, feats, report


def main():
    ap = argparse.ArgumentParser(description="Generate MedTwin-IDS datasets from the ward digital twin")
    ap.add_argument("--preset", choices=["all", "train", "test"], help="use the report configuration")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--start-hour", type=float, default=8.0)
    ap.add_argument("--campaigns-per-hour", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=C.WINDOW_SECONDS)
    ap.add_argument("--no-packets", action="store_true", help="do not save the raw packet log")
    args = ap.parse_args()

    if args.preset:
        names = ["train", "test"] if args.preset == "all" else [args.preset]
        for n in names:
            generate(n, C.ScenarioConfig(**PRESETS[n]), window=args.window, save_packets=not args.no_packets)
    else:
        cfg = C.ScenarioConfig(seed=args.seed, hours=args.hours, start_hour=args.start_hour,
                               campaigns_per_hour=args.campaigns_per_hour)
        generate(args.name, cfg, window=args.window, save_packets=not args.no_packets)


if __name__ == "__main__":
    main()
