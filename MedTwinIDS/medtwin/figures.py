"""Static figures for the report (python -m medtwin.figures)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402

# Validated categorical palette (fixed order) and chart chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOUR = {"Random Forest": SERIES[0], "XGBoost": SERIES[1], "MLP": SERIES[2]}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SHORT = {"normal": "Normal", "dos_flood": "DoS", "spoofing": "Spoofing", "data_alteration": "Data alt.",
         "unauthorised_access": "Unauth.", "reconnaissance": "Recon.", "exfiltration": "Exfil."}
COL_W, FULL_W = 3.45, 7.1

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def _read(name, **kw):
    p = C.RESULTS_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


# --------------------------------------------------------------------------
def model_comparison():
    comp = _read("model_comparison.csv")
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "Binary F1"),
               ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for j, (_, r) in enumerate(comp.iterrows()):
        ys = np.arange(len(metrics)) + (j - 1) * 0.22
        ax.scatter([r[m] for m, _ in metrics], ys, s=26, color=MODEL_COLOUR[r["model"]], label=r["model"],
                   zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(metrics)), [l for _, l in metrics])
    ax.invert_yaxis()
    lo = min(comp[m].min() for m, _ in metrics)
    ax.set_xlim(max(0.0, lo - 0.03), 1.002)
    ax.set_xlabel("Score on independent test run (dot plot; axis does not start at 0)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    _save(fig, "fig_model_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    for ax, (col, title, fmt) in zip(axes, [("train_s", "Training time (s)", "{:.0f}"),
                                            ("latency_ms", "Single-flow latency (ms)", "{:.1f}")]):
        bars = ax.bar(comp["model"], comp[col], color=[MODEL_COLOUR[m] for m in comp["model"]], width=0.6)
        for b, v in zip(bars, comp[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_title(title)
        ax.set_xticks(range(len(comp)), [m.replace("Random Forest", "RF") for m in comp["model"]])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, comp[col].max() * 1.22)
    fig.tight_layout()
    _save(fig, "fig_model_cost.png")


def confusion():
    comp = _read("model_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45))
    for ax, m in zip(axes, comp["model"]):
        cm = _read(f"confusion_{m.lower().replace(' ', '_')}.csv", index_col=0)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0).to_numpy()
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                if norm[i, j] >= 0.005:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                            color="white" if norm[i, j] > 0.55 else INK)
        labels = [SHORT[c] for c in cm.index]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels))
        ax.set_title(m)
        ax.grid(False)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True class")
    fig.tight_layout()
    _save(fig, "fig_confusion.png")


def per_class():
    comp = _read("model_comparison.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    w = 0.26
    for j, m in enumerate(comp["model"]):
        pc = _read(f"per_class_{m.lower().replace(' ', '_')}.csv")
        x = np.arange(len(pc)) + (j - 1) * w
        ax.bar(x, pc["f1"], width=w - 0.02, color=MODEL_COLOUR[m], label=m)
    ax.set_xticks(range(len(pc)), [SHORT[c] for c in pc["cls"]], rotation=30, ha="right")
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    _save(fig, "fig_per_class.png")


def ablation():
    ab = _read("e2_ablation.csv")
    if ab is None:
        return
    sets = list(C.FEATURE_SETS)
    names = {"network_only": "Network", "biometric_only": "Biometric", "network_biometric": "Net + bio",
             "full_twin": "Full twin"}
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
    for ax, (col, title) in zip(axes, [("macro_f1", "Macro F1 (all classes)"), ("recall_spoofing", "Recall: spoofing")]):
        w = 0.26
        for j, m in enumerate(["Random Forest", "XGBoost", "MLP"]):
            d = ab[ab["model"] == m].set_index("feature_set").loc[sets]
            ax.bar(np.arange(len(sets)) + (j - 1) * w, d[col], width=w - 0.02, color=MODEL_COLOUR[m], label=m)
        ax.set_xticks(range(len(sets)), [names[s] for s in sets], rotation=30, ha="right", fontsize=6.3)
        ax.set_title(title, fontsize=7.2)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, "fig_ablation.png")


def zero_day():
    zd = _read("e3_zero_day.csv")
    if zd is None:
        return
    best = json.loads((C.MODEL_DIR / "model_card.json").read_text())["selected_model"]
    order = ["Random Forest", "XGBoost", "MLP", "Isolation Forest", "Twin rules", f"Hybrid ({best} + IF)",
             f"Hybrid ({best} + twin rules)", f"Hybrid ({best} + IF + twin rules)"]
    order = [o for o in order if o in set(zd["detector"])]
    piv = zd.pivot(index="detector", columns="held_out", values="recall").loc[order, C.ATTACKS]
    fpr = zd.groupby("detector")["fpr"].mean().loc[order]
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    ax.imshow(piv.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(C.ATTACKS)), [SHORT[a] for a in C.ATTACKS], rotation=30, ha="right")
    short = lambda o: (o.replace("Random Forest", "RF").replace("Isolation Forest", "IF")
                       .replace("Hybrid (", "").rstrip(")"))
    ax.set_yticks(range(len(order)), [f"{short(o)}  (FPR {100 * fpr[o]:.2f}%)" for o in order], fontsize=6.3)
    ax.axhline(2.5, color="white", lw=2)
    ax.axhline(4.5, color="white", lw=2)
    ax.set_xlabel("Attack family held out of training")
    ax.grid(False)
    _save(fig, "fig_zero_day.png")


def intensity():
    it = _read("e4_intensity.csv")
    if it is None:
        return
    best = json.loads((C.MODEL_DIR / "model_card.json").read_text())["selected_model"]
    d = it[it["model"] == best]
    levels = ["low", "medium", "high"]
    shades = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    w = 0.26
    for j, lv in enumerate(levels):
        dd = d[d["intensity"] == lv].set_index("attack").reindex(C.ATTACKS)
        ax.bar(np.arange(len(C.ATTACKS)) + (j - 1) * w, dd["recall"], width=w - 0.02, color=shades[lv], label=lv)
    ax.set_xticks(range(len(C.ATTACKS)), [SHORT[a] for a in C.ATTACKS], rotation=30, ha="right")
    ax.set_ylabel(f"Recall ({best})")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Intensity", ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    _save(fig, "fig_intensity.png")


def window():
    ws = _read("e5_window.csv")
    if ws is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    axes[0].plot(ws["window_s"], ws["macro_f1"], marker="o", color=SERIES[0], lw=2, ms=5)
    axes[0].set_title("Macro F1")
    axes[1].plot(ws["window_s"], ws["mean_delay_s"], marker="o", color=SERIES[1], lw=2, ms=5)
    axes[1].set_title("Mean time-to-detect (s)")
    for ax, col in zip(axes, ["macro_f1", "mean_delay_s"]):
        ax.set_xticks(ws["window_s"])
        ax.set_xlabel("Window length (s)")
        for x, v in zip(ws["window_s"], ws[col]):
            ax.annotate(f"{v:.3f}" if col == "macro_f1" else f"{v:.1f}", (x, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6.5)
        pad = (ws[col].max() - ws[col].min()) * 0.4 + 1e-3
        ax.set_ylim(ws[col].min() - pad, ws[col].max() + pad)
        ax.set_xlim(ws["window_s"].min() - 1.5, ws["window_s"].max() + 1.5)  # room for the end labels
    fig.tight_layout()
    _save(fig, "fig_window.png")


def shap_importance():
    fi = _read("feature_importance.csv", index_col=0)
    if fi is None:
        return
    top = fi.sort_values("XGBoost_SHAP", ascending=False).head(15).iloc[::-1]
    gcol = {"network": SERIES[0], "biometric": SERIES[1], "twin": SERIES[2]}
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.barh(top.index, top["XGBoost_SHAP"], color=[gcol[g] for g in top["group"]], height=0.7)
    for y, v in enumerate(top["XGBoost_SHAP"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=6, color=INK2)
    share = fi.groupby("group")["XGBoost_SHAP"].sum()
    handles = [plt.Rectangle((0, 0), 1, 1, color=gcol[g]) for g in ("network", "biometric", "twin")]
    ax.legend(handles, [f"{g} ({share.get(g, 0):.0%} of total)" for g in ("network", "biometric", "twin")],
              loc="lower right")
    ax.set_xlabel("Normalised mean |SHAP| (XGBoost, all classes)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top["XGBoost_SHAP"].max() * 1.2)
    _save(fig, "fig_shap.png")


def realtime(width=COL_W, name="fig_realtime.png"):
    # the saved E6 live run (python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 ...), else the latest run
    tl_p, gt_p = C.RESULTS_DIR / "e6_live_timeline.csv", C.RESULTS_DIR / "e6_live_ground_truth.csv"
    if not tl_p.exists():
        tl_p, gt_p = C.RUNTIME_DIR / "timeline.csv", C.RUNTIME_DIR / "ground_truth.csv"
    if not tl_p.exists():
        return
    tl, gt = pd.read_csv(tl_p), pd.read_csv(gt_p)
    top = tl.groupby("device_id")["risk"].max().sort_values(ascending=False).head(4).index
    fig, ax = plt.subplots(figsize=(width, 2.0 if width < 5 else 2.1))
    for _, c in gt.iterrows():
        ax.axvspan(c["start"] / 60, c["end"] / 60, color="#f0efec", zorder=0)
        ax.text((c["start"] + c["end"]) / 120, 103, SHORT[c["attack"]], ha="center", va="bottom",
                fontsize=5.2 if width < 5 else 6, color=INK2, rotation=90 if width < 5 else 0)
    for i, dev in enumerate(top):
        d = tl[tl["device_id"] == dev]
        ax.plot(d["t"] / 60, d["risk"], lw=1.2, color=SERIES[i], label=dev.replace("UNKNOWN ", "rogue "))
    for thr in (80, 60, 30):
        ax.axhline(thr, color=MUTED, lw=0.5, ls="--")
    ax.set_ylim(0, 135 if width < 5 else 112)
    ax.set_yticks([0, 30, 60, 80, 100])
    ax.set_xlim(0, tl["t"].max() / 60)
    ax.set_xlabel("Minutes of live ward time (grey: ground-truth campaigns)")
    ax.set_ylabel("Device risk")
    ax.legend(ncol=4 if width < 5 else 5, loc="upper left", bbox_to_anchor=(0, -0.3), fontsize=6, handlelength=1.2)
    _save(fig, name)


def realtime_wide():
    """Slide version of the live-run risk figure."""
    realtime(width=7.0, name="fig_realtime_wide.png")


def twin_residual():
    """Monitor vs wearable heart rate: a benign clinical episode versus an in-path data alteration of a monitor."""
    pk = pd.read_parquet(C.DATA_DIR / "test_packets.parquet",
                         columns=["ts", "device_id", "hr", "mqtt", "attack", "campaign", "src_ip"])
    ev = pd.read_csv(C.DATA_DIR / "test_benign_events.csv")
    camps = pd.read_csv(C.DATA_DIR / "test_campaigns.csv")
    tel = pk[(pk["mqtt"] == "PUBLISH") & pk["hr"].notna()]
    clin_all = ev[ev["kind"].str.startswith("clinical_")]
    # a tachycardia episode with no other clinical episode on the same bed at the same time
    clin = None
    for kind in ("tachycardia", "bradycardia", "fever", "desaturation", "st_elevation"):
        for e in clin_all[clin_all["kind"] == f"clinical_{kind}"].itertuples():
            same = clin_all[(clin_all["bed"] == e.bed) & (clin_all.index != e.Index)]
            if not ((same["start"] < e.end + 300) & (same["end"] > e.start - 300)).any():
                clin = e
                break
        if clin is not None:
            break
    # a data-alteration campaign with a patient monitor among its victims, preferring medium intensity
    da = camps[(camps["attack"] == "data_alteration") & camps["target"].str.contains("MON")].copy()
    da["rank"] = da["intensity"].map({"medium": 0, "high": 1, "low": 2})
    alt = da.sort_values("rank").iloc[0]
    mon = [t for t in alt["target"].split(",") if t.startswith("MON")][0]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75))
    panels = [(f"(a) Benign {clin.kind.split('_', 1)[1].replace('_', ' ')}", int(clin.bed), clin.start, clin.end),
              (f"(b) Data alteration of MON-{mon[-2:]} ({alt['intensity']})", int(mon[-2:]), alt["start"], alt["end"])]
    for ax, (title, bed, a, b) in zip(axes, panels):
        lo, hi = a - 240, b + 240
        for dev, col, lab in [(f"MON-{bed:02d}", SERIES[0], "Monitor HR"), (f"WRB-{bed:02d}", SERIES[1], "Wearable HR")]:
            d = tel[(tel["device_id"] == dev) & (tel["ts"].between(lo, hi))]
            d = d.groupby((d["ts"] // 5) * 5)["hr"].mean()
            ax.plot((d.index - a) / 60, d.values, color=col, lw=1.2, label=lab)
        ax.axvspan(0, (b - a) / 60, color="#f0efec", zorder=0)
        ax.set_title(title, fontsize=6.6)
        ax.set_xlabel("Minutes from event start")
    axes[0].set_ylabel("Heart rate (bpm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, "fig_twin_residual.png")


def class_distribution():
    rep = {n: json.loads((C.DATA_DIR / f"{n}_report.json").read_text()) for n in ("train", "test")}
    df = pd.DataFrame({n: r["class_counts"] for n, r in rep.items()}).reindex(C.CLASSES)
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    w = 0.38
    for j, n in enumerate(["train", "test"]):
        ax.bar(np.arange(len(df)) + (j - 0.5) * w, df[n], width=w - 0.03, color=SERIES[j], label=f"{n} run")
    ax.set_yscale("log")
    ax.set_xticks(range(len(df)), [SHORT[c] for c in df.index], rotation=30, ha="right")
    ax.set_ylabel("Flow windows (log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    _save(fig, "fig_class_distribution.png")


def architecture_column():
    """Column-width version of the system diagram (the full-width one is docs/diagrams/architecture.puml)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(COL_W, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 116)
    ax.axis("off")
    stages = [  # (y, height, label, fill)
        (96, 19, "1 Twin", "#e3f2fd"), (74, 18, "2 Data", "#e8f5e9"), (40, 30, "3 Models", "#fff3e0"),
        (17, 19, "4 Live", "#fce4ec"), (1, 12, "5 UI", "#ede7f6")]
    for y, h, label, fill in stages:
        ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fill, ec="#90a4ae", lw=0.6))
        ax.text(4.2, y + h / 2, label, fontsize=6.3, fontweight="bold", color=INK, rotation=90, ha="center", va="center")

    def node(x, y, w, h, label):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec="#455a64", lw=0.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.0, color=INK, linespacing=1.1)
        return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))

    def arrow(p, q, label=None, dx=1.2, dy=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color="#37474f",
                                     shrinkA=0, shrinkB=0))
        if label:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=5.4, color=INK2, va="center")

    phy = node(9, 99.5, 19, 12, "Patient\nphysiology")
    dev = node(32, 99.5, 19, 12, "Medical\ndevices")
    net = node(55, 99.5, 19, 12, "Network\npackets")
    atk = node(79, 99.5, 18, 12, "Attack\ninjector")
    arrow(phy["r"], dev["l"], "1a", dx=-1.6, dy=2.4)
    arrow(dev["r"], net["l"], "1b", dx=-1.6, dy=2.4)
    arrow(atk["l"], net["r"], "1c", dx=-1.6, dy=2.4)
    raw = node(9, 77, 23, 11, "Packet log\n+ labels")
    par = node(39, 77, 23, 11, "Parser and\ncleaner")
    fea = node(69, 77, 28, 11, "Flow features\n(55)")
    arrow(net["b"], raw["t"], "2a", dx=2.0, dy=2.4)
    arrow(raw["r"], par["l"], "2b", dx=-1.6, dy=2.4)
    arrow(par["r"], fea["l"], "2c", dx=-1.6, dy=2.4)
    xs, ws = [9, 30, 51, 72], [18, 18, 18, 25]
    labels = ["Random\nForest", "XGBoost", "MLP", "Isolation Forest\n+ twin rules\n(normal only)"]
    models = [node(x, 54, w, 12, lab) for x, w, lab in zip(xs, ws, labels)]
    for i, m in enumerate(models):
        arrow(fea["b"], m["t"], {1: "3a", 3: "3b"}.get(i), dx=0.8, dy=0.8)
    node(9, 42.5, 54, 8, "Model selection (time-blocked CV)")
    for m in models[:3]:
        arrow(m["b"], (m["b"][0], 50.5))
    det = node(9, 20, 25, 11, "Stream\ndetector")
    risk = node(40, 20, 23, 11, "Risk\nengine")
    alr = node(69, 20, 28, 11, "Incidents and\nexplanations")
    arrow((25, 42.5), (25, 31), "4a", dx=1.0, dy=2.5)
    ax.plot([84.5, 84.5, 30], [54, 37, 37], color="#37474f", lw=0.7, solid_capstyle="butt")
    arrow((30, 37), (30, 31))
    ax.text(85.7, 45, "4b", fontsize=5.4, color=INK2, va="center")
    arrow(det["r"], risk["l"], "4c", dx=-1.6, dy=2.4)
    arrow(risk["r"], alr["l"], "4d", dx=-1.6, dy=2.4)
    node(52, 2.5, 45, 9, "Live ward map, evaluation,\nexperiments, explainability")
    arrow(alr["b"], (alr["b"][0], 11.5), "5", dx=1.2)
    _save(fig, "fig_architecture_column.png")


def main():
    for fn in (architecture_column, class_distribution, model_comparison, confusion, per_class, ablation, zero_day, intensity, window,
               shap_importance, realtime, realtime_wide, twin_residual):
        try:
            fn()
        except Exception as exc:  # keep going so one missing result does not block the rest
            print(f"  skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
