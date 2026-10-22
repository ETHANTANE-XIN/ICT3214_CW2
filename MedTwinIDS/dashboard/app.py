"""MedTwin-IDS analyst dashboard.

Run with:  streamlit run dashboard/app.py
The Live Ward Monitor page reads runtime/ files written by `python -m medtwin.realtime`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medtwin import config as C  # noqa: E402

st.set_page_config(page_title="MedTwin-IDS", page_icon="🩺", layout="wide")

# reserved status palette (good / warning / serious / critical); the level is always printed as text too
LEVEL_COLOUR = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
LEVEL_TEXT = {"Low": "#0b0b0b", "Medium": "#0b0b0b", "High": "#0b0b0b", "Critical": "#ffffff"}
CLASS_COLOUR = {"normal": "#90a4ae", "dos_flood": "#e53935", "spoofing": "#8e24aa", "data_alteration": "#d81b60",
                "unauthorised_access": "#fb8c00", "reconnaissance": "#1e88e5", "exfiltration": "#00897b",
                "anomaly_unknown": "#6d4c41"}
MODEL_COLOUR = {"Random Forest": "#2a78d6", "XGBoost": "#eb6834", "MLP": "#1baf7a"}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.tile {border-radius: 8px; padding: 6px 8px; margin: 2px 0; color: #fff; font-size: 0.80rem; line-height: 1.25;}
.bed {border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 8px; margin-bottom: 8px;}
.bed h4 {margin: 0 0 4px 0; font-size: 0.95rem;}
.vit {font-size: 0.78rem; opacity: 0.85;}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def read_csv(path: str, **kw):
    p = ROOT / path
    return pd.read_csv(p, **kw) if p.exists() else None


def read_state():
    p = C.RUNTIME_DIR / "state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def tile(dev):
    col = LEVEL_COLOUR[dev["level"]]
    cls = "" if dev["last_class"] == "normal" else f"<br><b>{dev['last_class']}</b>"
    return (f"<div class='tile' style='background:{col};color:{LEVEL_TEXT[dev['level']]}'>{dev['device_id']} &middot; risk {dev['risk']:.0f}"
            f" ({dev['level']}){cls}</div>")


# --------------------------------------------------------------------------
# Live page
# --------------------------------------------------------------------------
def live_page():
    st.title("Live Ward Monitor")
    st.caption("Digital twin of an 8-bed ward: every device's risk is updated each detection window. "
               "Start the stream with `python -m medtwin.realtime --minutes 20 --speed 10`.")

    @st.fragment(run_every="2s")
    def body():
        s = read_state()
        if s is None:
            st.info("No live run found yet. Start `python -m medtwin.realtime` in another terminal.")
            return
        k = s["kpis"]
        progress = min(1.0, s["sim_time"] / max(1.0, s["duration"]))
        st.progress(progress, text=f"Ward clock {s['clock']} | model: {s['model']} | window {s['window']:.0f} s | speed x{s['speed']}")
        c = st.columns(6)
        c[0].metric("Flows analysed", f"{k['flows_processed']:,}")
        c[1].metric("Alerts", f"{k['alerts_total']:,}")
        c[2].metric("Incidents", k["incidents"])
        c[3].metric("Highest device risk", f"{k['max_risk']:.0f}")
        c[4].metric("Devices High/Critical", k["devices_at_risk"])
        c[5].metric("Window latency", f"{k['avg_latency_ms']:.0f} ms", help=f"95th percentile {k['p95_latency_ms']:.0f} ms")

        devs = {d["device_id"]: d for d in s["devices"]}
        st.subheader("Ward map")
        cols = st.columns(4)
        for b in range(1, C.N_BEDS + 1):
            v = s["vitals"].get(str(b), {})
            vit = (f"HR {v.get('hr', '-')} bpm &middot; SpO2 {v.get('spo2', '-')}% &middot; "
                   f"BP {v.get('sys', '-')}/{v.get('dia', '-')} &middot; pump {v.get('pump_rate', '-')} ml/h")
            html = f"<div class='bed'><h4>Bed {b}</h4><div class='vit'>{vit}</div>"
            for prefix in ("MON", "PMP", "WRB"):
                html += tile(devs[f"{prefix}-{b:02d}"])
            cols[(b - 1) % 4].markdown(html + "</div>", unsafe_allow_html=True)
        infra = [d for d in s["devices"] if d["bed"] == 0]
        st.markdown("<div class='bed'><h4>Infrastructure</h4>" + "".join(
            f"<span style='display:inline-block;width:16%;margin-right:0.5%'>{tile(d)}</span>" for d in infra) + "</div>",
            unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Incidents (grouped alerts)")
            inc = pd.DataFrame(s["incidents"])
            if len(inc):
                inc = inc.sort_values("last_t", ascending=False)
                inc["why"] = inc["explanation"].apply(lambda e: ", ".join(f"{x['feature']}={x['value']}" for x in e))
                show = inc[["first_clock", "last_clock", "device", "attack", "alerts", "max_conf", "why", "action"]]
                st.dataframe(show, hide_index=True, width="stretch", height=320)
            else:
                st.success("No incidents so far.")
        with right:
            st.subheader("Risk timeline")
            tl = C.RUNTIME_DIR / "timeline.csv"
            if tl.exists():
                t = pd.read_csv(tl)
                top = t.groupby("device_id")["risk"].max().sort_values(ascending=False).head(6).index
                fig = px.line(t[t["device_id"].isin(top)], x="clock", y="risk", color="device_id", height=320)
                fig.add_hrect(y0=80, y1=100, fillcolor="#c62828", opacity=0.08, line_width=0)
                fig.add_hrect(y0=60, y1=80, fillcolor="#ef6c00", opacity=0.06, line_width=0)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_range=[0, 100])
                st.plotly_chart(fig, width="stretch")
        with st.expander("Ground truth of the simulation (evaluation only, hidden from the detector)"):
            gt = pd.DataFrame(s["ground_truth"])
            st.dataframe(gt if len(gt) else pd.DataFrame({"status": ["no attack campaign active"]}), hide_index=True)

    body()


# --------------------------------------------------------------------------
# Model evaluation page
# --------------------------------------------------------------------------
def evaluation_page():
    st.title("Model Evaluation (independent test run)")
    comp = read_csv("results/model_comparison.csv")
    if comp is None:
        st.warning("Run `python -m medtwin.train` first.")
        return
    sel = comp.loc[comp["selected"], "model"].iloc[0]
    st.success(f"Deployed model: **{sel}** (chosen on time-blocked cross-validation macro-F1, then tested once).")
    cols = ["model", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "macro_f1", "cv_macro_f1",
            "train_s", "infer_us_per_flow", "latency_ms", "size_kb"]
    st.dataframe(comp[cols].round(4), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        m = comp.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "macro_f1"])
        fig = px.bar(m, x="variable", y="value", color="model", barmode="group", color_discrete_map=MODEL_COLOUR,
                     title="Detection quality", range_y=[min(0.8, m["value"].min() - 0.02), 1.0])
        st.plotly_chart(fig, width="stretch")
    with c2:
        m = comp.melt(id_vars="model", value_vars=["train_s", "latency_ms"])
        m["variable"] = m["variable"].map({"train_s": "Training time (s)", "latency_ms": "Latency per flow (ms)"})
        fig = px.bar(m, x="model", y="value", color="model", facet_col="variable", color_discrete_map=MODEL_COLOUR,
                     text_auto=".2f", facet_col_spacing=0.12, title="Cost: training time and single-flow latency")
        # the two panels have different units, so each gets its own y-axis with visible tick labels
        fig.update_yaxes(matches=None, showticklabels=True, title_text="")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("Confusion matrix for", list(comp["model"]), index=list(comp["model"]).index(sel))
        cm = read_csv(f"results/confusion_{name.lower().replace(' ', '_')}.csv", index_col=0)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        fig = px.imshow(norm, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicted", y="True", color="Row share"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        pr = read_csv("results/test_predictions.csv")
        fig = go.Figure()
        for mname in comp["model"]:
            fpr, tpr, _ = roc_curve(pr["label"], pr[f"p_attack_{mname}"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{mname} (AUC {auc(fpr, tpr):.4f})",
                                     line=dict(color=MODEL_COLOUR[mname])))
        fig.update_layout(title="ROC (attack vs normal), log FPR axis", xaxis_type="log", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Per-class F1")
    rows = []
    for mname in comp["model"]:
        pc = read_csv(f"results/per_class_{mname.lower().replace(' ', '_')}.csv")
        pc["model"] = mname
        rows.append(pc)
    pc = pd.concat(rows)
    fig = px.bar(pc, x="cls", y="f1", color="model", barmode="group", color_discrete_map=MODEL_COLOUR)
    st.plotly_chart(fig, width="stretch")
    cv = read_csv("results/cv_folds.csv")
    if cv is not None:
        st.subheader("Five-fold time-blocked cross-validation")
        fig = px.box(cv, x="model", y="macro_f1", color="model", points="all", color_discrete_map=MODEL_COLOUR)
        st.plotly_chart(fig, width="stretch")
    ex = read_csv("results/e1_extra_runs.csv")
    if ex is not None:
        st.subheader("Robustness: the saved models on three extra unseen wards")
        ex = ex.assign(run=ex["seed"].astype(str) + " (" + ex["start_hour"].map(lambda h: f"{h:02.0f}:00") + ")")
        # a dot plot, so the axis can start near the scores without exaggerating bar lengths
        fig = px.scatter(ex, x="run", y="macro_f1", color="model", color_discrete_map=MODEL_COLOUR, text="macro_f1",
                         labels={"run": "run (seed, start)", "macro_f1": "macro F1"})
        fig.update_traces(marker_size=14, texttemplate="%{text:.3f}", textposition="middle right")
        fig.update_yaxes(range=[0.9, 1.0])
        st.plotly_chart(fig, width="stretch")
    bt = read_csv("results/e1_bootstrap.csv")
    if bt is not None:
        st.caption("95% bootstrap intervals of macro F1 on the test run, and of the difference to the selected model")
        num = bt.columns.drop("model")
        bt[num] = bt[num].map(lambda v: f"{v:.4f}" if pd.notna(v) else "selected")  # text column: Arrow-safe
        st.dataframe(bt, hide_index=True)


# --------------------------------------------------------------------------
def experiments_page():
    st.title("Experiments")
    ab = read_csv("results/e2_ablation.csv")
    if ab is not None:
        st.subheader("E2 Feature-set ablation")
        fig = px.bar(ab, x="feature_set", y="macro_f1", color="model", barmode="group", color_discrete_map=MODEL_COLOUR,
                     category_orders={"feature_set": list(C.FEATURE_SETS)})
        st.plotly_chart(fig, width="stretch")
        rc = ab.melt(id_vars=["feature_set", "model"], value_vars=["recall_spoofing", "recall_data_alteration"])
        fig = px.bar(rc, x="feature_set", y="value", color="model", facet_col="variable", barmode="group",
                     color_discrete_map=MODEL_COLOUR, category_orders={"feature_set": list(C.FEATURE_SETS)},
                     title="Recall on the two integrity attacks")
        st.plotly_chart(fig, width="stretch")
    zd = read_csv("results/e3_zero_day.csv")
    if zd is not None:
        st.subheader("E3 Zero-day: recall on an attack family never seen in training")
        comp = read_csv("results/model_comparison.csv")
        best = comp.loc[comp["selected"], "model"].iloc[0] if comp is not None else "XGBoost"
        order = ["Random Forest", "XGBoost", "MLP", "Isolation Forest", "Twin rules", f"Hybrid ({best} + IF)",
                 f"Hybrid ({best} + twin rules)", f"Hybrid ({best} + IF + twin rules)"]
        order = [o for o in order if o in set(zd["detector"])] + sorted(set(zd["detector"]) - set(order))
        piv = zd.pivot(index="detector", columns="held_out", values="recall").loc[order, list(C.ATTACKS)]
        st.plotly_chart(px.imshow(piv, text_auto=".2f", color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=1),
                        width="stretch")
        st.dataframe(zd.groupby("detector")[["recall", "fpr"]].mean().round(4).sort_values("recall", ascending=False))
        sens = read_csv("results/e3_rules_sensitivity.csv")
        if sens is not None:
            st.caption("Twin rules: recall per unseen family and false positive rate for three threshold quantiles "
                       "(the default is 0.999; the others are shown for operators, not used for selection)")
            st.dataframe(sens.round(4), hide_index=True)
    it = read_csv("results/e4_intensity.csv")
    if it is not None:
        st.subheader("E4 Recall by attack intensity")
        fig = px.bar(it, x="attack", y="recall", color="intensity", facet_col="model", barmode="group",
                     category_orders={"intensity": ["low", "medium", "high"]})
        st.plotly_chart(fig, width="stretch")
    ws = read_csv("results/e5_window.csv")
    if ws is not None:
        st.subheader("E5 Window size: accuracy versus time-to-detect")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(ws, x="window_s", y=["macro_f1", "f1"], markers=True), width="stretch")
        c2.plotly_chart(px.line(ws, x="window_s", y=["mean_delay_s", "median_delay_s"], markers=True), width="stretch")
    rows = {}
    for name, label in (("e6_replay_live_summary.json", "Replay of test run"), ("e6_live_live_summary.json", "Live twin (seed 99)"),
                        ("e6_live_rules_live_summary.json", "Live twin + twin rules")):
        p = C.RESULTS_DIR / name
        if p.exists():
            rows[label] = json.loads(p.read_text())["overall"]
    if rows:
        st.subheader("E6 Real-time detection")
        st.dataframe(pd.DataFrame(rows).T[["campaigns", "detected", "median_delay_s", "mean_delay_s", "alerts",
                                           "incidents", "false_alerts", "avg_window_latency_ms", "p95_window_latency_ms"]])


# --------------------------------------------------------------------------
def data_page():
    st.title("Dataset Explorer")
    rep = {n: json.loads((C.DATA_DIR / f"{n}_report.json").read_text()) for n in ("train", "test")
           if (C.DATA_DIR / f"{n}_report.json").exists()}
    if rep:
        st.dataframe(pd.DataFrame(rep).T[["seed", "hours", "packets", "attack_packets", "flows", "attack_flows",
                                          "campaigns", "dropped_exact_duplicates", "vital_glitches_nulled"]])
        cc = pd.DataFrame({n: r["class_counts"] for n, r in rep.items()}).fillna(0).reset_index().melt(id_vars="index")
        fig = px.bar(cc, x="index", y="value", color="variable", barmode="group", log_y=True,
                     labels={"index": "class", "value": "flows (log scale)", "variable": "run"})
        st.plotly_chart(fig, width="stretch")
    camps = read_csv("data/test_campaigns.csv")
    if camps is not None:
        st.subheader("Attack campaigns in the test run")
        camps = camps.assign(start_min=camps["start"] / 60, dur_min=camps["duration"] / 60)
        fig = px.bar(camps, x="dur_min", y="attack", base="start_min", color="intensity", orientation="h",
                     hover_data=["source_ip", "target", "variant"], labels={"dur_min": "minutes from start"})
        st.plotly_chart(fig, width="stretch")
    flows = read_csv("data/test_flows.csv")
    if flows is not None:
        st.subheader("Feature distribution by class")
        feat = st.selectbox("Feature", C.ALL_FEATURES, index=C.ALL_FEATURES.index("TwinResidualHR"))
        sample = pd.concat([g.sample(min(len(g), 1500), random_state=0) for _, g in flows.groupby("attack")])
        fig = px.box(sample, x="attack", y=feat, color="attack", color_discrete_map=CLASS_COLOUR, points=False)
        st.plotly_chart(fig, width="stretch")


def xai_page():
    st.title("Explainability")
    fi = read_csv("results/feature_importance.csv", index_col=0)
    if fi is None:
        st.warning("Run `python -m medtwin.experiments` first.")
        return
    top = fi.head(20).reset_index().rename(columns={"index": "feature"})
    fig = px.bar(top.iloc[::-1], x="XGBoost_SHAP", y="feature", color="group", orientation="h", height=600,
                 title="Top 20 features by mean |SHAP| (XGBoost)")
    st.plotly_chart(fig, width="stretch")
    grp = fi.groupby("group")[["XGBoost_SHAP", "RandomForest_Gini"]].sum().reset_index().melt(id_vars="group")
    st.plotly_chart(px.bar(grp, x="variable", y="value", color="group", title="Share of importance by feature group"),
                    width="stretch")
    pc = read_csv("results/shap_per_class.csv", index_col=0)
    if pc is not None:
        st.subheader("Mean |SHAP| per class (top 15 features)")
        top15 = pc.sum(axis=1).sort_values(ascending=False).head(15).index
        st.plotly_chart(px.imshow(pc.loc[top15], aspect="auto", color_continuous_scale="Viridis"), width="stretch")


PAGES = {"Live Ward Monitor": live_page, "Model Evaluation": evaluation_page, "Experiments": experiments_page,
         "Dataset Explorer": data_page, "Explainability": xai_page}
with st.sidebar:
    st.header("🩺 MedTwin-IDS")
    st.caption("AI-driven intrusion detection for a medical IoT digital twin")
    page = st.radio("View", list(PAGES))
    st.markdown("---")
    st.caption("Pipeline: digital twin → packet log → parser → flow features → RF / XGBoost / MLP + Isolation Forest → risk engine → dashboard")
PAGES[page]()
