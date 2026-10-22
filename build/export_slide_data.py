"""Collect the numbers the slide deck needs into build/slide_data.json."""
import json

import pandas as pd
from PIL import Image

from common import PROJ, RES, comparison, report

out = {}
tr, te = report("train"), report("test")
out["train"], out["test"] = tr, te
comp = comparison()
out["best"] = comp.index[comp["selected"]][0]
params = json.loads((RES / "best_params.json").read_text())
out["mlp_layers"] = "-".join(str(u) for u in params["MLP"]["hidden_layer_sizes"])
out["models"] = {m: {k: float(comp.loc[m, k]) for k in ("accuracy", "precision", "recall", "f1", "fpr", "macro_f1",
                                                        "cv_macro_f1", "train_s", "latency_ms", "infer_us_per_flow",
                                                        "size_kb", "roc_auc")} for m in comp.index}
pc = {}
for m in comp.index:
    d = pd.read_csv(RES / f"per_class_{m.lower().replace(' ', '_')}.csv").set_index("cls")
    pc[m] = {k: float(v) for k, v in d["f1"].items()}
out["per_class_f1"] = pc
ab = pd.read_csv(RES / "e2_ablation.csv")
out["ablation"] = ab.to_dict("records")
zd = pd.read_csv(RES / "e3_zero_day.csv")
out["zero_day_mean"] = zd.groupby("detector")[["recall", "fpr"]].mean().round(4).to_dict("index")
ws = pd.read_csv(RES / "e5_window.csv")
out["window"] = ws.to_dict("records")
fi = pd.read_csv(RES / "feature_importance.csv", index_col=0)
out["shap_group"] = fi.groupby("group")["XGBoost_SHAP"].sum().round(4).to_dict()
out["shap_top"] = fi["XGBoost_SHAP"].head(8).round(4).to_dict()
# saved E6 results (runtime/ is overwritten by every live run, e.g. the dashboard demo)
out["live"] = json.loads((RES / "e6_live_live_summary.json").read_text())["overall"]
out["replay"] = json.loads((RES / "e6_replay_live_summary.json").read_text())["overall"]
out["live_rules"] = json.loads((RES / "e6_live_rules_live_summary.json").read_text())["overall"]
boot = pd.read_csv(RES / "e1_bootstrap.csv").set_index("model")
out["boot"] = {m: {"low": float(boot.loc[m, "macro_f1_ci_low"]), "high": float(boot.loc[m, "macro_f1_ci_high"])}
               for m in boot.index}
out["extra_mean"] = pd.read_csv(RES / "e1_extra_runs.csv").groupby("model")["macro_f1"].mean().round(4).to_dict()
# pixel sizes of every image placed on a slide, so the deck keeps true aspect ratios
imgs = {n: RES / "figures" / n for n in ("fig_confusion.png", "fig_zero_day.png", "fig_intensity.png", "fig_window.png",
                                         "fig_realtime_wide.png", "dash_live_ward_monitor.png", "term_pytest.png")}
imgs["architecture.png"] = PROJ / "docs" / "diagrams" / "architecture.png"
out["sizes"] = {}
for name, path in imgs.items():
    w, h = Image.open(path).size
    out["sizes"][name] = {"w": w, "h": h}
(PROJ.parent / "build" / "slide_data.json").write_text(json.dumps(out, indent=2))
print("slide data exported")
