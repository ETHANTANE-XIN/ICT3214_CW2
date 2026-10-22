# MedTwin-IDS

AI-driven intrusion detection and device risk scoring for a **medical IoT digital twin** of a hospital ward
(ICT3214 Security Analytics, Coursework 2, Idea 1).

The project simulates an 8-bed ward (patient monitors, infusion pumps, wearables, IoMT gateway, nurse station,
EHR server), injects six labelled attack families at three intensities, parses the packet log into flow
features (network + biometric + digital-twin context), compares **Random Forest, XGBoost and an MLP**, adds an
**Isolation Forest** safety net and label-free **twin consistency rules** for unseen attacks, and runs the selected
model **in real time** with per-device risk scores, grouped incidents and a Streamlit dashboard.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m medtwin.simulate --preset all     # 1. generate train (6 h) and test (3 h) datasets   (~2 min)
python -m medtwin.train --tune              # 2. tune, cross-validate, train, test, select    (~12 min)
python -m medtwin.experiments               # 3. robustness, ablation, zero-day + twin rules, intensity, window, SHAP (~30-40 min)
python -m medtwin.realtime --replay test --minutes 180 --speed 0 --quiet --save-as replay           # 4a. E6 replay
python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --quiet --save-as live     # 4b. E6 live twin
python -m medtwin.realtime --minutes 60 --speed 0 --seed 99 --start-hour 14 --twin-rules --quiet --save-as live_rules  # 4c.
python -m medtwin.figures                   # 5. report figures -> results/figures/
streamlit run dashboard/app.py              # 6. dashboard (then, in a 2nd terminal, step 7)
python -m medtwin.realtime --minutes 20 --speed 10          # 7. live demo feeding the dashboard
python -m pytest -v                         # 8. automated tests (~1.5 min)
```

Full instructions are in the user manual submitted with the code (`ICT3214_CW2_User_Manual.docx`).

## Layout

| Path | Purpose |
|---|---|
| `medtwin/config.py` | ward layout, attack catalogue, feature groups, playbook |
| `medtwin/twin.py` | digital twin: patient physiology, devices, network events |
| `medtwin/attacks.py` | labelled attack campaign scheduler and generators |
| `medtwin/simulate.py` | dataset generation CLI |
| `medtwin/parser.py` | cleaning and flow-window feature extraction (offline and streaming) |
| `medtwin/models.py` | RF, XGBoost, MLP, Isolation Forest factory and imbalance handling |
| `medtwin/train.py` | tuning, time-blocked CV, independent test, model selection (E1) |
| `medtwin/experiments.py` | E1b robustness, E2 ablation, E3 zero-day, E4 intensity, E5 window size, SHAP |
| `medtwin/rules.py` | label-free twin consistency rules R1 to R5 (thresholds from normal training traffic) |
| `medtwin/risk.py` | risk engine, incident grouping, explanations |
| `medtwin/realtime.py` | real-time detector (live twin or replay) |
| `medtwin/figures.py` | static report figures |
| `medtwin/benchmark_wustl.py` | optional benchmark on the public WUSTL-EHMS-2020 dataset |
| `dashboard/app.py` | Streamlit analyst dashboard |
| `tests/test_medtwin.py` | automated tests (pytest) |
| `data/`, `models/`, `results/`, `runtime/` | generated outputs |

All simulated traffic uses private (RFC 1918) and documentation (RFC 5737) address ranges; nothing touches a
real network.
