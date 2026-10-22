# MedTwin-IDS: ICT3214 Coursework 2

AI-driven intrusion detection and device risk scoring for a **medical IoT digital twin** of an eight-bed
hospital ward (ICT3214 Security Analytics, Coursework 2, Idea 1 with elements of Ideas 7 and 12).

The twin simulates patients, 24 medical devices and their network traffic, injects six labelled attack families
at three intensities, and turns the packets into 55 flow features (network, biometric and twin consistency).
Random Forest, XGBoost and an MLP are compared under a leakage-aware protocol; an Isolation Forest and
label-free twin consistency rules cover attacks never seen in training; and the selected model runs in real time
with per-device risk scores, grouped incidents and a Streamlit dashboard.

## Repository layout

| Path | Contents |
|---|---|
| `deliverables/` | Final report (`.doc` as required, and `.docx`), project proposal, presentation slides, user manual |
| `MedTwinIDS/` | Source code, generated dataset, trained models, results and figures ([details](MedTwinIDS/README.md)) |
| `build/` | Scripts that build the report, proposal, manual, slides and submission zip from the result files |

## Key results (independent test run)

| Question | Result |
|---|---|
| Best model | XGBoost, chosen by time-blocked cross-validation: macro F1 0.986, binary F1 0.9974, 0.83 ms per flow |
| Do twin features help? | Macro F1 0.876 without them, 0.986 with them; spoofing recall 38% to 97% |
| Unseen attack families | Mean recall 0.24 (XGBoost), 0.77 with an Isolation Forest, 0.96 with the twin rules added |
| Real time | 30 of 30 test campaigns detected, median time-to-detect 3.5 s |

## Quick start

```bash
cd MedTwinIDS
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -v                                    # 17 automated tests
streamlit run dashboard/app.py                         # dashboard on the included results
```

The commands that regenerate every dataset, table and figure are in [MedTwinIDS/README.md](MedTwinIDS/README.md)
and, step by step, in the user manual (`deliverables/ICT3214_CW2_User_Manual.docx`).

## Rebuilding the documents (optional)

The documents in `deliverables/` are generated from the result files, so every number in them comes from
`MedTwinIDS/results/`. To rebuild them, install `python-docx`, then from `build/`:

```bash
python build_report.py && python build_proposal.py && python build_manual.py
python export_slide_data.py && python make_insights.py && npm install && node build_slides.js
python package.py                                      # deliverables/ICT3214_CW2_Submission.zip
```

The `.doc` copies are converted from the `.docx` files with LibreOffice
(`soffice --headless --convert-to "doc:MS Word 97" <file>.docx`).

## Safety

All traffic is synthetic and generated inside the program, using private (RFC 1918) and documentation
(RFC 5737) address ranges. No packets are sent to any network, the attack generators only write labelled records
that resemble each threat's network footprint, and no real patient data is used.
