"""Build deliverables/ICT3214_CW2_Submission.zip: report, slides, user manual, source code, dataset and results.

Usage: python package.py
"""
import zipfile
from pathlib import Path

from common import OUT, PROJ, TITLE

ZIP = OUT / "ICT3214_CW2_Submission.zip"
DOCS = ["ICT3214_CW2_Final_Report.doc", "ICT3214_CW2_Final_Report.docx", "ICT3214_CW2_Presentation.pptx",
        "ICT3214_CW2_User_Manual.docx"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".tmp", ".part"}

README = f"""{TITLE}
ICT3214 Security Analytics, Coursework 2

Contents of this archive
  ICT3214_CW2_Final_Report.doc    final report (IEEE template, Word .doc as required)
  ICT3214_CW2_Final_Report.docx   the same report in .docx format
  ICT3214_CW2_Presentation.pptx   presentation slides (speaker notes include the demo steps)
  ICT3214_CW2_User_Manual.docx    environment setup, data generation, analysis and reproduction
  MedTwinIDS/                     source code, generated dataset, trained models, results and figures
    medtwin/                      Python package (digital twin, parser, models, experiments, real-time detector)
    dashboard/app.py              Streamlit dashboard
    tests/                        automated tests (python -m pytest -v)
    data/                         generated dataset (packet logs, flow features, ground truth)
    models/                       trained models and runtime assets
    results/                      metrics tables and figures used in the report

Quick start (from the MedTwinIDS folder): pip install -r requirements.txt, then streamlit run dashboard/app.py.
The user manual explains how to regenerate every dataset, table and figure.
"""


def files():
    for path in sorted(PROJ.rglob("*")):
        rel = path.relative_to(PROJ.parent)
        if path.is_file() and not (SKIP_DIRS & set(rel.parts)) and path.suffix not in SKIP_SUFFIXES:
            yield path, rel


def main():
    missing = [d for d in DOCS if not (OUT / d).exists()]
    if missing:
        raise SystemExit(f"build these first: {missing}")
    n = 0
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", README)
        for d in DOCS:
            z.write(OUT / d, d)
            n += 1
        for path, rel in files():
            z.write(path, Path(*rel.parts).as_posix())
            n += 1
    print(f"wrote {ZIP} with {n + 1} files, {ZIP.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
