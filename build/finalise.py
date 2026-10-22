"""Give a saved .docx or .pptx its own metadata and British English proofing.

The IEEE template and the generator libraries leave their own title, author and dates in
docProps/core.xml (e.g. "Paper Title", "IEEE", "python-docx") and mark text as US English, so Word
and PowerPoint would flag British spellings. This rewrites both in place.

Usage: python finalise.py <file> --title "..." --subject "..."
"""
import argparse
import datetime
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

AUTHOR = "ICT3214 Coursework 2 Group"
KEYWORDS = "medical IoT; digital twin; intrusion detection; machine learning; real-time detection"


def core_xml(title: str, subject: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{escape(title)}</dc:title><dc:subject>{escape(subject)}</dc:subject>"
            f"<dc:creator>{AUTHOR}</dc:creator><cp:keywords>{KEYWORDS}</cp:keywords>"
            f"<cp:lastModifiedBy>{AUTHOR}</cp:lastModifiedBy><cp:revision>1</cp:revision>"
            "<dc:language>en-GB</dc:language>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
            "</cp:coreProperties>")


def british(name: str, data: bytes) -> bytes:
    """Switch the proofing language of styles (Word) or text runs (PowerPoint) to en-GB."""
    if name in ("word/styles.xml", "word/document.xml"):
        return re.sub(rb'(<w:lang\b[^>]*\bw:val=")en-US"', rb'\1en-GB"', data)
    if name.startswith("ppt/") and name.endswith(".xml"):
        return data.replace(b'lang="en-US"', b'lang="en-GB"')
    return data


def finalise(path, title: str, subject: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():  # same order and compression as the original package
            data = zin.read(item.filename)
            data = core_xml(title, subject).encode("utf-8") if item.filename == "docProps/core.xml" \
                else british(item.filename, data)
            zout.writestr(item, data)
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subject", required=True)
    a = ap.parse_args()
    finalise(a.file, a.title, a.subject)
    print("finalised", a.file)
