"""Compose IEEE-conference-template Word documents with python-docx.

The provided template is converted once to transitional OOXML (it ships as Strict OOXML,
which python-docx cannot open). All styles, numbering (I./A., Fig. N, TABLE I, [n]) and
section layouts come from the template itself; this module only arranges content.
"""
from __future__ import annotations

import copy
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|\^[^^]+\^|~[^~]+~)")


class IEEEDoc:
    def __init__(self, template: str, footer_text: str = ""):
        self.doc = Document(template)
        # resolve styles by their exact names (python-docx's built-in name mapping misses this template's headings)
        self.styles = {st.name: st for st in self.doc.styles}
        body = self.doc.element.body
        kids = list(body)
        sect_paras = [el for el in kids if el.tag == qn("w:p") and el.find(".//" + qn("w:sectPr")) is not None]
        # template order: title section, author section (3 cols), spacer (3 cols), body (2 cols)
        self.sp_title = copy.deepcopy(sect_paras[0])
        self.sp_authors = copy.deepcopy(sect_paras[1])
        self.sp_body2 = copy.deepcopy(sect_paras[-1])
        self.final = copy.deepcopy(kids[-1])  # body-level sectPr (single column)
        self.sp_one = self._one_col_break()
        for el in kids:
            body.remove(el)
        body.append(self.final)
        self.body = body
        self.fig_no = 0
        self.tab_no = 0
        self.mode = "two"
        self.footer_text = footer_text

    # ------------------------------------------------------------------ helpers
    def _one_col_break(self):
        p = copy.deepcopy(self.sp_body2)
        cols = p.find(".//" + qn("w:cols"))
        for a in list(cols.attrib):
            del cols.attrib[a]
        cols.set(qn("w:space"), "720")
        return p

    def _append(self, el):
        self.final.addprevious(el)
        return el

    def _para(self, style, text="", align=None, keep_next=False, space_after=None):
        p = self.doc.add_paragraph(style=self.styles[style])
        if text:
            self._runs(p, text)
        if align is not None:
            p.alignment = align
        if keep_next:
            p.paragraph_format.keep_with_next = True
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    def _runs(self, p, text, size=None):
        for part in INLINE.split(text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("`") and part.endswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Courier New"
                r._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
                if size is None:
                    r.font.size = Pt(9)
            elif part.startswith("^") and part.endswith("^"):
                r = p.add_run(part[1:-1]); r.font.superscript = True
            elif part.startswith("~") and part.endswith("~"):
                r = p.add_run(part[1:-1]); r.font.subscript = True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                r = p.add_run(part[1:-1]); r.italic = True
            else:
                r = p.add_run(part)
            if size is not None:
                r.font.size = Pt(size)
        return p

    def _section_break(self, template_para):
        self._append(copy.deepcopy(template_para))

    # ------------------------------------------------------------------ front matter
    def title(self, text, subtitle_lines=()):
        self._para("paper title", text)
        for line in subtitle_lines:
            p = self._para("Author", line, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
            for r in p.runs:
                r.font.size = Pt(10)
        self._section_break(self.sp_title)

    def authors(self, members, affiliation="Singapore Institute of Technology", per_col=2):
        n = len(members)
        cols = 3
        # fill column-major so the visual order reads left to right, row by row
        rows = (n + cols - 1) // cols
        order = []
        for c in range(cols):
            for r in range(rows):
                k = r * cols + c
                if k < n:
                    order.append(k)
        col_of = {k: i % cols for i, k in enumerate([r * cols + c for r in range(rows) for c in range(cols)])}
        prev_col = None
        for idx in order:
            m = members[idx]
            c = col_of[idx]
            p = self.doc.add_paragraph(style=self.styles["Author"])
            if prev_col is not None and c != prev_col:
                p.add_run().add_break(WD_BREAK.COLUMN)
            prev_col = c
            lines = [(m["name"], False), (f"Student ID: {m['sid']}", False), (affiliation, True), ("Singapore", False),
                     (m["email"], False)]
            for i, (line, ital) in enumerate(lines):
                r = p.add_run(line)
                r.italic = ital
                r.font.size = Pt(9)
                if i < len(lines) - 1:
                    r.add_break()
            p.paragraph_format.space_after = Pt(6)
        self._section_break(self.sp_authors)

    def abstract(self, text):
        p = self.doc.add_paragraph(style=self.styles["Abstract"])
        r = p.add_run("Abstract"); r.italic = True
        p.add_run("—")  # the template's Abstract style prescribes this separator
        self._runs(p, text)

    def keywords(self, text):
        p = self.doc.add_paragraph(style=self.styles["Keywords"])
        r = p.add_run("Keywords"); r.italic = True
        p.add_run("—" + text)

    # ------------------------------------------------------------------ body
    def h1(self, text):
        return self._para("Heading 1", text, keep_next=True)

    def h2(self, text):
        return self._para("Heading 2", text, keep_next=True)

    def h3(self, text):
        return self._para("Heading 3", text, keep_next=True)

    def h5(self, text):
        return self._para("Heading 5", text, keep_next=True)

    def p(self, text):
        return self._para("Body Text", text)

    def bullets(self, items):
        for it in items:
            self._para("bullet list", it)

    def equation(self, expr, number):
        p = self.doc.add_paragraph(style=self.styles["equation"])
        p.add_run("\t")
        self._runs(p, expr)
        p.add_run(f"\t({number})")
        # the template's equation style uses the Symbol font, which turns Latin letters into Greek ones
        for r in p.runs:
            r.font.name = "Times New Roman"
            rf = r._element.get_or_add_rPr().get_or_add_rFonts()
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), "Times New Roman")
        return p

    def figure(self, path, caption, width_in=3.35, wide=False):
        self.fig_no += 1
        wide = wide and self.mode == "two"
        if wide:
            self._section_break(self.sp_body2)
        p = self.doc.add_paragraph(style=self.styles["Body Text"])
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(4)
        # the template's Body Text uses exact line spacing, which would crop an inline picture
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(str(path), width=Inches(width_in))
        cap = self.doc.add_paragraph(style=self.styles["figure caption"])
        self._runs(cap, caption)
        if wide:
            self._section_break(self.sp_one)
        return self.fig_no

    def table(self, caption, header, rows, widths_in, wide=False, size=7.5, footnote=None, align=None):
        self.tab_no += 1
        wide = wide and self.mode == "two"
        if wide:
            self._section_break(self.sp_body2)
        cap = self.doc.add_paragraph(style=self.styles["table head"])
        self._runs(cap, caption)
        cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        self._borders(t)
        align = align or (["left"] + ["center"] * (len(header) - 1))
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Inches(widths_in[j])
                para = cell.paragraphs[0]
                para.style = self.styles["table col head" if i == 0 else "table copy"]
                para.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                                  "right": WD_ALIGN_PARAGRAPH.RIGHT}[align[j] if i else "center"]
                self._runs(para, str(val), size=size)
                if i == 0:
                    for r in para.runs:
                        r.bold = True
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths_in[j] * 1440)))
        if footnote:
            fn = self.doc.add_paragraph(style=self.styles["Body Text"])
            fn.paragraph_format.first_line_indent = Inches(0)
            self._runs(fn, footnote, size=7)
        else:
            spacer = self.doc.add_paragraph(style=self.styles["Body Text"])
            spacer.paragraph_format.space_after = Pt(2)
        if wide:
            self._section_break(self.sp_one)
        return self.tab_no

    def _borders(self, table):
        tblPr = table._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):  # schema order
            e = OxmlElement(f"w:{edge}")
            if edge in ("top", "bottom", "insideH"):
                e.set(qn("w:val"), "single")
                e.set(qn("w:sz"), "4" if edge != "insideH" else "2")
                e.set(qn("w:space"), "0")
                e.set(qn("w:color"), "000000" if edge != "insideH" else "BFBFBF")
            else:
                e.set(qn("w:val"), "nil")
            b.append(e)
        # narrow cell padding so short numeric columns do not break words
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), "40")
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        # schema order inside tblPr: ... tblBorders, shd, tblLayout, tblCellMar, tblLook ...
        look = tblPr.find(qn("w:tblLook"))
        layout = tblPr.find(qn("w:tblLayout"))
        anchor = layout if layout is not None else look
        if anchor is not None:
            anchor.addprevious(b)
            (look if look is not None else anchor).addprevious(mar) if look is not None else anchor.addnext(mar)
        else:
            tblPr.append(b)
            tblPr.append(mar)

    def references(self, refs, heading="References"):
        self.h5(heading)
        for r in refs:
            self._para("references", r)

    def one_column(self):
        """End the two-column body; everything after this (e.g. an appendix) is single column."""
        self._section_break(self.sp_body2)
        self.mode = "one"

    def page_break(self):
        p = self.doc.add_paragraph()
        p.add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        # close the 2-column body section; the document's final section is single column
        if self.mode == "two":
            self._section_break(self.sp_body2)
        if self.footer_text:
            for s in self.doc.sections:
                for ft in (s.first_page_footer, s.footer):
                    if ft.is_linked_to_previous:
                        continue
                    for para in ft.paragraphs:
                        for r in para.runs:
                            r.text = ""
                    if ft.paragraphs:
                        r = ft.paragraphs[0].add_run(self.footer_text)
                        r.font.size = Pt(8)

        self.doc.save(path)
