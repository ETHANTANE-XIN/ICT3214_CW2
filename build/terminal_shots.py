"""Render captured program output as terminal-window images for the report appendix.

Each capture is real console output saved by running the command shown in its title bar.
Long captures are trimmed with an explicit "lines omitted" marker; nothing else is edited.
"""
import html
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERM = HERE / "terminal"
FIG = HERE.parent / "MedTwinIDS" / "results" / "figures"
ANSI = {"91": "#f47067", "92": "#57ab5a", "93": "#daaa3f", "96": "#39c5cf"}


def to_html(text: str) -> str:
    out, colour = [], None
    for part in re.split(r"(\x1b\[\d+m)", text):
        m = re.fullmatch(r"\x1b\[(\d+)m", part)
        if m:
            colour = ANSI.get(m.group(1))
            continue
        esc = html.escape(part)
        out.append(f'<span style="color:{colour}">{esc}</span>' if colour else esc)
    return "".join(out)


def trim(lines, head, tail=0):
    if len(lines) <= head + tail:
        return lines
    marker = f"\x1b[92m... {len(lines) - head - tail} lines omitted ...\x1b[0m"
    return lines[:head] + [marker] + (lines[-tail:] if tail else [])


def page(command: str, body: str, width: int) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:#ffffff; }}
.term {{ width:{width}px; background:#22272e; border-radius:8px; overflow:hidden; font-family:'DejaVu Sans Mono',monospace; }}
.bar {{ background:#2d333b; color:#adbac7; padding:6px 12px; font-size:12px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:5px; margin-right:6px; }}
pre {{ margin:0; padding:10px 14px; color:#cdd9e5; font-size:12.5px; line-height:1.35; white-space:pre-wrap; word-break:break-all; }}
.cmd {{ color:#57ab5a; }}
</style></head><body><div class="term"><div class="bar"><span class="dot" style="background:#f47067"></span><span class="dot"
style="background:#daaa3f"></span><span class="dot" style="background:#57ab5a"></span>&nbsp;MedTwinIDS</div>
<pre><span class="cmd">$ {html.escape(command)}</span>
{body}</pre></div></body></html>"""


def build():
    pipeline = (HERE / "terminal" / "pipeline_log.txt").read_text().splitlines()
    i_train = next(i for i, l in enumerate(pipeline) if l.startswith("train flows"))
    i_sel = next(i for i, l in enumerate(pipeline) if l.startswith("Selected model"))
    shots = {
        "term_simulate": ("python -m medtwin.simulate --preset all", pipeline[:i_train], 900),
        "term_train": ("python -m medtwin.train --tune",
                       trim(pipeline[i_train:i_sel + 1], 4, 25), 900),
        "term_realtime": ("python -m medtwin.realtime --minutes 60 --stop-at 16 --speed 0 --seed 99 --start-hour 14",
                          trim((TERM / "realtime_console.txt").read_text().splitlines(), 9, 15), 1050),
        "term_pytest": ("python -m pytest -v", trim((TERM / "pytest.txt").read_text().splitlines(), 2, 26), 1000),
    }
    for name, (cmd, lines, width) in shots.items():
        (TERM / f"{name}.html").write_text(page(cmd, to_html("\n".join(lines)), width))
    subprocess.run(["node", str(HERE / "terminal_shots.js"), str(TERM), str(FIG), *shots], check=True)


if __name__ == "__main__":
    build()
