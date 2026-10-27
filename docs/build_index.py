#!/usr/bin/env python3
"""Build the documentation index: docs/index.html.

The documentation is a lot of markdown in a lot of directories, and it is read
in the browser: Chrome's markdown reader renders a `.md` straight from disk, so
the index links to the `.md` files themselves, not to copies of them.

The page is GENERATED, and the generator's default is `--check`: the committed
page is compared byte for byte with what this would write, and a test runs
that check, so a hand-edited index fails `cargo test` and a stale one fails
too. The same discipline as `kingdoms/futhark_shapes/gen_tables.py`, and for
the same reason: the index is a lie the moment it stops matching the tree.

    python3 docs/build_index.py           # write docs/index.html
    python3 docs/build_index.py --check   # compare, and fail if they differ

It indexes two trees: this repository's `docs/`, and the sibling `phylu`
checkout's `docs/` (linked by a relative path, so it needs the two checkouts
to be siblings — the same assumption every other cross-repo reference makes).
The sibling is skipped if it is not there.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "index.html"

# The sibling checkout, relative to this repository's root.
PHYLU = REPO.parent / "phylu" / "docs"

# The groups of the top level, in the order they are read. The rule is the
# file name's prefix; anything unclaimed lands in the last group, so a new
# document is never lost — it is just filed with the rest.
FAMILIES = [
    ("Plans", lambda n: n.startswith("PLAN_")),
    ("Assessments", lambda n: n.startswith("ASSESSMENT_")),
    ("Change requests", lambda n: n.startswith("CR_")),
    ("Specs and designs", lambda n: n.startswith("SPEC_") or n.startswith("DESIGN_")),
    ("Reports", lambda n: n.startswith("REPORT_")),
    ("Reference and results", lambda n: True),
]

# The hand-picked shortlist: where a reader starts. The order is the reading
# order, and each line says why it is there.
START_HERE = [
    ("REPORT_2026-10-27_shape_kingdom.md", "the shape kingdom end to end: what was built, measured, and what is next"),
    ("GOAL.md", "what this repository is for"),
    ("USAGE.md", "how to drive it"),
    ("PLAN_futhark_shapes_product.md", "the shape kingdom's product decision, and what the phylu half turned out to be"),
    ("FirstClassShapes.md", "the design that made a matrix shape a TYPE"),
    ("ASSESSMENT_first_class_shapes.md", "that design, measured, with the build's corrections"),
    ("EGRAPH_futhark.md", "the Futhark kingdom inside the e-graph, and the rules that are bit-identical"),
    ("RESCUE_futhark.md", "the compiler hang: what was measured, and why the e-graph rescue is negative"),
    ("SPLIT_fuller_phylu.md", "the boundary between this repository and phylu — read every line before changing it"),
    ("PLAN_population_dag.md", "the population DAG: the next structural step"),
    ("WGSL_TYPED_GENE.md", "the typed gene, the WGSL kingdom's"),
    ("SPEC_fuller_gpu.md", "the GPU design"),
    ("PLAN_fuller_gpu.md", "the GPU plan, and what was built of it"),
    ("WGPU_EVOLUTION_TARGETS.md", "what the evolution is aiming at"),
    ("FRONTIER_EVOLUTION.md", "the frontier the search is chasing"),
]

HEADING = re.compile(r"^#\s+(.*)$")
FENCE = re.compile(r"^\s*```")


def title_of(text: str, fallback: str) -> str:
    """The first level-one heading, or the file name."""
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            return m.group(1).strip()
    return fallback


def description_of(text: str) -> str:
    """The first prose paragraph after the title.

    Code fences, tables, headings and list bullets are skipped: a bullet is
    often a fragment, and a table row reads as noise in a list. What is left
    is the sentence that says what the document is.
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if HEADING.match(line):
            start = i + 1
            break
    fence = False
    para: list[str] = []
    for line in lines[start:]:
        if FENCE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        stripped = line.strip()
        if not stripped:
            if para:
                break
            continue
        if stripped.startswith(("#", "|", ">", "---", "===", "<!--")):
            continue
        if stripped.startswith(("-", "*", "+")) or re.match(r"^\d+\.\s", stripped):
            continue
        para.append(stripped)
    if not para:
        return ""
    text = " ".join(para)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:217].rsplit(" ", 1)[0] + "…"
    return text


def collect(root: Path, prefix: str) -> list[dict]:
    """Every markdown file under `root`, as a dict with its link and text."""
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.rglob("*.md")):
        if path.name == "index.md":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"build_index: {path}: {e}", file=sys.stderr)
            continue
        rel = path.relative_to(root)
        out.append({
            "link": f"{prefix}{rel.as_posix()}",
            "path": rel.as_posix(),
            "title": title_of(text, path.stem.replace("_", " ")),
            "description": description_of(text),
        })
    return out


def group_of(name: str) -> str:
    for label, rule in FAMILIES:
        if rule(name):
            return label
    return FAMILIES[-1][0]


def item(entry: dict) -> str:
    desc = f'<div class="d">{html.escape(entry["description"])}</div>' if entry["description"] else ""
    path = f'<div class="p">{html.escape(entry["path"])}</div>' if "/" in entry["path"] else ""
    return (
        f'<li class="i" data-t="{html.escape((entry["title"] + " " + entry["path"] + " " + entry["description"]).lower())}">'
        f'<a class="t" href="{html.escape(entry["link"])}">{html.escape(entry["title"])}</a>'
        f"{desc}{path}</li>"
    )


def section(title: str, entries: list[dict], open_: bool = True, note: str = "") -> str:
    if not entries:
        return ""
    head = f"<h2>{html.escape(title)} <span class='n'>{len(entries)}</span></h2>"
    if note:
        head += f"<p class='note'>{html.escape(note)}</p>"
    body = "".join(item(e) for e in entries)
    return f"<details class='g'{' open' if open_ else ''}><summary>{head}</summary><ul class='l'>{body}</ul></details>"


def build() -> str:
    fuller = collect(HERE, "")
    phylu = collect(PHYLU, "../../phylu/docs/")

    top = [e for e in fuller if "/" not in e["path"]]
    subs: dict[str, list[dict]] = {}
    for e in fuller:
        if "/" in e["path"]:
            subs.setdefault(e["path"].split("/")[0], []).append(e)

    by_name = {e["path"]: e for e in top}
    start = []
    for name, why in START_HERE:
        e = by_name.get(name)
        if e is not None:
            start.append((e, why))

    out = []
    out.append("<!doctype html>")
    out.append('<html lang="en"><head><meta charset="utf-8">')
    out.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    out.append("<title>fuller — documentation</title>")
    out.append("""<style>
:root{--bg:#0f1115;--panel:#161a21;--line:#252b35;--fg:#dfe4ec;--dim:#8d97a8;--acc:#7cc4ff;--acc2:#ffd479}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(15,17,21,.96);border-bottom:1px solid var(--line);padding:18px 24px 14px}
h1{margin:0 0 4px;font-size:20px;letter-spacing:.2px}
h1 b{color:var(--acc)}
.sub{color:var(--dim);font-size:13px;margin-bottom:12px}
#q{width:100%;padding:9px 12px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font-size:14px}
#q:focus{outline:none;border-color:var(--acc)}
main{padding:18px 24px 80px;max-width:1080px}
details.g{border:1px solid var(--line);border-radius:10px;margin:0 0 14px;background:var(--panel)}
details.g>summary{list-style:none;cursor:pointer;padding:11px 14px;user-select:none}
details.g>summary::-webkit-details-marker{display:none}
details.g>summary:before{content:"▸";display:inline-block;width:1em;color:var(--dim);transition:transform .12s}
details.g[open]>summary:before{transform:rotate(90deg)}
details.g>summary h2{display:inline;font-size:15px;margin:0;font-weight:600}
.n{color:var(--dim);font-weight:400;font-size:13px}
.note{color:var(--dim);font-size:13px;margin:0 14px 8px}
ul.l{list-style:none;margin:0;padding:0 8px 8px}
li.i{padding:7px 8px;border-top:1px solid var(--line)}
li.i:first-child{border-top:0}
li.i:hover{background:#1b2029}
a.t{color:var(--acc);text-decoration:none;font-weight:600}
a.t:hover{text-decoration:underline}
.d{color:var(--fg);font-size:13.5px;margin-top:2px}
.p{color:var(--dim);font-size:12px;margin-top:2px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.start{background:#131822;border-color:#2d3a4d}
.start li.i{background:#141b26}
.start .why{color:var(--acc2);font-size:13px}
.foot{color:var(--dim);font-size:12.5px;margin-top:26px;border-top:1px solid var(--line);padding-top:12px}
code{background:#1b2029;padding:1px 5px;border-radius:4px;font-size:12.5px}
a{color:var(--acc)}
.hidden{display:none}
.empty{color:var(--dim);padding:10px 4px;display:none}
</style></head><body>""")
    out.append(f"""<header>
<h1><b>fuller</b> — the documentation index</h1>
<div class="sub">{len(fuller)} documents here, {len(phylu)} in the sibling <b>phylu</b> checkout. The links are to the markdown files themselves: open one in Chrome and your markdown reader renders it.</div>
<input id="q" type="search" placeholder="filter by title, path or words…  (press / to focus)" autocomplete="off">
</header><main>""")

    # Start here.
    out.append("<details class='g start' open><summary><h2>Start here <span class='n'>%d</span></h2></summary><ul class='l'>" % len(start))
    for e, why in start:
        desc = f'<div class="d">{html.escape(e["description"])}</div>' if e["description"] else ""
        out.append(
            f'<li class="i" data-t="{html.escape((e["title"] + " " + e["path"] + " " + e["description"] + " " + why).lower())}">'
            f'<a class="t" href="{html.escape(e["link"])}">{html.escape(e["title"])}</a>'
            f'<div class="why">{html.escape(why)}</div>{desc}</li>'
        )
    out.append("</ul></details>")

    # The top level, by family.
    families: dict[str, list[dict]] = {}
    for e in top:
        families.setdefault(group_of(Path(e["path"]).name), []).append(e)
    for label, _ in FAMILIES:
        entries = families.get(label, [])
        if entries:
            out.append(section(label, sorted(entries, key=lambda e: e["path"].lower())))

    # The subdirectories.
    for name in sorted(subs):
        entries = sorted(subs[name], key=lambda e: e["path"].lower())
        out.append(section(name, entries, open_=(name in ("reports", "rule_proposals", "measurements"))))

    # The sibling checkout.
    if phylu:
        out.append(section("phylu (sibling checkout)", sorted(phylu, key=lambda e: e["path"].lower()), note="Linked by a relative path: it assumes the two checkouts are siblings."))

    out.append("""<p class="empty" id="none">nothing matches that.</p>
<div class="foot">Generated by <code>docs/build_index.py</code> — the page is a map of the tree, so it is checked
against the tree (<code>python3 docs/build_index.py --check</code>) and a hand-edited index fails the test.
Edit the generator, not this file.</div>
</main>
<script>
const q=document.getElementById('q'),items=[...document.querySelectorAll('li.i')],groups=[...document.querySelectorAll('details.g')],none=document.getElementById('none');
function apply(){
  const v=q.value.trim().toLowerCase();
  for(const g of groups){
    let n=0;
    for(const li of g.querySelectorAll('li.i')){
      const hit=!v||li.dataset.t.includes(v);
      li.classList.toggle('hidden',!hit); if(hit)n++;
    }
    g.style.display=n?'':'none';
    if(n)g.open=true;
  }
  let total=0; for(const li of items) if(!li.classList.contains('hidden')) total++;
  none.style.display=total? 'none':'block';
}
q.addEventListener('input',apply);
document.addEventListener('keydown',e=>{ if(e.key==='/'&&document.activeElement!==q){e.preventDefault();q.focus();} if(e.key==='Escape'){q.value='';apply();q.blur();} });
</script>
</body></html>""")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare with the committed page instead of writing it")
    args = ap.parse_args()
    page = build()
    if args.check:
        if not OUT.exists():
            print(f"build_index: {OUT} does not exist", file=sys.stderr)
            return 1
        want = OUT.read_text(encoding="utf-8")
        if want != page:
            print(f"build_index: {OUT} differs from what the generator writes", file=sys.stderr)
            print("build_index: run `python3 docs/build_index.py` to write it", file=sys.stderr)
            return 1
        print(f"build_index: {OUT} MATCHES the generator")
        return 0
    OUT.write_text(page, encoding="utf-8")
    print(f"build_index: wrote {OUT} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
