#!/usr/bin/env python3
"""THE REGEX KINGDOM AS ROWS of the gene frame's fixed schema (docs/DESIGN_geneframe.md 3.3 and 8.2;
docs/PLAN_futhark_kingdom.md 7.7, task K2).

Four data types (Pattern, CharClass, Char, I), the POSIX ERE alphabet as Symbol and Argument rows in the order the
typed regex kingdom always had them, no Flow row (a regex carries no property up or down), and ONE Guard row: the
counted repetition `regex_rep_range(CharClass, I lo, I hi)` is legal only when `lo <= hi` (POSIX rejects `a{5,3}`),
`must`, repaired at the node by rendering the bounds swapped. `I` carries `value`, the count a `?_I` placeholder
stands for, which is how the guard reads its two bounds.

The constant pools (`C_Char`, `C_Pat`: the 95 printable bytes; `C_I`: the counts 0..=8) are the kingdom's data in
code (`src/regex_typed.rs`), as the master's pools are the Futhark kingdom's: a pool is not one of the four kinds of
row. The spelling column is the regex surface the compiler's printer writes (phylu `regex_compile::to_regex_string`),
so a table reader can render a pattern without the compiler; the compiler stays the renderer of record.

    python3 -I gen_tables.py            # writes the kingdom `regex` into kingdoms/master/ (write_rows.py)
    python3 -I gen_tables.py --check    # 0 when the committed master rows equal a fresh generation

The rows live in the one set of master tables (`kingdoms/master/`, task K0: every table carries a `kingdom` column),
as the kingdom `regex`. Loaded by fuller `src/regex_typed.rs` (`RegexRows::load`, through
`geneframe::master::select("regex")`), which keeps or drops the `?_Pat` literal-Pattern terminal by the run's
`pattern_literal` switch, as a view keeps the rows a problem has.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(os.path.dirname(HERE), "master")
sys.path.insert(0, MASTER)
import write_rows  # noqa: E402

KINGDOM = "regex"
DESCRIPTION = "the regex kingdom: POSIX ERE over Pattern, CharClass, Char and I; one guard, the counted range's bounds in order"

TYPES = ["Pattern", "CharClass", "Char", "I"]
PAT, CC, CH, I = TYPES

# (name, alias, inputs, output, spelling). The order is the typed regex kingdom's (the compiler's opcode names), with
# the counted range after the two single-count repetitions and the placeholders last.
SYMBOLS = [
    ("regex_dot", ".", [], PAT, "."),
    ("regex_anchor_start", "^", [], PAT, "^"),
    ("regex_anchor_end", "$", [], PAT, "$"),
    ("regex_empty", "", [], PAT, ""),
    ("regex_cc_digit", r"\d", [], CC, r"\d"),
    ("regex_cc_word", r"\w", [], CC, r"\w"),
    ("regex_cc_space", r"\s", [], CC, r"\s"),
    ("regex_star", "*", [PAT], PAT, "(?:{0})*"),
    ("regex_plus", "+", [PAT], PAT, "(?:{0})+"),
    ("regex_opt", "?", [PAT], PAT, "(?:{0})?"),
    ("regex_concat", "concat", [PAT, PAT], PAT, "{0}{1}"),
    ("regex_alt", "|", [PAT, PAT], PAT, "(?:{0}|{1})"),
    # Counted repetition takes a CLASS, not a pattern: a class folds to one 256-bit set, so a repetition of it emits
    # at most 2k instructions however large the class tree is, and the compiler's buffers are sized from the layout.
    ("regex_rep_n", "{n}", [CC, I], PAT, "(?:{0}){{{=in1.value}}}"),
    ("regex_rep_upto", "{0,k}", [CC, I], PAT, "(?:{0}){{0,{=in1.value}}}"),
    ("regex_rep_range", "{lo,hi}", [CC, I, I], PAT, "(?:{0}){{{=in1.value},{=in2.value}}}"),
    ("regex_class_of", "class_of", [CC], PAT, "[{0}]"),
    ("regex_lit", "lit", [CH], PAT, "{0}"),
    ("regex_cc_range", "a-z", [CH, CH], CC, "{0}-{1}"),
    ("regex_cc_union", "cc_union", [CC, CC], CC, "{0}{1}"),
    ("regex_cc_negate", "^CC", [CC], CC, "^{0}"),
]
# The placeholders: (name, data type). `?_Pat` is the literal-Pattern terminal the `pattern_literal` switch adds.
CONSTANTS = [("?_Char", CH), ("?_I", I), ("?_Pat", PAT)]
PATTERN_LITERAL = "?_Pat"

# The one guard: the range's bounds in order, repaired by the spelling with the bounds exchanged.
GUARDED = "regex_rep_range"
GUARD = "in1.value <= in2.value"
REPAIR = "(?:{0}){{{=in2.value},{=in1.value}}}"


def four_rows():
    out = [f"data_type\t{t}\tscalar" for t in TYPES]
    out.append(f"carries\t{I}\tvalue")
    out.append(f"constant\t{CH}\t?_Char\t-")
    out.append(f"constant\t{I}\t?_I\tvalue=?")
    out.append(f"constant\t{PAT}\t?_Pat\t-")
    for name, alias, _ins, _out, _sp in SYMBOLS:
        out.append(f"symbol\t{name}\t{alias}\tfunction\t-\t-")
    for name, ty in CONSTANTS:
        out.append(f"symbol\t{name}\t{name}\tconstant\t-\t-")
    for name, _alias, _ins, _out, spelling in SYMBOLS:
        out.append(f"spelling\t{name}\t\t{spelling}")
    for name, _alias, ins, res, _sp in SYMBOLS:
        out += [f"argument\t{name}\t{i}\tin\t{t}\t-" for i, t in enumerate(ins)]
        out.append(f"argument\t{name}\t0\tout\t{res}\t-")
    for name, ty in CONSTANTS:
        out.append(f"argument\t{name}\t0\tout\t{ty}\t-")
    out.append(f"guard\t{GUARDED}\t0\tmust\t{GUARD}")
    out.append(f"repair\t{GUARDED}\t0\tnode\ttrue\ttrue\t{REPAIR}")
    out.append(f"entry\tregex\t{PAT}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare the committed master rows with a fresh generation")
    ap.add_argument("--named", action="store_true", help="print the named text form instead of writing")
    a = ap.parse_args()
    text = four_rows()
    if a.named:
        sys.stdout.write(text)
        return
    rows = write_rows.rows_of_named(text)
    kingdom_row = (KINGDOM, "", DESCRIPTION, PAT, "")
    if a.check:
        bad = write_rows.check_kingdom(KINGDOM, rows, kingdom_row)
        for b in bad:
            print(b)
        print(f"check: the master rows of {KINGDOM} {'equal' if not bad else 'DIFFER FROM'} a fresh generation")
        sys.exit(1 if bad else 0)
    write_rows.write_kingdom(KINGDOM, rows, kingdom_row)
    print(f"{len(SYMBOLS) + len(CONSTANTS)} symbols, 1 guard -> kingdoms/master/ as the kingdom {KINGDOM}")


if __name__ == "__main__":
    main()
