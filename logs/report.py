#!/usr/bin/env python3
"""THE EXPERIMENT RECORD for a cascade: what was run, how it was configured,
when, for how long, and what came back.

    logs/report.py                  the record, to the terminal
    logs/report.py --md > R.md      the same as markdown, to keep

The tally (`logs/tally.sh`) is a live count you glance at. This is the thing you
file: it carries the settings that produced the numbers, so a result can be
argued with months later.

WHY IT IS A SEPARATE READER. Every fit writes a `card.json` holding its whole
Config, the git commit and the derived objective count -- but a cascade's fits
all write to the SAME path, so law 133 overwrites law 1 and only the last card
survives. Rather than change the engine's card contract, this reads the card
that is there (the settings are identical across the sweep by construction --
one script, one environment) and takes the per-law facts from the run logs,
which are never overwritten.

WHAT IT REFUSES TO DO. It does not recompute a verdict. `solved/<law>` is the
engine's own early_stop marker and the run log's trailer is the engine's own
timing; both are read, neither is inferred. A law with no log is reported as not
reached, not as a failure.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = "/Users/andrewmorgan/Dev/gamakon/fuller"
ORDER = f"{ROOT}/experiments/order133.txt"

# The settings worth printing, grouped as they are reasoned about. Anything in
# the card but not here is still dumped under "every other setting", so a knob
# added later cannot silently vanish from the record.
GROUPS = [
    ("Population", ["pop_intake", "pop_champion", "n_pairs", "float_zone", "lanes"]),
    ("Genes", ["head", "n_genes", "n_rnc", "rnc_lo", "rnc_hi", "typed_depth"]),
    ("Budget", ["max_seconds", "max_generations", "progress_every"]),
    ("Selection", ["tournament_fraction", "elites", "balanced_tournaments",
                   "champion_tournament_fraction", "champion_elites", "champion_open_fight"]),
    ("The pump / ALPS", ["pump_every", "cohort_merge", "champion_cohort_merge",
                         "promote_fraction", "arrival_children", "cross_every", "k_migrants"]),
    ("Gene editors", ["fold_every", "fold_top_k", "snap_every", "snap_top_k",
                      "snap_rel_tol", "snap_r2_drop", "beam_every", "beam_width",
                      "beam_tree", "beam_wraps", "cleanse", "vhead_every", "vhead_start"]),
    ("HFF & the stop bar", ["gene_subsets", "tower", "redundancy", "compounds",
                            "log_scale", "hff_without_validation", "hff_on_host",
                            "stop_log10_p", "stop_one_minus_r2"]),
    ("Data", ["smogd"]),
]


def live_run_dir():
    """The most recently written run directory that holds results."""
    cands = [d for d in (f"{ROOT}/logs/RUN", f"{ROOT}/logs/cascade133") if os.path.isdir(f"{d}/solved")]
    return max(cands, key=os.path.getmtime) if cands else None


def read_log(path):
    """The per-law facts the engine itself wrote: generations, seconds, pace,
    why it stopped, the model, and test R2. Returns {} for a log with no
    trailer, which is a fit still running."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    out = {}
    m = re.search(r"(\d+) generations in ([\d.]+) s = ([\d.]+) ms per generation", text)
    if m:
        out["gens"], out["secs"], out["ms_per_gen"] = int(m[1]), float(m[2]), float(m[3])
    m = re.search(r"= (\d+) per second", text)
    if m:
        out["per_sec"] = int(m[1])
    m = re.search(r"stopped by (\w+)", text)
    if m:
        out["stopped_by"] = m[1]
    m = re.search(r"^MODEL_INFIX\t(.*)$", text, re.M)
    if m:
        out["model"] = m[1].strip()
    # A fit that reported no test score prints a dash here, so the value is
    # parsed rather than assumed to be a number.
    m = re.search(r"R² on the unseen test rows: (\S+)", text)
    if m:
        try:
            out["test_r2"] = float(m[1])
        except ValueError:
            pass
    return out


def collect(run):
    laws = [l.strip() for l in open(ORDER) if l.strip()]
    stages = sorted(
        (d for d in os.listdir(run) if d.endswith("s") and os.path.isdir(f"{run}/{d}")),
        key=lambda s: int(s[:-1]),
    )
    rows = []
    for law in laws:
        r = {"law": law, "verdict": "...", "stage": None, "attempts": []}
        for st in stages:
            p = f"{run}/{st}/{law}/run.log"
            if os.path.exists(p):
                a = read_log(p)
                a["stage"] = st
                r["attempts"].append(a)
        if r["attempts"]:
            last = r["attempts"][-1]
            r.update({k: v for k, v in last.items() if k != "stage"})
            r["stage"] = last["stage"]
            if os.path.exists(f"{run}/solved/{law}"):
                r["verdict"] = "FOUND"
            elif "gens" in last:
                r["verdict"] = "UNFOUND"
            else:
                r["verdict"] = "RUNNING"
        rows.append(r)
    return rows, stages


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main():
    md = "--md" in sys.argv
    run = live_run_dir()
    if not run:
        print("no cascade on disk")
        return
    card = {}
    if os.path.exists(f"{run}/card.json"):
        card = json.load(open(f"{run}/card.json"))
    cfg = card.get("config", {})
    rows, stages = collect(run)

    found = [r for r in rows if r["verdict"] == "FOUND"]
    unfound = [r for r in rows if r["verdict"] == "UNFOUND"]
    total_secs = sum(a.get("secs", 0.0) for r in rows for a in r["attempts"])
    total_gens = sum(a.get("gens", 0) for r in rows for a in r["attempts"])
    mtimes = [os.path.getmtime(f"{run}/{r['stage']}/{r['law']}/run.log") for r in rows if r["stage"]]

    h1, h2, b = ("# ", "## ", "") if md else ("", "", "")
    rule = (lambda: print()) if md else (lambda: print("-" * 100))

    print(f"{h1}THE 133 — experiment record")
    print()
    print(f"{h2}The run")
    print()
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    facts = [
        ("directory", run.replace(ROOT + "/", "")),
        ("started", ts(min(mtimes)) if mtimes else "—"),
        ("last write", ts(max(mtimes)) if mtimes else "—"),
        ("report written", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")),
        ("git commit", card.get("code", {}).get("git_commit", "—")[:12]),
        ("tree dirty", fmt(card.get("code", {}).get("git_dirty"))),
        ("seed", fmt(card.get("run", {}).get("seed"))),
        ("stages", " → ".join(stages) if stages else "—"),
        ("HFF objectives", fmt(card.get("derived", {}).get("hff_objectives"))),
        ("population", fmt(card.get("derived", {}).get("population"))),
    ]
    for k, v in facts:
        print(f"  {k:18} {v}")
    print()
    print(f"{h2}The result")
    print()
    print(f"  FOUND              {len(found)} / {len(rows)}   ({100*len(found)//max(len(rows),1)}%)")
    print(f"  UNFOUND            {len(unfound)}")
    print(f"  not reached        {len(rows) - len(found) - len(unfound)}")
    print(f"  compute            {total_secs/3600:.2f} h over {total_gens:,} generations")
    if found:
        g = sorted(r.get("gens", 0) for r in found)
        print(f"  generations to a law   median {g[len(g)//2]}   max {g[-1]}")
    for st in stages:
        won = [r for r in found if r["stage"] == st]
        att = [r for r in rows if any(a["stage"] == st for a in r["attempts"])]
        print(f"  stage {st:>5}        {len(won):3} found of {len(att):3} attempted")
    print()
    print(f"{h2}The settings that produced it")
    print()
    shown = set()
    for name, keys in GROUPS:
        present = [(k, cfg[k]) for k in keys if k in cfg]
        if not present:
            continue
        shown.update(k for k, _ in present)
        print(f"  {name}")
        for k, v in present:
            print(f"    {k:32} {fmt(v)}")
    rest = sorted(set(cfg) - shown)
    if rest:
        print("  every other setting")
        for k in rest:
            print(f"    {k:32} {fmt(cfg[k])}")
    syn = card.get("synthetic", {})
    if syn:
        print("  synthetic data")
        for k, v in syn.items():
            print(f"    {k:32} {fmt(v)}")
    print()
    print(f"{h2}Every law")
    print()
    print(f"  {'VERDICT':9} {'LAW':26} {'STAGE':>5} {'GEN':>6} {'SEC':>7} {'ms/gen':>7}  MODEL")
    rule()
    for r in rows:
        print(f"  {r['verdict']:9} {r['law']:26} {fmt(r.get('stage')):>5} "
              f"{fmt(r.get('gens')):>6} {r.get('secs', 0):7.1f} {r.get('ms_per_gen', 0):7.1f}  "
              f"{(r.get('model') or '—')[:70]}")


if __name__ == "__main__":
    main()
