#!/usr/bin/env python3
"""A live view of an HGP run: the population's fitness evaluation as it
proceeds. Reads the run directory (logs/HGP/<run>): gen0/index.tsv (the
initial population and its strategies), ladder.tsv (one line per evaluated
individual), ladder_kNN.log (the individual under evaluation, its current
stage) and run.log. Refreshes every few seconds.

    python3 tools/hgp_watch.py logs/HGP/001 [--interval 5] [--once]
"""
import os, sys, time, glob, re

STRATEGIES = ["thread per cell", "2x2 register block", "4x4 block + vec4 loads", "shared-memory tile", "vec4 loads/accumulators", "split accumulators", "unrolled item loop", "tile + 4x4 block", "block + vec4 + unroll", "free choice"]

def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [l.rstrip("\n").split("\t") for l in f]
    if not lines:
        return []
    head = lines[0]
    return [dict(zip(head, l)) for l in lines[1:] if len(l) == len(head)]

def stage_of(log):
    """The last stage line of an individual's evaluation log."""
    try:
        with open(log) as f:
            lines = [l.strip() for l in f if l.startswith("[")]
    except OSError:
        return ""
    if not lines:
        return ""
    m = re.search(r"\] (.*)", lines[-1])
    return m.group(1)[:90] if m else lines[-1][:90]

import math

def hff(r):
    """The HFF angle of a timed individual from its ledger line: six columns in
    [0, 1], 0 best (three time ratios, three relative errors); theta =
    acos(1 - mean(x^2)). Lines written before the HFF columns existed carry
    one worst-error figure, used for all three accuracy columns."""
    if r.get("rung") != "4":
        return None
    if r.get("hff") not in (None, "", "-"):
        return float(r["hff"])
    xs = []
    for part in r["speedups"].split(","):
        m = float(part.split(":")[1].split("[")[0])
        xs.append(min(1.0, 1.0 / m))
    e = float(r["worst_pair_ratio"])
    xs += [min(1.0, e / 2.0)] * 3
    energy = sum(x * x for x in xs) / len(xs)
    return math.acos(max(-1.0, min(1.0, 1.0 - min(1.0, energy))))

def render(run):
    pop = read_tsv(os.path.join(run, "gen0", "index.tsv"))
    evals = {os.path.basename(r["kernel"]).replace(".wgsl", ""): r for r in read_tsv(os.path.join(run, "ladder.tsv"))}
    out = []
    out.append(f"HGP run {run}   {time.strftime('%H:%M:%SZ', time.gmtime())}   population {len(pop)}   evaluated {len(evals)}")
    feasible = [r for r in evals.values() if r["rung"] in ("3", "4") and r["pairs"].split("/")[0] == r["pairs"].split("/")[1] and r["pairs"] != "0/0"]
    timed = [r for r in feasible if r["rung"] == "4"]
    infeasible = [r for r in evals.values() if r not in feasible]
    speedups = sorted(float(r["geo"]) for r in timed)
    angles = sorted(a for a in (hff(r) for r in timed) if a is not None)
    best = speedups[-1] if speedups else 0.0
    median = speedups[len(speedups) // 2] if speedups else 0.0
    faster = sum(1 for s in speedups if s > 1.05)
    best_a = angles[0] if angles else float("nan")
    med_a = angles[len(angles) // 2] if angles else float("nan")
    out.append(f"feasible {len(feasible)}   infeasible {len(infeasible)}   timed {len(timed)}   faster than shipped {faster}   best HFF {best_a:.3f}   median HFF {med_a:.3f}   best speed-up {best:.2f}x   median {median:.2f}x")
    out.append("")
    out.append(f"{'id':<5} {'strategy':<24} {'feasible':<10} {'HFF':>7} {'speed-up':>9} {'min speed-up':>13}  status")
    for p in pop:
        k = p["id"]
        strat = STRATEGIES[int(p["strategy"])] if p["strategy"].isdigit() else p["strategy"]
        r = evals.get(k)
        if r is None:
            log = os.path.join(run, f"ladder_{k}.log")
            status = stage_of(log) if os.path.exists(log) else "waiting"
            out.append(f"{k:<5} {strat:<24} {'':<10} {'':>7} {'':>9} {'':>13}  {status}")
            continue
        feas = "yes" if r in feasible else "NO"
        sp = f"{float(r['geo']):.2f}x" if r["rung"] == "4" else "-"
        worst = f"{float(r['worst']):.2f}x" if r["rung"] == "4" else "-"
        a = hff(r)
        ang = f"{a:.3f}" if a is not None else "-"
        detail = r["detail"][:60]
        out.append(f"{k:<5} {strat:<24} {feas:<10} {ang:>7} {sp:>9} {worst:>13}  {detail}")
    return "\n".join(out)

def main():
    args = sys.argv[1:]
    run = args[0] if args and not args[0].startswith("--") else "logs/HGP/001"
    interval = 5
    once = "--once" in args
    if "--interval" in args:
        interval = int(args[args.index("--interval") + 1])
    while True:
        text = render(run)
        if once:
            print(text)
            return
        os.system("clear")
        print(text)
        time.sleep(interval)

if __name__ == "__main__":
    main()
