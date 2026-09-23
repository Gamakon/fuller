"""Summarise the TSR sweep's run logs into one table.

Reads every logs/tsr/tsr_*/run.log and prints, per fit: generations,
ms/generation, train and validation 1-R2, log10 p, typed refusals with their
two honest denominators, and the depth histogram.

usage: python3 logs/tsr/summarise.py [--cards]
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def parse(log: pathlib.Path) -> dict | None:
    t = log.read_text(errors="replace")
    out: dict = {"tag": log.parent.name}
    m = re.search(r"(\d+) generations in ([\d.]+) s = ([\d.]+) ms per generation", t)
    if not m:
        return None
    out["gens"], out["secs"], out["ms"] = int(m[1]), float(m[2]), float(m[3])
    m = re.search(r"^TYPED\tceiling=(\S+)\trefused=(\d+)\tdepths=(.*)$", t, re.M)
    if m:
        out["ceiling"], out["refused"] = m[1], int(m[2])
        out["depths"] = {int(k): int(v) for k, v in (p.split(":") for p in m[3].split())}
    m = re.search(r"1 - R²: train ([\d.e+-]+), validation ([\d.e+-]+)", t)
    if m:
        out["train"], out["val"] = float(m[1]), float(m[2])
    m = re.search(r"^PVALUE\t(\S+)\t(\S+)\t(\d+)", t, re.M)
    if m:
        out["log10p"] = m[2]
    m = re.search(r"^GENES\t(\d+)\t(\d+)\t(\d+)", t, re.M)
    if m:
        out["unique"], out["oversized"] = int(m[1]), int(m[2])
    m = re.search(r"^GENERATIONS\t\d+\t(\S+)", t, re.M)
    if m:
        out["stopped_by"] = m[1]
    # NOTE ON ORDER: `stopped_by` must be parsed before this.
    # `early_stop` is what the engine sets when a fit MEETS THE STOP BAR --
    # train, validation and (when there is one) edge 1-R2 under the bar AND
    # log10 p under its own. It is not the same as "recovered the law": the
    # skill file is explicit that a model scoring R2 1.000000 with the wrong
    # FORM is a miss. The form is checked by hand against the true law.
    out["met_stop_bar"] = out.get("stopped_by") == "early_stop"
    m = re.search(r"^MODEL_INFIX\t(.*)$", t, re.M)
    if m:
        out["model"] = m[1]
    m = re.search(r"R² on the unseen test rows: ([\d.]+)", t)
    if m:
        out["test_r2"] = m[1]
    # The population the fit ran, for the gene-slot denominator.
    m = re.search(r"population \d+ x \((\d+)\+(\d+)\) = (\d+)", t)
    if m:
        out["pop"] = int(m[3])
    return out


def main() -> None:
    rows = []
    for d in sorted(ROOT.glob("tsr_*")):
        log = d / "run.log"
        if log.exists() and (r := parse(log)):
            rows.append(r)
    if not rows:
        print("no finished fits yet")
        return
    print(f"{'fit':44} {'gens':>7} {'ms/gen':>7} {'train 1-R2':>11} {'val 1-R2':>11} {'log10p':>8} {'refused%slots':>13} {'>2':>5}")
    for r in rows:
        slots = r["gens"] * r.get("pop", 1200) * 3
        pct = 100 * r.get("refused", 0) / slots if slots else 0
        deep = sum(n for d, n in r.get("depths", {}).items() if d > 2)
        tot = sum(r.get("depths", {}).values()) or 1
        print(
            f"{r['tag']:44} {r['gens']:7} {r['ms']:7.1f} {r.get('train', float('nan')):11.3e} "
            f"{r.get('val', float('nan')):11.3e} {str(r.get('log10p', '-')):>8} {pct:12.2f}% "
            f"{100 * deep / tot:4.1f}%"
        )
    print()
    print(f"fits: {len(rows)}   met the stop bar: {sum(r.get('met_stop_bar', False) for r in rows)}")
    ceil_used = [r for r in rows if r.get("depths", {}).get(2, 0) > 0]
    print(f"fits whose population reaches the T2 ceiling: {len(ceil_used)} of {len(rows)}")
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
