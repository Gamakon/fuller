"""Bisect a WGSL file to the top-level item that naga's front end rejects.

The prelude is one flat list of top-level items (functions, aliases, consts).
Cutting the file after item k leaves a prefix that is still a syntactically
complete module, so the probe can be run on a series of prefixes and the
first failing one tells us which item broke it.

    python3 tools/bisect_wgsl.py prelude.wgsl            # coarse, then fine
"""
import os
import subprocess
import sys

PROBE = os.environ.get(
    "PROBE",
    "target/release/examples/futhark_wgsl_probe",
)


def items(text):
    """The (start, end) line spans of the top-level items."""
    spans = []
    start = None
    depth = 0
    for i, line in enumerate(text.split("\n"), start=1):
        if start is None and line and not line[0].isspace() and not line.startswith("//"):
            start = i
            depth = 0
        if start is not None:
            depth += line.count("{") - line.count("}")
            if depth == 0 and (line.rstrip().endswith("}") or line.rstrip().endswith(";")):
                spans.append((start, i))
                start = None
    return spans


def write_prefix(text, spans, k, path):
    end = spans[k - 1][1]
    lines = text.split("\n")[:end]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def probe(paths):
    out = subprocess.run([PROBE] + paths, capture_output=True, text=True)
    verdicts = {}
    current = None
    for line in out.stdout.split("\n"):
        if line.startswith("=== "):
            current = line[4:].split(" ")[0]
        elif line.strip().startswith("naga "):
            verdicts[current] = "OK" if " OK " in line or line.strip().endswith("OK") else "FAIL"
    return verdicts


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = argv[1]
    text = open(path).read()
    spans = items(text)
    print(f"{len(spans)} top-level items in {path}")
    outdir = "/tmp/wgsl_bisect"
    os.makedirs(outdir, exist_ok=True)

    # Coarse: binary search for the smallest prefix that fails.
    lo, hi = 0, len(spans)
    cache = {}

    def fails(k):
        if k in cache:
            return cache[k]
        p = os.path.join(outdir, f"p{k}.wgsl")
        write_prefix(text, spans, k, p)
        v = probe([p]).get(p, "FAIL")
        cache[k] = v == "FAIL"
        return cache[k]

    if not fails(len(spans)):
        print("the whole file parses: nothing to bisect")
        return 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fails(mid):
            hi = mid
        else:
            lo = mid
    print(f"the first failing prefix is {hi} items; the culprit is item {hi}:")
    s, e = spans[hi - 1]
    for i, line in enumerate(text.split("\n")[s - 1:e], start=s):
        print(f"{i:5d}| {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
