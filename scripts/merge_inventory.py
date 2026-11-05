"""The merge inventory (plan 6.2.2 findings 4 and 5): every cross-crate compile-time path and every path naming the
other repository, with what it must become under the one repository (phylu root: crates/fuller, crates/phylu).

    python3 -I scripts/merge_inventory.py           # writes docs/fuller_merge_inventory.tsv
    python3 -I scripts/merge_inventory.py --check   # 0 when the committed inventory equals a fresh one

It reads both checkouts' tracked files (git grep), so run it with fuller and phylu side by side under GAMAKON."""
import os
import re
import subprocess
import sys

G = os.environ.get("GAMAKON", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OLD = {"fuller": f"{G}/fuller", "phylu": f"{G}/phylu", "hff": f"{G}/hff"}
NEW_ROOT = f"{G}/phylu"


def new_of(path):
    """The new absolute location of an old absolute path."""
    p = os.path.normpath(path)
    for repo in ("fuller", "phylu"):
        root = OLD[repo]
        if p == root or p.startswith(root + "/"):
            rel = os.path.relpath(p, root)
            top = rel.split("/")[0]
            if top == "target":
                return os.path.join(NEW_ROOT, rel)  # one workspace target at the root
            if repo == "fuller":
                if top == "docs":
                    return os.path.join(NEW_ROOT, "docs/fuller", rel[len("docs"):].lstrip("/"))
                if top == "papers":
                    return os.path.join(NEW_ROOT, "papers", rel[len("papers"):].lstrip("/"))
                if top == "vendor-egglog":
                    return os.path.join(NEW_ROOT, "vendor/egglog", rel[len("vendor-egglog"):].lstrip("/"))
                if top in ("futhark-rust-rt", "futhark-wgpu-shim"):
                    return os.path.join(NEW_ROOT, "crates", rel)
                if top == "futhark-rust-backend":
                    return os.path.join(NEW_ROOT, "tools", rel)
                return os.path.normpath(os.path.join(NEW_ROOT, "crates/fuller", rel if rel != "." else ""))
            if rel == ".":
                return NEW_ROOT
            if top in ("docs", "logs", "papers"):
                return os.path.join(NEW_ROOT, rel)
            return os.path.normpath(os.path.join(NEW_ROOT, "crates/phylu", rel if rel != "." else ""))
    return p  # hff and anything else stay where they are


def crate_root(repo, file_rel):
    """The old crate root a file belongs to (the directory holding its Cargo.toml)."""
    root = OLD[repo]
    for side in ("vendor-egglog", "futhark-rust-rt", "futhark-wgpu-shim", "futhark-rust-backend"):
        if repo == "fuller" and (file_rel == side or file_rel.startswith(side + "/")):
            return os.path.join(root, side)
    return root


def new_crate(repo, f):
    """The new directory of the crate a file belongs to (phylu's crate is crates/phylu, not the repository root)."""
    root = crate_root(repo, f)
    return os.path.join(NEW_ROOT, "crates/phylu") if root == OLD["phylu"] else new_of(root)


rows = []


HISTORY = ("logs/", "docs/measurements/", "docs/reviews/", "docs/issues/", "stale/", "papers/", "experiments/")


def add(repo, f, line, text, kind, become):
    """`action`: rewrite (code, scripts, live docs, build files) or leave (a record of a past run or review: its paths
    name where things were then, and git history keeps them)."""
    action = "leave: a record of a past run or review" if f.startswith(HISTORY) or "/results/" in f or "/reports/" in f else ("none: still right" if become.startswith("unchanged") else "rewrite")
    rows.append((f"{repo}/{f}", str(line), text.strip().replace("\t", " ")[:300], kind, become, action))


def grep(repo, pattern):
    out = subprocess.run(["git", "-C", OLD[repo], "grep", "-n", "-I", "-E", pattern], capture_output=True, text=True).stdout
    for l in out.splitlines():
        f, n, text = l.split(":", 2)
        if repo == "fuller" and f in ("docs/fuller_merge_inventory.tsv", "scripts/merge_inventory.py"):
            continue  # the inventory and its generator name every path they list
        yield f, int(n), text


LIT = re.compile(r'"([^"]*)"')
for repo in ("fuller", "phylu"):
    seen = set()
    # A. #[path], B. include_str!/include_bytes!: the literal path resolved against the file's directory.
    for f, n, text in grep(repo, r'#\[path *=|include_(str|bytes)!'):
        if not f.endswith(".rs"):
            continue
        lits = [x for x in LIT.findall(text) if "/" in x or x.endswith((".fut", ".tsv", ".wgsl", ".md", ".rs", ".txt", ".json"))]
        concat = "concat!" in text
        if concat:
            lits = [x for x in LIT.findall(text) if x != "CARGO_MANIFEST_DIR" and x != "OUT_DIR"][:1]
            if "OUT_DIR" in text:
                continue
        base = os.path.dirname(os.path.join(OLD[repo], f))
        if "env!(\"CARGO_MANIFEST_DIR\")" in text:
            base = crate_root(repo, f)
        for lit in lits:
            if concat and not lit.startswith(("../", "/")) and lit != lits[0]:
                continue
            old_target = os.path.normpath(os.path.join(base, lit.lstrip("/") if base == crate_root(repo, f) else lit))
            new_rel = os.path.relpath(new_of(old_target), os.path.dirname(new_of(os.path.join(OLD[repo], f))) if base != crate_root(repo, f) else new_crate(repo, f))
            if os.path.normpath(lit) != os.path.normpath(new_rel) and not lit.startswith("{"):
                kind = "#[path]" if "#[path" in text else ("include_bytes!" if "include_bytes" in text else "include_str!")
                add(repo, f, n, text, kind, f"{lit} -> {new_rel}" + (" (the head of a concat!)" if concat else ""))
                seen.add((f, n))
    # C. CARGO_MANIFEST_DIR joined to a path that leaves the crate or names a moved directory.
    for f, n, text in grep(repo, r'CARGO_MANIFEST_DIR'):
        if (f, n) in seen:
            continue
        for lit in LIT.findall(text):
            if lit == "CARGO_MANIFEST_DIR" or "/" not in lit and lit not in ("..", "docs", "logs", "papers", "vendor-egglog"):
                continue
            old_target = os.path.normpath(os.path.join(crate_root(repo, f), lit.lstrip("/")))
            new_rel = os.path.relpath(new_of(old_target), new_crate(repo, f))
            if os.path.normpath(lit.lstrip("/")) != new_rel:
                add(repo, f, n, text, "CARGO_MANIFEST_DIR", f"{lit} -> {new_rel}")
                seen.add((f, n))
    # D. ../fuller, ../phylu, ../hff anywhere. A Rust include or #[path] reads relative to its file; everything else
    # (Cargo.toml, scripts run from the root, docs) relative to the repository root, which becomes the crate root.
    for f, n, text in grep(repo, r'\.\./(fuller|phylu|hff)([^a-zA-Z_]|$)'):
        if (f, n) in seen:
            continue
        file_relative = (f.endswith(".rs") and re.search(r'include_(str|bytes)!|#\[path', text)) or f.endswith((".md", ".html"))
        old_base = os.path.dirname(os.path.join(OLD[repo], f)) if file_relative else crate_root(repo, f)
        new_base = os.path.dirname(new_of(os.path.join(OLD[repo], f))) if file_relative else new_crate(repo, f)
        becomes, changed = [], False
        for m in re.finditer(r'((?:\.\./)+)(fuller|phylu|hff)((?:/[\w.\-*<>{}]+)*)', text):
            old_target = os.path.normpath(os.path.join(old_base, m.group(0)))
            base_new = new_base
            if file_relative and not os.path.exists(old_target):
                old_target = os.path.normpath(os.path.join(crate_root(repo, f), m.group(0)))
            new_rel = os.path.relpath(new_of(old_target), base_new)
            changed |= os.path.normpath(m.group(0)) != new_rel
            becomes.append(f"{m.group(0)} -> {new_rel}" + (" (from the file)" if file_relative else " (from the crate root)"))
        if not becomes:
            continue
        add(repo, f, n, text, "../ path", "; ".join(becomes) if changed else "unchanged: " + "; ".join(becomes))
        seen.add((f, n))
    # E. absolute gamakon/ paths.
    for f, n, text in grep(repo, r'gamakon/(fuller|phylu|hff)'):
        if (f, n) in seen:
            continue
        becomes = []
        for m in re.finditer(r'(/Users/andrewmorgan/[Dd]ev/gamakon/|~/[Dd]ev/gamakon/|gamakon/)(fuller|phylu|hff)((?:/[\w.\-*<>{}]+)*)', text):
            if m.group(2) == "hff":
                continue
            old_target = os.path.join(OLD[m.group(2)], m.group(3).lstrip("/"))
            new_rel = os.path.relpath(new_of(old_target), NEW_ROOT)
            becomes.append(f"{m.group(0)} -> {m.group(1)}phylu" + ("" if new_rel == "." else f"/{new_rel}"))
        if becomes:
            same = all(b.split(" -> ")[0] == b.split(" -> ")[1] for b in becomes)
            add(repo, f, n, text, "absolute path", ("unchanged: " if same else "") + "; ".join(becomes))

# The layout-bound code of review finding 5, by hand: each is code that knows where the other repository is.
MANUAL = {
    ("fuller", "docs/build_index.py", 37): "PHYLU = REPO.parent / \"phylu\" / \"docs\" -> the one docs tree: collect the root's docs/ (phylu's) and docs/fuller/ (fuller's) under the repository root; the generator moves to docs/ at the root (finding 5)",
    ("fuller", "docs/build_index.py", 177): "../../phylu/docs/ -> the links of the one index: phylu's docs at docs/, fuller's at docs/fuller/ (relative to docs/index.html); regenerate the index in stage C",
    ("fuller", "src/docs_index.rs", 19): "root.join(\"docs/build_index.py\") -> the generator's new place, CARGO_MANIFEST_DIR/../../docs/build_index.py, run with current_dir the repository root (or the test moves to the workspace root)",
    ("phylu", "build.rs", 44): "{CARGO_MANIFEST_DIR}/../fuller -> {CARGO_MANIFEST_DIR}/../fuller still names crates/fuller, but the two crates are ONE repository: fuller's commit is phylu's commit; read the one HEAD once and drop the second commit",
    ("phylu", "examples/wgsl_true_image_fit.rs", 475): "{CARGO_MANIFEST_DIR}/../fuller -> unchanged (crates/phylu/../fuller = crates/fuller); FULLER_DIR keeps working",
}
for (repo, f, n), become in MANUAL.items():
    text = open(os.path.join(OLD[repo], f)).read().splitlines()[n - 1]
    rows[:] = [r for r in rows if r[:2] != (f"{repo}/{f}", str(n))]
    rows.append((f"{repo}/{f}", str(n), text.strip().replace("\t", " ")[:300], "layout-bound code", become, "rewrite by hand" if "unchanged" not in become else "none: still right"))

rows.sort()
text = "file\tline\ttext\tkind\tbecomes\taction\n" + "".join("\t".join(r) + "\n" for r in rows)
out = os.path.join(OLD["fuller"], "docs", "fuller_merge_inventory.tsv")
if sys.argv[1:] == ["--check"]:
    same = os.path.exists(out) and open(out).read() == text
    print(f"check: docs/fuller_merge_inventory.tsv {'equals' if same else 'DIFFERS FROM'} a fresh inventory ({len(rows)} rows)")
    sys.exit(0 if same else 1)
with open(out, "w") as f:
    f.write(text)
print(f"{len(rows)} rows -> {out}")
