#!/usr/bin/env python3
"""Run BLOCK_SR candidates through the REAL kernel: each candidate's `evolved_icdf(u: f32, span: f32) -> f32` replaces
the inverse-CDF table lookup in vblock_v4.wgsl (the truncation CDF stays a table lookup), and the layer is generated
on the GPU into <out>/<id>_layer<L>_s<seed>.f32 (204,288 x 64 f32 for layer 13, safetensors order).

    experiments/virtual_blocks/wgsl/run_candidates.sh candidates.wgsl.jsonl OUT [--layer 13] [--seed 1013] [--self-check]

--time also splices the candidate into the layer-13 ROW MATVEC kernel (row_gen_v4.wgsl) and reports its time per
dispatch (20 dispatches a submission, median of 20 submissions; the v4 tables measure 0.346 ms, the 4-bit read 0.06 ms).
--self-check first runs the hook with the ORIGINAL table lookup and requires the output to be byte-identical to
v4_layer<L>_s<seed>.f32 (the validated v4 file), so the splice itself is proven before any candidate is trusted.
Every candidate is checked as text before it reaches the GPU: one function with the agreed signature, no loops, no
storage, no barriers. A candidate that fails is reported with the reason and skipped. Writes <out>/results.jsonl:
{id, rank, file, seconds, finite, level_std_median} (level_std_median is the guard faxl-core uses: real is 2.96).
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(REPO, "target/release/examples/vblock_wgsl")
KERNEL = os.path.join(HERE, "vblock_v4.wgsl")
ROW_KERNEL = os.path.join(HERE, "row_gen_v4.wgsl")
TABLES = os.path.join(HERE, "tables_v3.f32")
CALL = "inv_cdf(k, "
FORBIDDEN = re.compile(r"\b(loop|for|while|tabs|params|out|cfg|red|ptr|workgroupBarrier|storageBarrier|subgroup\w*|"
                       r"texture\w*|atomic\w*|var\s*<|override|enable)\b|@")
SIGNATURE = re.compile(r"^\s*fn\s+evolved_icdf\s*\(\s*u\s*:\s*f32\s*,\s*span\s*:\s*f32\s*\)\s*->\s*f32\s*\{", re.S)


def check(wgsl):
    if not SIGNATURE.match(wgsl):
        return "the text does not start with fn evolved_icdf(u: f32, span: f32) -> f32 {"
    if wgsl.count("fn ") != 1:
        return f"the text defines {wgsl.count('fn ')} functions, wants exactly one"
    m = FORBIDDEN.search(wgsl)
    if m:
        return f"the text uses {m.group(0)!r}, which a candidate may not"
    return None


def spliced(hook_body, kernel=KERNEL):
    src = open(kernel).read()
    if src.count(CALL) != 2:
        raise SystemExit(f"{kernel}: expected 2 calls of {CALL!r}, found {src.count(CALL)}")
    src = src.replace(CALL, "icdf_hook(k, span, ")
    return src.replace("// one block: the two cells", hook_body + "\n// one block: the two cells", 1)


def run(kernel_text, params, seed, out_file, kfile):
    with open(kfile, "w") as f:
        f.write(kernel_text)
    t0 = time.time()
    p = subprocess.run([RUNNER, kfile, TABLES, params, str(seed), out_file], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["no stderr"])[-1]
    return time.time() - t0, None


def row_ms(kernel_text, params, seed, kfile):
    with open(kfile, "w") as f:
        f.write(kernel_text)
    p = subprocess.run([RUNNER, kfile, TABLES, params, str(seed), kfile + ".rows.f32", "20", "1", "10624"],
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, VBLOCK_INNER="20"))
    os.remove(kfile + ".rows.f32") if os.path.exists(kfile + ".rows.f32") else None
    m = re.search(r"median ([0-9.]+) ms", p.stdout)
    return float(m.group(1)) if m else None


def level_std_median(f, params):
    y = np.fromfile(f, np.float32).reshape(-1, 64).astype(np.float64)
    pr = np.fromfile(params, np.float32).reshape(-1, 4).astype(np.float64)
    q = (y - pr[:, 3:4]) / pr[:, 2:3]
    return bool(np.isfinite(y).all()), float(np.median(q.std(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates"); ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=13); ap.add_argument("--seed", type=int, default=1013)
    ap.add_argument("--self-check", action="store_true"); ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    params = os.path.join(HERE, f"params_layer{a.layer}.f32")
    if not os.path.exists(params):
        raise SystemExit(f"{params} missing: python3 {HERE}/layer_params.py {a.layer}")
    res = open(os.path.join(a.out, "results.jsonl"), "a")
    if a.self_check:
        ref = os.path.join(HERE, f"v4_layer{a.layer}_s{a.seed}.f32")
        f = os.path.join(a.out, f"selfcheck_layer{a.layer}_s{a.seed}.f32")
        hook = "fn icdf_hook(k: u32, span: f32, u: f32) -> f32 { return inv_cdf(k, u); }"
        sec, err = run(spliced(hook), params, a.seed, f, os.path.join(a.out, "selfcheck.wgsl"))
        same = err is None and open(f, "rb").read() == open(ref, "rb").read()
        print(f"self-check: {'IDENTICAL to ' + os.path.basename(ref) if same else 'FAILED ' + str(err)}", flush=True)
        if not same:
            sys.exit(1)
        os.remove(f)
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line)
        cid = str(c["id"])
        rec = {"id": cid, "rank": c.get("rank")}
        why = check(c["wgsl"])
        if why is None:
            hook = c["wgsl"].strip() + "\nfn icdf_hook(k: u32, span: f32, u: f32) -> f32 { return evolved_icdf(u, span); }"
            f = os.path.join(a.out, f"{cid}_layer{a.layer}_s{a.seed}.f32")
            sec, why = run(spliced(hook), params, a.seed, f, os.path.join(a.out, f"{cid}.wgsl"))
            if why is None:
                fin, lsm = level_std_median(f, params)
                rec.update(file=f, seconds=round(sec, 2), finite=fin, level_std_median=round(lsm, 4))
                if a.time:
                    if a.layer != 13:
                        raise SystemExit("--time: the row matvec kernel maps layer 13's rows only")
                    rec["row_ms"] = row_ms(spliced(hook, ROW_KERNEL), params, a.seed, os.path.join(a.out, f"{cid}.rows.wgsl"))
        if why is not None:
            rec["error"] = why
        print(json.dumps(rec), flush=True)
        res.write(json.dumps(rec) + "\n"); res.flush()


if __name__ == "__main__":
    main()
