//! THE POOL of real Futhark programs for a benchmark's signature kingdom (`docs/PLAN_futhark_fitness_fixes.md`
//! section H): the programs whose coding symbols fill the rings a canvas gene is cut from (`fuller::futhark_rings`).
//! Two sources, kept apart by their tag:
//!
//! * `bench`: every function of the published benchmarks (diku-dk/futhark-benchmarks), read into the kingdom by
//!   `fuller::futhark_reader`, in a fixed order (files sorted by path, functions in file order). Every function is
//!   recorded, read or a gap with the construct the kingdom lacks: the gaps measure the kingdom's coverage of real
//!   Futhark.
//! * `llm`: programs a language model writes for the benchmark's entry signature (the `claude` CLI, sonnet), from a
//!   contract that gives the signature, the kingdom's operations in Futhark spelling and a short description of the
//!   task, never the published program. Each call asks for several distinct programs, varied by a strategy phrase.
//!   Every prompt and reply is saved under the run directory; the calls are capped at 200 a benchmark.
//!
//! ```text
//! cargo run --release --example futhark_pool -- bench --problem histogram --benchmarks /private/tmp/claude-501/futhark-benchmarks \
//!     --out kingdoms/futhark_shapes/sig/histogram/pool --run /private/tmp/claude-501/fpool/histogram_bench
//! cargo run --release --example futhark_pool -- llm --problem histogram --calls 150 --parallel 6 \
//!     --out kingdoms/futhark_shapes/sig/histogram/pool --run /private/tmp/claude-501/fpool/histogram_llm
//! cargo run --release --example futhark_pool -- reread --problem histogram --out ... --run ...   (no new calls)
//! cargo run --release --example futhark_pool -- contract --problem histogram
//! ```
//!
//! Written to `--out`: `bench.tsv` and `llm.tsv` (the pool: `fuller::futhark_rings::load_pool`), `bench_functions.tsv`
//! and `llm_programs.tsv` (every function or program, read or a gap), `manifest.tsv` (the counts, the gaps by
//! construct, and the file excluded from the benchmark's own discovery runs: leave one out). The standard
//! (`docs/STANDARD_runners.md`): `RUN/stream.jsonl` (a snapshot a file or a batch of calls, its "fitness" 0 for a
//! function read and 1 for a gap), `RUN/killme.trigger` (checked before every file and call), a mutation event for
//! every program read (`model_rewrite` for a model's program, `note` for a benchmark's function and for a gap).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use fuller::futhark_kingdom::FutharkKingdom;
use fuller::futhark_reader::{Gap, Reader};
use fuller::futhark_rings::{published_source, Entry, Source, POOL_HEADER};
use fuller::futhark_sig::{load, problem, sexp, signature, Signature};
use fuller::telemetry::{EventKind, GenerationStats, Mutation, RunInfo, RunTelemetry};

/// The most model calls a benchmark may have (the cost cap of the brief).
const CALL_CAP: usize = 200;
const CLAUDE: &str = "/Users/andrewmorgan/.local/bin/claude";
const PROGRAMS_PER_CALL: usize = 5;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn utc() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The task, in a few sentences: what the program computes, never how the published program does it.
fn description(problem: &str) -> Result<&'static str, String> {
    match problem {
        "histogram" => Ok("Return k bins of i32. Element i of the inputs belongs to bin (is[i] mod k), where mod is Futhark's % (non-negative for positive k). Bin b holds the sum, with wrapping i32 arithmetic, of vs[i] over the elements i in bin b; an empty bin is 0."),
        "pathfinder" => Ok("walls is a grid of rows by cols cell costs. Return, for every column of the last row, the least total cost of a path from the top row down to it: the cost of row 0 is the row itself; for each later row i, the cost of column j is walls[i][j] plus the least of the previous row's costs at columns j - 1, j and j + 1, a neighbour off the edge of the grid left out. Arithmetic is wrapping i32."),
        "reduce" => Ok("Return the sum of the elements of xs, with wrapping i32 arithmetic."),
        "scan" => Ok("Return the inclusive prefix sums of xs: element i is the sum of xs[0] .. xs[i], with wrapping i32 arithmetic."),
        o => Err(format!("no task description for {o}")),
    }
}

/// The strategy phrases: one a call, in turn, so the calls ask for different programs.
const STRATEGIES: [&str; 30] = [
    "the most direct program you can write",
    "programs that use hist or reduce_by_index where they fit",
    "programs that avoid hist and reduce_by_index entirely",
    "programs built from tabulate with explicit indexing",
    "programs built from map and map2 over whole arrays",
    "programs that name every intermediate array with let",
    "programs that use helper functions defined with def",
    "programs that use a sequential loop with an array state",
    "programs that use a sequential loop with a scalar state inside tabulate",
    "programs that compute each output element independently",
    "programs that use i32.sum, i32.minimum or i32.maximum over a computed array",
    "programs that use i32.min and i32.max heavily",
    "deliberately naive but correct programs",
    "programs written as pipelines with |>",
    "programs that build the result up from replicate and map2",
    "programs that use the clamped index forms for neighbours",
    "programs with nested tabulate",
    "programs that use conditionals of the form if e < 0 then a else b",
    "programs that use reduce with an operator and its neutral element",
    "programs that use loops from 1 (for i in 1..<n)",
    "programs that differ from each other in their loop structure",
    "programs that use subtraction and negation in their arithmetic",
    "programs that use multiplication to mask values (multiply by 0 or 1)",
    "programs that use the sizes of the arrays as values",
    "programs a performance engineer would write",
    "programs a student would write on a first attempt that are still correct",
    "short programs of one or two lines",
    "long programs with many small steps",
    "programs that use map with a lambda of several let bindings",
    "programs that combine two different approaches in one",
];

/// THE TOPICS (Andrew, 2026-10-28): the rings must carry many unrelated real structures, so each program computes
/// its own topic on the benchmark's inputs, never the benchmark's task (which the model is not told).
const TOPICS: [&str; 60] = [
    "image blur (a box filter over neighbours)", "edge detection (differences of neighbours)", "a hash function mixing the elements", "a sorting network of compare-exchange steps",
    "a prefix sum", "a run-length encoding (the length of each run)", "a game of life step", "a 1-D convolution with a small kernel",
    "the maximum subarray sum (Kadane)", "a moving average", "polynomial evaluation (Horner)", "bit counting (population count by arithmetic)",
    "histogram equalisation", "a distance transform (distance to the nearest zero)", "an elementary cellular automaton step", "matrix transpose (or its 1-D analogue)",
    "a dot product", "clamping values into a range", "normalisation (subtract the minimum, scale)", "a breadth-first search frontier step",
    "a Mandelbrot-style iteration count", "a running maximum", "the second largest element", "counting inversions between neighbours",
    "a checksum (Adler or Fletcher style)", "a linear congruential random number generator", "the greatest common divisor by subtraction", "integer square root by iteration",
    "a Fibonacci sequence", "a Collatz step count", "a triangular number table", "a pairwise difference (discrete derivative)",
    "a discrete integral with a decay", "a stencil of a heat equation step", "thresholding into 0 and 1", "a majority vote of neighbours",
    "the longest increasing run", "an argmax (index of the largest)", "a rank (how many elements are smaller)", "a reversal by index arithmetic",
    "a cyclic shift by index arithmetic", "a cumulative product with wrapping", "a min-plus matrix-vector step", "a knapsack-style dynamic programming row",
    "an edit-distance-style dynamic programming row", "a longest-common-subsequence-style dynamic programming row", "a parity bit of each element", "a saturating add",
    "an absolute deviation from the mean", "a variance (sum of squared deviations)", "a weighted sum with positional weights", "a sawtooth or triangle wave by modulo",
    "a bucket count of values modulo a size", "a sieve-like marking of multiples", "a binary counter's carry propagation", "a sum of absolute differences between two arrays",
    "a peak detector (greater than both neighbours)", "a flood fill step (spread the minimum to neighbours)", "a two-player score table (pairwise max minus min)", "a mirrored padding by clamped indices",
];

/// THE ANGLES: other problem statements of the same task, for alternatives genuinely different from those before.
const ANGLES: [&str; 20] = [
    "compute every output element independently with tabulate, no histogram operation",
    "a sequential loop over the input carrying the whole output array",
    "a loop over the output positions with a reduction over the input inside",
    "an algorithm built from map2 over whole arrays, no explicit indexing",
    "an arithmetic masking formulation: multiply by 0 or 1 instead of choosing",
    "a formulation with conditionals of the form if e < 0 then a else b",
    "a two-pass algorithm: first a transformed array, then the combination",
    "a formulation with a loop from 1 over a dimension",
    "a formulation using scalar loops nested inside tabulate",
    "a formulation that reuses one intermediate array several times",
    "a divide-and-combine formulation with partial results combined by map2",
    "a formulation using i32.min and i32.max in an essential way",
    "a formulation that counts first and then scales",
    "a formulation using the sizes of the arrays as values",
    "a formulation that transforms the indices before use",
    "a formulation using subtraction and negation instead of addition where possible",
    "a deliberately unusual but correct algorithm",
    "the simplest correct program written in a style unlike the obvious one",
    "a formulation with the loop structure reversed (outer and inner swapped)",
    "a formulation that builds the answer element by element in a loop with an index-dependent update",
];

const TOPIC_RULE: &str = "THE TOPIC is given in each request. Your program computes that topic, its own result, on the entry's inputs, read as data of that topic in any sensible way, with a result of the entry's output type. It does not compute anything else.";

/// THE CONTRACT: the entry signature, every operation of the kingdom's top context in Futhark spelling (a
/// placeholder `<class>` for each argument), the rules a program the kingdom can read must keep, and the task.
fn contract(k: &FutharkKingdom, sig: &Signature, pname: &str, topics: bool) -> Result<String, String> {
    let dims = |d: &[String]| d.iter().map(|x| if sig.size_inputs.contains(x) { format!("[i64.i32 {x}]") } else { format!("[{x}]") }).collect::<String>();
    let mut entry = format!("entry main {}", sig.dims.iter().filter(|d| !sig.size_inputs.contains(d)).map(|d| format!("[{d}]")).collect::<String>());
    for p in &sig.inputs {
        entry.push_str(&format!(" ({}: {}{})", p.name, dims(&p.dims), p.elem));
    }
    entry.push_str(&format!(" : {}{} =", dims(&sig.output.dims), sig.output.elem));
    let notes = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("kingdoms/futhark_shapes/sig/{pname}/functions.tsv"))).map_err(|e| e.to_string())?;
    let note_of: BTreeMap<&str, &str> = notes.lines().skip(1).filter_map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f.len() >= 7).then(|| (f[1], f[6]))
    }).collect();
    let mut ops = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for r in &k.rows {
        let Some(d) = k.dual(r.output) else { continue };
        let top = !d.dual.contains(':') || d.dual.ends_with(":c");
        if !top || r.kind == fuller::futhark_kingdom::FutharkKind::Const {
            continue;
        }
        // `zeros_D` is bound by the kingdom's own entry point: to a program it is `replicate D 0`.
        let mut s = match r.spelling.strip_prefix("zeros_") {
            Some(d) => format!("replicate {d} 0"),
            None => r.spelling.clone(),
        };
        for (i, t) in r.inputs.iter().enumerate() {
            let cls = k.dual(*t).map(|d| d.dual.split(':').next().unwrap_or("").to_string()).unwrap_or_default();
            s = s.replace(&format!("{{{i}}}"), &format!("<{cls}>"));
        }
        let cls = d.dual.split(':').next().unwrap_or("");
        if seen.insert(s.clone()) {
            ops.push(format!("  {s}    -- <{cls}>: {}", note_of.get(r.name.as_str()).copied().unwrap_or("")));
        }
    }
    let lits = k.duals.iter().find(|d| d.dual == "i32:c").map(|d| d.literals.iter().map(|l| l.trim_end_matches("i32").to_string()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    let sizes = if sig.size_inputs.is_empty() {
        String::new()
    } else {
        format!(" In these spellings {} is the size as an i64: begin the body with {} as the kingdom's own programs do.", sig.size_inputs.join(", "), sig.size_inputs.iter().map(|d| format!("let {d} = i64.i32 {d}")).collect::<Vec<_>>().join(" and "))
    };
    let classes = sig.dims.iter().map(|d| format!("<a.{d}> is a [{d}]i32 array, <ix.{d}> an i64 index in range for [{d}]")).collect::<Vec<_>>().join("; ");
    Ok(format!(
        "You write programs in Futhark (version 0.28). Every program you write must be a complete Futhark program with exactly this entry point:\n\n  {entry}\n\n{}\n\nYOUR PROGRAMS ARE READ INTO A RESTRICTED KINGDOM OF FUTHARK, which has only these operations (Futhark spelling; <i32> is an i32 value, {classes}; inside a binder the binder's own variables, under any names you choose, are values of those classes).{sizes}\n{}\n\nRULES a program must keep, or it cannot be read:\n- i32 values only: no floats, no bool values other than the condition below, no tuples, no records, no array literals, no in-place updates (with), no while loops, no filter, scatter, zip, rotate, reverse, iota-based arithmetic on i64 values.\n- Integer literals only {lits}.\n- A conditional only as: if e < 0 then a else b.\n- An index into an array must be in range by construction: a loop or tabulate counter over that array's size, 0, the size minus 1, i64.max 0 (j - 1), i64.min (D - 1) (j + 1), or i64.max 0 (i64.min (D - 1) (i64.i32 e)).\n- At most {} nested binders (tabulate, loop, or the lambda of a map).\n- You may use let bindings, def helper functions, map and map2 with lambdas, reduce (+) 0, reduce i32.min i32.highest, reduce i32.max i32.lowest, and |>.\n\n{}",
        if topics { TOPIC_RULE.to_string() } else { format!("THE TASK. {}", description(pname)?) },
        ops.join("\n"),
        sig.depth,
        if topics { "Answer with ONE correct program on the topic you are given, complete, in one ```futhark code block, and nothing else.".to_string() } else { format!("Answer with {PROGRAMS_PER_CALL} DISTINCT correct programs, each complete, each in its own ```futhark code block, and nothing else.") },
    ))
}

/// Every ```futhark (or ```fut, or bare ```) code block of a reply.
fn code_blocks(reply: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for line in reply.lines() {
        let t = line.trim_start();
        if t.starts_with("```") {
            match cur.take() {
                Some(b) => out.push(b),
                None => cur = Some(String::new()),
            }
            continue;
        }
        if let Some(b) = cur.as_mut() {
            b.push_str(line);
            b.push('\n');
        }
    }
    out
}

struct Ctx {
    /// Each call's problem statement takes its own angle (`ANGLES`).
    angles: bool,
    /// The programs are on topics of their own, never the benchmark's task (`topics`).
    topics: bool,
    pname: String,
    k: FutharkKingdom,
    sig: Signature,
    out: PathBuf,
    run: PathBuf,
}

fn gap_row(g: &Gap) -> String {
    format!("{}\t{}\t{}", g.construct, g.dual, g.detail.replace(['\t', '\n'], " "))
}

/// The manifest: key, value lines (merged with what the other source wrote).
fn update_manifest(out: &Path, prefix: &str, lines: Vec<(String, String)>) -> Result<(), String> {
    let path = out.join("manifest.tsv");
    let old = std::fs::read_to_string(&path).unwrap_or_default();
    let mut kept: Vec<String> = old.lines().skip(1).filter(|l| !l.starts_with(prefix)).map(str::to_string).collect();
    kept.extend(lines.into_iter().map(|(k, v)| format!("{k}\t{v}")));
    kept.sort();
    write(&path, &format!("key\tvalue\n{}\n", kept.join("\n")))
}

fn gap_counts(gaps: &[Gap]) -> Vec<(String, usize)> {
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for g in gaps {
        *m.entry(g.construct.clone()).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

fn bench(c: &Ctx, root: &Path) -> Result<(), String> {
    let reader = Reader::for_pool(&c.k, &c.sig)?;
    let mut files = Vec::new();
    fn walk(d: &Path, root: &Path, out: &mut Vec<String>) -> Result<(), String> {
        for e in std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.is_dir() {
                walk(&p, root, out)?;
            } else if p.extension().is_some_and(|x| x == "fut") {
                out.push(p.strip_prefix(root).map_err(|e| e.to_string())?.to_string_lossy().to_string());
            }
        }
        Ok(())
    }
    walk(root, root, &mut files)?;
    files.sort();
    let commit = std::process::Command::new("git").arg("-C").arg(root).args(["rev-parse", "HEAD"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let mut tel = RunTelemetry::start(
        &c.run,
        format!("futhark-pool-bench-{}", c.pname),
        RunInfo { kingdom: "futhark_sig".into(), dataset: format!("{}: futhark-benchmarks {commit}, {} files", c.pname, files.len()), seed: 0, population: 1, max_generations: files.len() as u32, budget_ms: 0, hff_objectives: vec!["read (0) or a gap of the kingdom (1), a function each".into()], config_json: None, resume: false, already_spent_seconds: 0.0 },
    )?;
    let mut pool = vec![POOL_HEADER.to_string()];
    let mut all = vec!["file\tfunction\tstatus\tdual\tsexp_or_construct\tdual_of_gap\tdetail".to_string()];
    let (mut gaps, mut read, mut functions, mut files_read) = (Vec::new(), 0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    let mut stopped = "early_stop";
    for (g, f) in files.iter().enumerate() {
        if tel.stop_requested() {
            stopped = "killed";
            break;
        }
        let src = std::fs::read_to_string(root.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let res = reader.read_functions(&src);
        let mut fit = Vec::new();
        let mut any = false;
        for (name, r) in res {
            functions += 1;
            match r {
                Ok((ty, tree)) => {
                    let dual = c.k.dual(ty).map(|d| d.dual.clone()).unwrap_or_default();
                    let s = sexp(&c.k, &tree);
                    let e = Entry { source: Source::Bench, origin: f.clone(), function: name.clone(), dual: dual.clone(), sexp: s.clone() };
                    pool.push(e.line(pool.len()));
                    all.push(format!("{f}\t{name}\tread\t{dual}\t{s}\t\t"));
                    let mut m = Mutation::new(EventKind::Note, "pool_read_benchmark", format!("{f} {name}: read at {dual}, {} nodes", tree.level_order().len()));
                    m.text_after = Some(s);
                    m.nodes = Some(tree.level_order().len() as u32);
                    tel.mutation(g as u32, m)?;
                    read += 1;
                    any = true;
                    fit.push(0.0);
                }
                Err(gap) => {
                    all.push(format!("{f}\t{name}\tgap\t\t{}", gap_row(&gap)));
                    let mut m = Mutation::new(EventKind::Note, "pool_gap", format!("{f} {name}: a gap of the kingdom: {gap}"));
                    m.entered = Some(false);
                    tel.mutation(g as u32, m)?;
                    gaps.push(gap);
                    fit.push(1.0);
                }
            }
        }
        files_read += usize::from(any);
        if fit.is_empty() {
            fit.push(1.0);
        }
        let mut extras = BTreeMap::new();
        extras.insert("functions_read".to_string(), read as f64);
        extras.insert("functions".to_string(), functions as f64);
        tel.generation(GenerationStats { generation: g as u32, fitness: &fit, distinct: None, angle_groups: BTreeMap::new(), extras, evaluated: fit.len() as u64 })?;
    }
    write(&c.out.join("bench.tsv"), &(pool.join("\n") + "\n"))?;
    write(&c.out.join("bench_functions.tsv"), &(all.join("\n") + "\n"))?;
    let mut m = vec![
        ("bench.benchmarks_commit".to_string(), commit),
        ("bench.files".into(), files.len().to_string()),
        ("bench.files_with_a_function_read".into(), files_read.to_string()),
        ("bench.functions".into(), functions.to_string()),
        ("bench.functions_read".into(), read.to_string()),
        ("bench.functions_gap".into(), gaps.len().to_string()),
        ("bench.excluded_for_discovery".into(), format!("{} (leave one out: the benchmark's own published program and file are excluded from the rings of its R and C runs; P may include them)", published_source(&c.pname).unwrap_or("-"))),
    ];
    for (construct, n) in gap_counts(&gaps) {
        m.push((format!("bench.gap {construct}"), n.to_string()));
    }
    update_manifest(&c.out, "bench.", m)?;
    tel.end(stopped, files.len() as u32, None)?;
    println!("# {} {}: {} files, {functions} functions, {read} read ({files_read} files with one read), {} gaps; {:.1} s", utc(), c.pname, files.len(), gaps.len(), t0.elapsed().as_secs_f64());
    for (construct, n) in gap_counts(&gaps).into_iter().take(15) {
        println!("  gap {n:5}  {construct}");
    }
    Ok(())
}

/// The calls already made (their replies saved), by number.
fn calls_made(dir: &Path) -> Vec<usize> {
    let mut v: Vec<usize> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok()).filter_map(|e| e.file_name().to_string_lossy().strip_prefix("call_").and_then(|s| s.strip_suffix(".reply.json")).and_then(|n| n.parse().ok())).collect()).unwrap_or_default();
    v.sort_unstable();
    v
}

fn one_call(system: &str, prompt: &str, dir: &Path, i: usize) -> Result<(), String> {
    write(&dir.join(format!("call_{i:03}.prompt.txt")), prompt)?;
    let out = fuller::futhark_exec::output_with_timeout(
        // No tools, no MCP servers, and the run's own directory as the working directory: the model sees the contract
        // and the prompt only, never this repository (which holds the published program).
        std::process::Command::new(CLAUDE).current_dir(dir).args(["-p", "--model", "sonnet", "--output-format", "json", "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--system-prompt", system, prompt]),
        std::time::Duration::from_secs(900),
        None,
    )?;
    if !out.status.success() {
        return Err(format!("call {i}: claude failed: {}\n{}", String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout)));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("call {i}: the reply is not JSON: {e}"))?;
    let r = result_of(&v).ok_or_else(|| format!("call {i}: the reply has no result"))?;
    if r["is_error"].as_bool() == Some(true) {
        return Err(format!("call {i}: claude reported an error: {}", r["result"]));
    }
    write(&dir.join(format!("call_{i:03}.reply.txt")), r["result"].as_str().unwrap_or(""))?;
    write(&dir.join(format!("call_{i:03}.reply.json")), &String::from_utf8_lossy(&out.stdout))
}

/// The `result` event of the CLI's JSON output (an array of events, or the one result object).
fn result_of(v: &serde_json::Value) -> Option<&serde_json::Value> {
    match v {
        serde_json::Value::Array(a) => a.iter().find(|e| e["type"] == "result"),
        o if o["type"] == "result" => Some(o),
        _ => None,
    }
}

/// Read every saved reply into `llm.tsv` and `llm_programs.tsv`, in call order.
fn read_replies(c: &Ctx, tel: &mut RunTelemetry, generation: u32) -> Result<(usize, usize, usize), String> {
    let reader = Reader::for_pool(&c.k, &c.sig)?;
    let dir = c.run.join("llm");
    let mut pool = vec![POOL_HEADER.to_string()];
    let mut all = vec!["reply\tprogram\tstatus\tdual\tsexp_or_construct\tdual_of_gap\tdetail".to_string()];
    let (mut gaps, mut read, mut programs) = (Vec::new(), 0, 0);
    let root = c.k.roots()[0];
    let calls = calls_made(&dir);
    let mut cost = 0.0;
    for i in &calls {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("call_{i:03}.reply.json"))).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let r = result_of(&j).ok_or_else(|| format!("call {i}: the saved reply has no result"))?;
        cost += r["total_cost_usd"].as_f64().unwrap_or(0.0);
        let reply = r["result"].as_str().unwrap_or("").to_string();
        for (b, src) in code_blocks(&reply).iter().enumerate() {
            programs += 1;
            let topic = std::fs::read_to_string(dir.join(format!("call_{i:03}.topic.txt"))).ok();
            let origin = match &topic {
                Some(t) => format!("call_{i:03}#{b} topic: {}", t.replace(['\t', '\n'], " ")),
                None => format!("call_{i:03}#{b}"),
            };
            match reader.read_program(src) {
                Ok(t) => {
                    let s = sexp(&c.k, &t);
                    let dual = c.k.dual(root).map(|d| d.dual.clone()).unwrap_or_default();
                    pool.push(Entry { source: Source::Llm, origin: origin.clone(), function: "main".into(), dual: dual.clone(), sexp: s.clone() }.line(pool.len()));
                    all.push(format!("{origin}\t{b}\tread\t{dual}\t{s}\t\t"));
                    let mut m = Mutation::new(EventKind::ModelRewrite, "pool_read_llm", format!("{origin}: read, {} nodes", t.level_order().len()));
                    m.text_before = Some(src.clone());
                    m.text_after = Some(s);
                    m.nodes = Some(t.level_order().len() as u32);
                    tel.mutation(generation, m)?;
                    read += 1;
                }
                Err(g) => {
                    all.push(format!("{origin}\t{b}\tgap\t\t{}", gap_row(&g)));
                    let mut m = Mutation::new(EventKind::Note, "pool_gap", format!("{origin}: a gap of the kingdom: {g}"));
                    m.text_before = Some(src.clone());
                    m.entered = Some(false);
                    tel.mutation(generation, m)?;
                    gaps.push(g);
                }
            }
        }
    }
    // The topic programs are the canvas's rings (`llm.tsv`); the task programs are the population's alternatives
    // (`alternatives.tsv`, Andrew's clarification of 2026-10-29), never in the rings.
    let (file, every, px) = if c.topics { ("llm.tsv", "llm_programs.tsv", "llm") } else { ("alternatives.tsv", "alternatives_programs.tsv", "alt") };
    write(&c.out.join(file), &(pool.join("\n") + "\n"))?;
    write(&c.out.join(every), &(all.join("\n") + "\n"))?;
    let mut m = vec![
        (format!("{px}.model"), "claude sonnet (claude -p --model sonnet), the contract in llm/contract.txt of the run".to_string()),
        (format!("{px}.calls"), calls.len().to_string()),
        (format!("{px}.cost_usd"), format!("{cost:.2}")),
        (format!("{px}.programs"), programs.to_string()),
        (format!("{px}.programs_read"), read.to_string()),
        (format!("{px}.programs_gap"), gaps.len().to_string()),
        (format!("{px}.run"), c.run.display().to_string()),
    ];
    for (construct, n) in gap_counts(&gaps) {
        m.push((format!("{px}.gap {construct}"), n.to_string()));
    }
    update_manifest(&c.out, &format!("{px}."), m)?;
    println!("# {} {}: {} calls (${cost:.2}), {programs} programs, {read} read, {} gaps", utc(), c.pname, calls.len(), gaps.len());
    for (construct, n) in gap_counts(&gaps).into_iter().take(15) {
        println!("  gap {n:5}  {construct}");
    }
    Ok((calls.len(), programs, read))
}

fn llm(c: &Ctx, calls: usize, parallel: usize, new_calls: bool) -> Result<(), String> {
    let dir = c.run.join("llm");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let system = contract(&c.k, &c.sig, &c.pname, c.topics)?;
    if c.topics && calls > TOPICS.len() {
        return Err(format!("--calls {calls}: there are {} topics", TOPICS.len()));
    }
    write(&dir.join("contract.txt"), &system)?;
    let mut tel = RunTelemetry::start(
        &c.run,
        format!("futhark-pool-llm-{}", c.pname),
        RunInfo { kingdom: "futhark_sig".into(), dataset: format!("{}: model-written programs", c.pname), seed: 0, population: parallel as u32, max_generations: calls as u32, budget_ms: 0, hff_objectives: vec!["a call answered (0) or failed (1)".into()], config_json: Some(serde_json::json!({"calls": calls, "parallel": parallel, "cap": CALL_CAP}).to_string()), resume: false, already_spent_seconds: 0.0 },
    )?;
    let mut stopped = "early_stop";
    if new_calls {
        let made = calls_made(&dir);
        if calls > CALL_CAP {
            return Err(format!("--calls {calls}: the cap is {CALL_CAP} a benchmark"));
        }
        let todo: Vec<usize> = (0..calls).filter(|i| !made.contains(i)).collect();
        println!("# {} {}: {} calls made, {} to make, {parallel} at a time", utc(), c.pname, made.len(), todo.len());
        let next = AtomicUsize::new(0);
        let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let tel_m = Mutex::new(&mut tel);
        let done = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..parallel.max(1) {
                s.spawn(|| loop {
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&i) = todo.get(n) else { break };
                    if tel_m.lock().map(|t| t.stop_requested()).unwrap_or(true) {
                        break;
                    }
                    let prompt = if c.topics {
                        let _ = write(&dir.join(format!("call_{i:03}.topic.txt")), TOPICS[i]);
                        format!("Write one program on the topic: {}.", TOPICS[i])
                    } else {
                        format!("Write {PROGRAMS_PER_CALL} distinct programs. Strategy for this batch: {}. (Batch {i}.)", STRATEGIES[i % STRATEGIES.len()])
                    };
                    // `--angles`: the problem statement itself changes from call to call (a different algorithm family,
                    // data layout or loop structure), recorded beside the call (Andrew, 2026-10-29: when the
                    // alternatives are not different enough, change the statement, never re-ask the same prompt).
                    let system_i = if c.angles {
                        let a = ANGLES[i % ANGLES.len()];
                        let _ = write(&dir.join(format!("call_{i:03}.statement.txt")), a);
                        format!("{system}\n\nTHE ANGLE of this request, which your programs must take: {a}")
                    } else {
                        system.clone()
                    };
                    let r = one_call(&system_i, &prompt, &dir, i);
                    let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                    let fit = [if r.is_ok() { 0.0 } else { 1.0 }];
                    if let Ok(mut t) = tel_m.lock() {
                        let _ = t.generation(GenerationStats { generation: i as u32, fitness: &fit, distinct: None, angle_groups: BTreeMap::new(), extras: BTreeMap::new(), evaluated: 1 });
                    }
                    match r {
                        Ok(()) => println!("# {} call {i} answered ({d} of {})", utc(), todo.len()),
                        Err(e) => {
                            eprintln!("# {} {e}", utc());
                            if let Ok(mut v) = errors.lock() {
                                v.push(e);
                            }
                        }
                    }
                });
            }
        });
        let errors = errors.into_inner().map_err(|e| e.to_string())?;
        if tel.stop_requested() {
            stopped = "killed";
        }
        if !errors.is_empty() {
            println!("# {} {} calls failed (saved nothing): {}", utc(), errors.len(), errors.join(" | "));
        }
    }
    read_replies(c, &mut tel, calls as u32)?;
    tel.end(stopped, calls as u32, None)
}

/// How many task alternatives (`alternatives.tsv`) pass the population's distance bar, greedily in order: a tree edit
/// distance of at least `bar` to the reference and to every alternative kept (the output bar is the runner's, on the
/// GPU). With the distribution of the distances to the reference, to choose the bar from the data.
fn distinct(c: &Ctx, bar: usize) -> Result<(), String> {
    let p = problem(&c.pname)?;
    let reference = fuller::futhark_sig::build(&c.k, p.reference)?;
    let alts = fuller::futhark_rings::load_pool(&c.out.join("alternatives.tsv"))?;
    let trees: Vec<fuller::typed_gep::Tree> = alts.iter().map(|e| fuller::futhark_rings::build_at(&c.k, &e.sexp, &e.dual)).collect::<Result<_, _>>()?;
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut kept: Vec<&fuller::typed_gep::Tree> = Vec::new();
    let (mut near_ref, mut near_other) = (0, 0);
    for t in &trees {
        let d = fuller::futhark_sig::tree_distance(t, &reference);
        *hist.entry(d).or_default() += 1;
        if d < bar {
            near_ref += 1;
        } else if kept.iter().any(|o| fuller::futhark_sig::tree_distance(t, o) < bar) {
            near_other += 1;
        } else {
            kept.push(t);
        }
    }
    println!("{}: {} alternatives; bar {bar}: {} kept, {near_ref} near the reference, {near_other} near another kept", c.pname, trees.len(), kept.len());
    println!("distance to the reference: {}", hist.iter().map(|(d, n)| format!("{d}:{n}")).collect::<Vec<_>>().join(" "));
    Ok(())
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().ok_or("futhark_pool bench|llm|reread|contract --problem P ...")?;
    let pname = arg(&args, "--problem").ok_or("--problem P")?;
    let p = problem(&pname)?;
    let k = load(&p.instance)?;
    let sig = signature(&p.instance)?;
    if cmd == "contract" {
        println!("{}", contract(&k, &sig, &pname, args.iter().any(|a| a == "--topics"))?);
        return Ok(());
    }
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out POOLDIR")?);
    let run = PathBuf::from(arg(&args, "--run").ok_or("--run RUNDIR")?);
    for d in [&out, &run] {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let c = Ctx { angles: args.iter().any(|a| a == "--angles"), topics: cmd == "topics" || (cmd == "reread" && args.iter().any(|a| a == "--topics")), pname, k, sig, out, run };
    match cmd.as_str() {
        "bench" => bench(&c, &PathBuf::from(arg(&args, "--benchmarks").ok_or("--benchmarks DIR")?)),
        "llm" => {
            let calls: usize = arg(&args, "--calls").ok_or("--calls N")?.parse().map_err(|_| "--calls N")?;
            let parallel: usize = arg(&args, "--parallel").map_or(Ok(4), |v| v.parse()).map_err(|_| "--parallel N")?;
            llm(&c, calls, parallel, true)
        }
        "topics" => {
            let parallel: usize = arg(&args, "--parallel").map_or(Ok(4), |v| v.parse()).map_err(|_| "--parallel N")?;
            llm(&c, TOPICS.len(), parallel, true)
        }
        "reread" => {
            let calls = calls_made(&c.run.join("llm")).len();
            llm(&c, calls, 1, false)
        }
        "distinct" => {
            let bar: usize = arg(&args, "--bar").map_or(Ok(5), |v| v.parse()).map_err(|_| "--bar N")?;
            distinct(&c, bar)
        }
        o => Err(format!("{o}: bench, llm, topics, reread, distinct or contract")),
    }
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
