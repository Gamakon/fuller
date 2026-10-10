//! Hallucino-genetic programming, generation 0 (`docs/PLAN_hgp_gram.md`,
//! the smallest experiment, step 1): `--n` isolated calls to Sonnet through
//! the Claude Code CLI in bare mode with no tools, each asked for a complete
//! Gram kernel under the role's contract and one strategy; every prompt and
//! reply saved verbatim, the kernel text extracted (fences stripped) and
//! written as `k<NN>.wgsl` for the ladder (`wgsl_gram_ladder`, one process
//! per kernel). Nothing here judges a kernel.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_hgp -- gen0 [--n 32] [--seed 1] [--out logs/HGP/001/gen0] [--model sonnet]

use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use fuller::wgsl::gram::SHIPPED;

fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

/// The strategies, one per call in turn; the diversity source, since the
/// CLI exposes no sampling temperature.
pub const STRATEGIES: [&str; 10] = [
    "one thread per output cell, as the reference kernel does, but with any improvement you see",
    "register blocking: each thread computes a 2 x 2 block of H, reading each phi value once per block row and column",
    "register blocking: each thread computes a 4 x 4 block of H with vec4 loads of four consecutive columns",
    "a workgroup tile: stage a tile of items' phi values (the block's rows and columns) in var<workgroup> memory with workgroupBarrier, then accumulate from shared memory",
    "vector loads: read four consecutive columns of phi as a vec4 and keep vec4 accumulators",
    "split accumulators: several independent partial sums over interleaved items, added at the end",
    "an unrolled item loop: four or eight items per iteration with the remainder handled after",
    "combine a workgroup tile of items in shared memory with a 4 x 4 register block per thread",
    "combine register blocking with vec4 loads and an unrolled item loop",
    "a structure unlike the reference kernel: choose any correct organisation of the work you believe is fastest on an Apple M3 Max GPU under Metal",
];

fn contract() -> String {
    format!(
        "You write WGSL compute kernels for wgpu on an Apple M3 Max (Metal, naga 0.20 front end; WGSL as of 2024, no f16, no subgroup operations, no atomics on f32).\n\n\
TASK. Write a compute kernel that computes the weighted Gram matrix H = Phi^T Phi over a band of items, accumulating into a resident output band, exactly as the reference kernel below does, but faster.\n\n\
CONTRACT (fixed, the harness binds exactly these):\n\
```wgsl\nstruct Cfg {{ d: u32, m: u32, r_rows: u32, r_off: u32, }};\n@group(0) @binding(0) var<storage, read> phi: array<f32>;         // m x d, item-major: item t's d values are phi[t*d .. t*d+d]\n@group(0) @binding(1) var<storage, read_write> h: array<f32>;     // r_rows x d, the output-row band: h[lr*d + c] for local row lr in [0, r_rows), column c in [0, d)\n@group(0) @binding(2) var<uniform> cfg: Cfg;\n```\n\
SEMANTICS. For every local row lr < r_rows and column c < d, with the global row gr = r_off + lr: h[lr*d + c] += sum over t in [0, m) of phi[t*d + gr] * phi[t*d + c]. The driver dispatches the kernel once per item band; h already holds earlier bands' partial sums, so the kernel must READ-ADD-WRITE h, never overwrite it. Rows outside [0, r_rows) and columns outside [0, d) must not be written. gr can exceed d - 1 when r_off + r_rows > d: such rows must not be read or written. A cell may be written by exactly one thread. d and m are arbitrary (not multiples of anything); r_rows <= d.\n\n\
ENTRY POINT. `@compute @workgroup_size(X, Y, 1) fn main(...)` with any builtins you need. The FIRST LINE of your kernel must be the comment `// grid X Y BX BY` where X, Y are your workgroup size and BX, BY are the cells per thread along columns and rows (1 if one cell per thread). The harness dispatches ceil(d / (X*BX)) x ceil(r_rows / (Y*BY)) x 1 workgroups. Use workgroup_id, local_invocation_id or global_invocation_id consistently with that.\n\n\
CONSTRAINTS. Only the three bindings above. No other global variables except var<workgroup> arrays (at most 16 KiB in total). workgroupBarrier() only in uniform control flow (never after a conditional return; guard with if-blocks instead). Every loop must terminate: bound item loops by cfg.m, tile loops by a literal. f32 arithmetic only; the result must match the reference to within f32 rounding (reordering the sum is fine).\n\n\
REFERENCE KERNEL (correct, one thread per cell, the one to beat):\n```wgsl\n{SHIPPED}```\n\n\
OUTPUT. Return ONLY the complete WGSL kernel text, starting with the `// grid` line, with no explanation, no markdown fences, no prose before or after.")
}

fn task(strategy: &str, seed_phrase: &str) -> String {
    format!("Write the kernel. Strategy to use: {strategy}. Variation key: {seed_phrase}. Remember: first line `// grid X Y BX BY`, only the kernel text.")
}

fn strip(reply: &str) -> String {
    // A fenced block if present, else the text from the first `// grid` line.
    let mut text = reply.trim().to_string();
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        let end = body.find("```").unwrap_or(body.len());
        text = body[..end].to_string();
    }
    if let Some(i) = text.find("// grid") {
        text = text[i..].to_string();
    }
    text.trim().to_string() + "\n"
}

fn call(model: &str, system: &str, prompt: &str) -> Result<(String, String, f64), String> {
    let t = Instant::now();
    let mut child = Command::new("claude")
        .args(["-p", "--bare", "--model", model, "--output-format", "json", "--no-session-persistence", "--tools", "", "--system-prompt", system])
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("claude: {e}"))?;
    child.stdin.take().ok_or("stdin")?.write_all(prompt.as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.status.success() {
        return Err(format!("claude exited {}: {}", out.status, String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or("")));
    }
    // --output-format json: one JSON object (or an array of events) with a `result` string.
    let json: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| format!("json: {e}: {}", &stdout[..stdout.len().min(200)]))?;
    let result = json.get("result").and_then(|r| r.as_str()).map(str::to_string).or_else(|| json.as_array().and_then(|a| a.iter().find_map(|v| v.get("result").and_then(|r| r.as_str()).map(str::to_string)))).ok_or_else(|| format!("no result field: {}", &stdout[..stdout.len().min(300)]))?;
    Ok((result, stdout, t.elapsed().as_secs_f64()))
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let sub = args.next().ok_or("usage: wgsl_gram_hgp gen0 [--n 32] [--seed 1] [--out dir] [--model sonnet]")?;
    if sub != "gen0" {
        return Err(format!("unknown subcommand {sub}"));
    }
    let (mut n, mut seed, mut out, mut model) = (32usize, 1u64, PathBuf::from("logs/HGP/001/gen0"), "sonnet".to_string());
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--n" => n = value("--n")?.parse().map_err(|e| format!("--n: {e}"))?,
            "--seed" => seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--out" => out = PathBuf::from(value("--out")?),
            "--model" => model = value("--model")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    std::fs::create_dir_all(out.join("calls")).map_err(|e| e.to_string())?;
    let system = contract();
    std::fs::write(out.join("contract.txt"), &system).map_err(|e| e.to_string())?;
    println!("[{}] gen0: {n} calls to {model}, seed {seed}, out {}", stamp(), out.display());
    let mut index = std::fs::File::create(out.join("index.tsv")).map_err(|e| e.to_string())?;
    writeln!(index, "id\tstrategy\tseconds\tbytes\thas_grid_header\tfile").map_err(|e| e.to_string())?;
    let t_all = Instant::now();
    for i in 0..n {
        let strategy = STRATEGIES[i % STRATEGIES.len()];
        let seed_phrase = format!("s{seed}-k{i}-{:x}", (seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64)).rotate_left(17));
        let prompt = task(strategy, &seed_phrase);
        std::fs::write(out.join(format!("calls/k{i:02}.prompt.txt")), format!("[system]\n{system}\n\n[user]\n{prompt}\n")).map_err(|e| e.to_string())?;
        println!("[{}] k{i:02} will call: strategy {}: {}", stamp(), i % STRATEGIES.len(), &strategy[..strategy.len().min(70)]);
        match call(&model, &system, &prompt) {
            Ok((reply, raw, secs)) => {
                std::fs::write(out.join(format!("calls/k{i:02}.reply.json")), &raw).map_err(|e| e.to_string())?;
                std::fs::write(out.join(format!("calls/k{i:02}.reply.txt")), &reply).map_err(|e| e.to_string())?;
                let kernel = strip(&reply);
                let file = out.join(format!("k{i:02}.wgsl"));
                std::fs::write(&file, &kernel).map_err(|e| e.to_string())?;
                let has_header = kernel.starts_with("// grid ");
                println!("[{}] k{i:02} replied in {secs:.1} s: {} bytes, grid header {}", stamp(), kernel.len(), has_header);
                writeln!(index, "k{i:02}\t{}\t{secs:.1}\t{}\t{has_header}\t{}", i % STRATEGIES.len(), kernel.len(), file.display()).map_err(|e| e.to_string())?;
            }
            Err(e) => {
                println!("[{}] k{i:02} FAILED: {e}", stamp());
                writeln!(index, "k{i:02}\t{}\t-\t0\tfalse\tFAILED: {e}", i % STRATEGIES.len()).map_err(|e| e.to_string())?;
            }
        }
    }
    println!("[{}] gen0 done: {n} calls in {:.0} s", stamp(), t_all.elapsed().as_secs_f64());
    Ok(())
}
