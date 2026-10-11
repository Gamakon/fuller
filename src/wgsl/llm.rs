//! THE MODEL CALLER: the one way this crate asks a language model for a
//! kernel or an opinion (`claude -p` for sonnet, opus and fable; `codex
//! exec` for codex; no tools, a temporary working directory, no session
//! persistence, no timeout). Generation 0's contract and task
//! prompts live here with it, so a repair or a proposal prompt quotes the
//! same contract the drafts were written to.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::wgsl::gram::SHIPPED;

/// The strategies generation 0 drew from, one per call.
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
pub fn contract() -> String {
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

pub fn task(strategy: &str, seed_phrase: &str) -> String {
    format!("Write the kernel. Strategy to use: {strategy}. Variation key: {seed_phrase}. Remember: first line `// grid X Y BX BY`, only the kernel text.")
}

pub fn strip(reply: &str) -> String {
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

/// THE MODEL MIX (Andrew, 2026-10-11): every call draws its model, Sonnet
/// 70 %, Opus 20 %, Fable 5 %, Codex 5 %, from the run's own generator so a
/// run is reproducible call for call; the ledger names the model of every
/// kernel so the mix can be judged per model.
pub const MIX: [(&str, u64); 4] = [("sonnet", 70), ("opus", 20), ("fable", 5), ("codex", 5)];

/// The next model of the mix from a 64-bit generator state (advanced here).
pub fn draw_model(state: &mut u64) -> &'static str {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut r = (*state >> 33) % 100;
    for (name, share) in MIX {
        if r < share {
            return name;
        }
        r -= share;
    }
    MIX[0].0
}

/// A call saved: the prompt and the reply as files under `dir`, named by
/// `id` and the model, so every kernel a model wrote can be read later.
pub fn save_call(dir: &std::path::Path, id: &str, model: &str, system: &str, prompt: &str, reply: &str) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join(format!("{id}.{model}.prompt.txt")), format!("{system}\n\n=====\n\n{prompt}")).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{id}.{model}.reply.txt")), reply).map_err(|e| e.to_string())
}

/// One call to the model named: `claude -p --model <m>` for sonnet, opus and
/// fable; `codex exec` for codex (the system text and the prompt as one
/// message, the last message read back from a file). No timeout: a model
/// takes as long as it takes (Andrew). Returns (reply, raw output, seconds).
pub fn call(model: &str, system: &str, prompt: &str) -> Result<(String, String, f64), String> {
    if model == "codex" {
        return call_codex(system, prompt);
    }
    let t = Instant::now();
    let mut child = Command::new("claude")
        // Not `--bare`: bare mode reads only an API key and this machine is
        // logged in through OAuth ("Not logged in" under --bare, measured).
        // The isolation is: no tools at all, a temporary working directory
        // with no project instructions, no session persistence.
        .args(["-p", "--model", model, "--output-format", "json", "--no-session-persistence", "--tools", "", "--system-prompt", system])
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
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = &stdout[stdout.len().saturating_sub(700)..];
        return Err(format!("claude exited {}: stderr {:?}; stdout tail {:?}", out.status, &err[..err.len().min(400)], tail));
    }
    // --output-format json: one JSON object (or an array of events) with a `result` string.
    let json: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| format!("json: {e}: {}", &stdout[..stdout.len().min(200)]))?;
    let event = if json.is_array() { json.as_array().and_then(|a| a.iter().find(|v| v.get("type").and_then(|t| t.as_str()) == Some("result"))).cloned() } else { Some(json) };
    let event = event.ok_or_else(|| format!("no result event: {}", &stdout[..stdout.len().min(300)]))?;
    let result = event.get("result").and_then(|r| r.as_str()).map(str::to_string).ok_or_else(|| format!("no result field: {}", &stdout[..stdout.len().min(300)]))?;
    if event.get("is_error").and_then(|e| e.as_bool()) == Some(true) {
        return Err(format!("claude error: {result}"));
    }
    Ok((result, stdout, t.elapsed().as_secs_f64()))
}

fn call_codex(system: &str, prompt: &str) -> Result<(String, String, f64), String> {
    let t = Instant::now();
    let last = std::env::temp_dir().join(format!("codex_last_{}_{}.txt", std::process::id(), t.elapsed().as_nanos()));
    let mut child = Command::new("codex")
        .args(["exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only", "--output-last-message"])
        .arg(&last)
        .arg("-")
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("codex: {e}"))?;
    child.stdin.take().ok_or("stdin")?.write_all(format!("{system}\n\n{prompt}").as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("codex exited {}: {:?}", out.status, &err[..err.len().min(400)]));
    }
    let result = std::fs::read_to_string(&last).map_err(|e| format!("codex last message {}: {e}", last.display()))?;
    let _ = std::fs::remove_file(&last);
    Ok((result, stdout, t.elapsed().as_secs_f64()))
}

/// What one parent (or any measured kernel) is, for a brief.
#[derive(Debug, Clone)]
pub struct KernelBrief {
    pub id: String,
    pub text: String,
    pub hff: f64,
    pub speedups: [f64; 3],
    pub columns: [f64; 6],
    /// Its problem in one line, e.g. "slower than the other parent".
    pub problem: String,
}

/// THE FITNESS EXPLAINED to the model (Andrew: "explain the fitness measures
/// for each, and the problem to solve for each; quality for example").
pub fn fitness_explainer() -> String {
    "HOW A KERNEL IS SCORED. First the constraint: 128 correctness cases (random d, m, r_rows, r_off, including partial tiles and degenerate shapes) against an f64 reference; one wrong cell and the kernel is infeasible, score pi/2, the worst. \
A feasible kernel gets six columns, each in [0, 1], 0 best, and its score is the angle (HFF) between the column vector and the origin direction, so LOWER IS BETTER and the columns must ALL be small: \
(1) (2) (3) SPEED on three shapes (d 2048 m 65536; d 4096 m 16384; d 8192 m 4096): 1 / speed-up over the shipped kernel, so 0.125 means 8x faster; \
(4) (5) (6) QUALITY, the numeric error: the worst max-ratio error against the f64 reference on three sets of (phi, H) pairs (train, validation, extrapolation), each divided by the shipped kernel's own error on the same pair, then halved and capped at 1; 0.29 means the error is about 0.6 of shipped's, 0.5 means equal to shipped's, and more than twice shipped's error is INFEASIBLE. \
Accumulating in a different order changes the error: more independent partial sums and pairwise adds keep it low; a long serial sum raises it."
        .to_string()
}

fn brief_block(k: &KernelBrief, label: &str) -> String {
    format!(
        "===== {label}: {} =====\nHFF {:.4}; speed-ups {:.2}x / {:.2}x / {:.2}x; columns {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3}\nPROBLEM: {}\n```wgsl\n{}```\n",
        k.id, k.hff, k.speedups[0], k.speedups[1], k.speedups[2], k.columns[0], k.columns[1], k.columns[2], k.columns[3], k.columns[4], k.columns[5], k.problem, k.text
    )
}

/// THE CROSSOVER BRIEF: two feasible parents, their measurements and problems,
/// the sections they share (the integration point), the size bound.
pub fn crossover_task(a: &KernelBrief, b: &KernelBrief, shared: &[(usize, String)], max_nodes: usize) -> String {
    let mut shared_text = String::new();
    for (nodes, text) in shared.iter().take(4) {
        shared_text.push_str(&format!("{nodes} nodes: {text}\n"));
    }
    format!(
        "You are the integrator in a crossover. Two correct kernels below were written by different strategies. Write ONE new kernel that keeps the SHARED STRUCTURE exactly (the same workgroup shape, the same thread-to-cell mapping, the same store guard) and combines the technique of each parent so that the child beats both on the score: as fast as the faster parent, as accurate as the more accurate one.\n\n{}\n\n{}\n{}\n===== THE SHARED STRUCTURE (sub-trees of the two kernels' typed parse trees that are identical, largest first; Karva s-expressions over the kernel grammar) =====\n{shared_text}\nSIZE: the kernel must parse to at most {max_nodes} nodes of the grammar (roughly {} lines); prefer the smaller of two equivalent spellings.\nRETURN ONLY the kernel text, starting with the header comment line `// grid x y bx by`, no prose, no fences.",
        fitness_explainer(),
        brief_block(a, "PARENT A"),
        brief_block(b, "PARENT B"),
        max_nodes / 12
    )
}

/// THE REINVENTION BRIEF (Andrew, 2026-10-11: "crossover is often about the
/// mixing of genes producing surprising outcomes"): the second child of a
/// pair. The model keeps `keep`'s processing outside the shared section
/// exactly and REINVENTS the shared middle, having seen how `other` does it.
pub fn reinvent_task(keep: &KernelBrief, other: &KernelBrief, shared: &[(usize, String)], max_nodes: usize) -> String {
    let mut shared_text = String::new();
    for (nodes, text) in shared.iter().take(4) {
        shared_text.push_str(&format!("{nodes} nodes: {text}\n"));
    }
    format!(
        "You are the inventor in a crossover. Two correct kernels below share a MIDDLE: the structure named under THE SHARED STRUCTURE (how a thread is mapped to its output cells and how the store is guarded). Write ONE new kernel that keeps PARENT A's processing OUTSIDE that shared middle (its loads, its accumulation, its stores, its loop structure) as it is, and COMPLETELY REINVENTS the shared middle: a different thread-to-cell mapping, block shape, tile shape or work split than either parent uses, chosen to be faster on an Apple M3 Max under Metal. Parent B is shown so you know what the other parent did with the same middle; do not copy either parent's middle. The result must be correct for every d, m, r_rows, r_off the harness can pass, including partial tiles at the edges.\n\n{}\n\n{}\n{}\n===== THE SHARED STRUCTURE (what both parents do; reinvent this) =====\n{shared_text}\nSIZE: at most {max_nodes} nodes of the grammar (roughly {} lines).\nRETURN ONLY the kernel text, starting with the header comment line `// grid x y bx by`, no prose, no fences.",
        fitness_explainer(),
        brief_block(keep, "PARENT A (keep its processing outside the shared middle)"),
        brief_block(other, "PARENT B (for reference only)"),
        max_nodes / 12
    )
}

/// THE REPAIR BRIEF: an offspring that failed, what failed, its parents.
pub fn repair_task(child: &str, failure: &str, parents: &[&KernelBrief], max_nodes: usize) -> String {
    let mut p = String::new();
    for (i, k) in parents.iter().enumerate() {
        p.push_str(&brief_block(k, &format!("PARENT {}", (b'A' + i as u8) as char)));
    }
    format!(
        "You are repairing an offspring kernel. It was made from the parents below and it FAILED the constraint. Make it correct while keeping its technique (its tiling, its accumulation, its loads): change as little as the failure needs. If the failure names a case, that case's shape is where it is wrong; think about partial tiles, the band (r_rows, r_off), bounds, barriers under conditions, and uninitialised accumulators.\n\n{}\n\n===== THE OFFSPRING (failed) =====\nFAILURE: {failure}\n```wgsl\n{child}```\n\n{p}\nSIZE: at most {max_nodes} nodes of the grammar.\nRETURN ONLY the repaired kernel text, starting with the header comment line `// grid x y bx by`, no prose, no fences.",
        fitness_explainer()
    )
}

/// A PROPOSAL REQUEST: an individual the safeWGSL kingdom constructed that
/// failed (the kernel text and what failed), asked of the model as a question
/// about the GRAMMAR, never as a repair: what rule or type would make this
/// failure impossible to spell. The answer is collected, not acted on.
pub fn proposal_system(forms_md: &str) -> String {
    format!(
        "You are reviewing a typed grammar (a 'kingdom') for evolving WGSL compute kernels by genetic programming. \
Every kernel the search builds is a typed tree over the kingdom's rows; the types ('duals', slot.form) are designed so that a kernel that would crash, hang or write outside its cell CANNOT be constructed. \
The kingdom's forms are below. You will be shown a kernel the kingdom DID construct that failed, with the failure. \
Your task is NOT to fix the kernel. It is to propose the smallest change to the kingdom (a new form, a split of a form, a row whose inputs are narrowed, a row removed) that would make this failure unconstructible, and to say what the change would also forbid. \
Answer in at most 200 words of plain English, as: FAILURE CLASS: ...; PROPOSAL: ...; WOULD ALSO FORBID: ...; CONFIDENCE: low/medium/high.\n\nTHE KINGDOM'S FORMS:\n{forms_md}"
    )
}

pub fn proposal_task(id: &str, failure: &str, kernel: &str) -> String {
    format!("Individual {id}.\nFAILURE: {failure}\n\nKERNEL (as the kingdom rendered it):\n```wgsl\n{kernel}```\n\nGive the proposal in the stated format only.")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mix over ten thousand draws is 70 / 20 / 5 / 5 within a point,
    /// and the same seed draws the same sequence.
    #[test]
    fn the_model_mix_draws_its_shares_reproducibly() {
        let mut counts = std::collections::BTreeMap::new();
        let mut state = 42u64;
        for _ in 0..10_000 {
            *counts.entry(draw_model(&mut state)).or_insert(0u32) += 1;
        }
        for (name, share) in MIX {
            let got = f64::from(counts[name]) / 100.0;
            assert!((got - share as f64).abs() < 1.5, "{name}: {got} % for {share} %");
        }
        let (mut a, mut b) = (7u64, 7u64);
        let x: Vec<&str> = (0..50).map(|_| draw_model(&mut a)).collect();
        let y: Vec<&str> = (0..50).map(|_| draw_model(&mut b)).collect();
        assert_eq!(x, y);
    }

    /// The briefs carry the fitness explanation, both parents' measurements
    /// and problems, the shared sections and the size bound.
    #[test]
    fn the_briefs_carry_the_measurements() {
        let k = |id: &str| KernelBrief { id: id.into(), text: "// grid 1 1 1 1\n".into(), hff: 0.33, speedups: [6.25, 6.82, 7.03], columns: [0.16, 0.147, 0.142, 0.296, 0.289, 0.293], problem: "slower".into() };
        let t = crossover_task(&k("k15"), &k("k07"), &[(33, "(logic.and ...)".into())], 3000);
        for needle in ["QUALITY", "PARENT A: k15", "PARENT B: k07", "6.25x", "0.296", "PROBLEM: slower", "33 nodes: (logic.and ...)", "at most 3000 nodes"] {
            assert!(t.contains(needle), "{needle}");
        }
        let r = repair_task("// grid 1 1 1 1\n", "constraint 0/128 cases: A t0 c14 v3", &[&k("k15")], 3000);
        assert!(r.contains("FAILURE: constraint 0/128 cases: A t0 c14 v3") && r.contains("PARENT A: k15"));
        let v = reinvent_task(&k("k15"), &k("k07"), &[(33, "(logic.and ...)".into())], 3000);
        for needle in ["REINVENTS the shared middle", "PARENT A (keep its processing outside the shared middle): k15", "PARENT B (for reference only): k07", "33 nodes: (logic.and ...)"] {
            assert!(v.contains(needle), "{needle}");
        }
    }
}
