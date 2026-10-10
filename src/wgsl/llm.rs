//! THE MODEL CALLER: the one way this crate asks a language model for a
//! kernel or an opinion (`claude -p`, no tools, a temporary working
//! directory, no session persistence). Generation 0's contract and task
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

pub fn call(model: &str, system: &str, prompt: &str) -> Result<(String, String, f64), String> {
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
