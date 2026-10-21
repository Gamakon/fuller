//! ST4: ask a weak model (Haiku, through the `claude` CLI) for Karva symbol sequences for the true WGSL kingdom and decode
//! each reply into a gene by the rule in `fuller::wgsl::karva_seed`.
//!
//!   cargo run --release --features wgsl --example st4_seeds -- --out /path/phylu/logs/ST4 --count 100 --parallel 4
//!
//! Every prompt and every reply is saved under `<out>/calls/` (`NNN.prompt.txt`, `NNN.reply.json`, `NNN.response.txt`); a call
//! whose reply file exists is not repeated. The genes go to `<out>/seeds.tsv` (one row of space-separated symbol ids and
//! Dc indices a line), the decode counts and cost to `<out>/seeds_summary.tsv`. `--prompt-only I` prints prompt I and stops.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use fuller::typed_gep::{express, GeneLayout};
use fuller::wgsl::karva_seed::SeedSpec;
use fuller::wgsl::true_kingdom::{default_dir, TrueKingdom};

const SYSTEM: &str = "You write symbol-name sequences for a table decoder. Reply with the requested lines only.";

const STYLES: [&str; 20] = [
    "You remember the classic one-thread-per-output-cell Gram kernel.",
    "You remember a kernel that accumulates in a register over k and stores once at the end.",
    "You remember a kernel whose threads each own a small block of output cells.",
    "You remember a tiled kernel that stages columns in workgroup memory.",
    "You remember a kernel written by a careful CUDA veteran who avoids every race.",
    "You remember a minimal kernel: shortest correct loop, no extras.",
    "You remember a kernel that reads phi through vec4 loads.",
    "You remember a kernel with an unrolled inner loop.",
    "You remember a kernel that uses the row and column of the invocation id as i and j.",
    "You remember the textbook triple loop moved onto the GPU.",
    "You remember a kernel that exploits the symmetry of H.",
    "You remember a kernel that keeps one column of Phi in registers.",
    "You remember a kernel with a 16 by 16 workgroup.",
    "You remember a kernel with an 8 by 8 workgroup.",
    "You remember a kernel with a 32 by 8 workgroup.",
    "You remember a kernel that guards against indices past d.",
    "You remember a kernel that uses fused multiply-add for the accumulation.",
    "You remember a kernel that sums into a vec4 and reduces it at the end.",
    "You remember a kernel that reads cfg first and branches on r_rows.",
    "You remember a kernel written by a numerical analyst who cares about summation order.",
];

const BLOCKS: [&str; 5] = ["Prefer one cell per thread.", "Prefer two by two cells per thread.", "Prefer a block of four cells along the row.", "Prefer a block of four cells along the column.", "Prefer whatever block size you remember."];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn phrase(i: usize) -> String {
    format!("Variation {i:03}. {} {}", STYLES[i % STYLES.len()], BLOCKS[(i / STYLES.len() + i) % BLOCKS.len()])
}

struct Call {
    response: String,
    cost: f64,
    ms: f64,
}

/// One attempt: the CLI's stdout and whether it is a usable reply. A reply the model provider's safeguards cut off
/// (`system` / `informational` "safeguards stopped") or that is an error is not usable.
fn attempt(prompt: &str) -> Result<(String, Option<serde_json::Value>), String> {
    let mut child = Command::new("claude")
        .args(["-p", "--model", "haiku", "--output-format", "json", "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--effort", "low", "--system-prompt", SYSTEM])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("claude: {e}"))?;
    child.stdin.take().ok_or("stdin")?.write_all(prompt.as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let v: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(_) => return Ok((format!("{stdout}\n--stderr--\n{}", String::from_utf8_lossy(&out.stderr)), None)),
    };
    let elements = match &v {
        serde_json::Value::Array(a) => a.clone(),
        o => vec![o.clone()],
    };
    let stopped = elements.iter().any(|x| x["subtype"] == "informational" && x["content"].as_str().is_some_and(|c| c.contains("safeguards stopped")));
    let result = elements.iter().find(|x| x["type"] == "result").cloned();
    match result {
        Some(r) if !stopped && r["is_error"] != true && out.status.success() => Ok((stdout, Some(r))),
        _ => Ok((stdout, None)),
    }
}

/// Up to three attempts; every attempt's reply is kept (`NNN.aK.reply.json` for a failed one).
fn call_claude(prompt: &str, calls: &Path, i: usize) -> Result<Call, String> {
    let reply_path = calls.join(format!("{i:03}.reply.json"));
    let t = Instant::now();
    std::fs::write(calls.join(format!("{i:03}.prompt.txt")), prompt).map_err(|e| e.to_string())?;
    let mut result = None;
    if reply_path.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&reply_path).map_err(|e| e.to_string())?).map_err(|e| format!("call {i}: {e}"))?;
        result = v.as_array().and_then(|a| a.iter().find(|x| x["type"] == "result").cloned());
    }
    let mut cost_failed = 0.0;
    for k in 0..3 {
        if result.is_some() {
            break;
        }
        let (stdout, r) = attempt(prompt)?;
        match r {
            Some(r) => {
                std::fs::write(&reply_path, &stdout).map_err(|e| e.to_string())?;
                result = Some(r);
            }
            None => {
                std::fs::write(calls.join(format!("{i:03}.a{k}.reply.json")), &stdout).map_err(|e| e.to_string())?;
                cost_failed += serde_json::from_str::<serde_json::Value>(&stdout).ok().and_then(|v| v.as_array().and_then(|a| a.iter().find(|x| x["type"] == "result").and_then(|r| r["total_cost_usd"].as_f64()))).unwrap_or(0.0);
            }
        }
    }
    let result = result.ok_or_else(|| format!("call {i}: three attempts gave no usable reply (see {i:03}.aK.reply.json)"))?;
    let response = result["result"].as_str().ok_or_else(|| format!("call {i}: no result text"))?.to_string();
    std::fs::write(calls.join(format!("{i:03}.response.txt")), &response).map_err(|e| e.to_string())?;
    Ok(Call { response, cost: result["total_cost_usd"].as_f64().unwrap_or(0.0) + cost_failed, ms: result["duration_ms"].as_f64().unwrap_or(t.elapsed().as_secs_f64() * 1e3) })
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    let count: usize = arg(&args, "--count").map_or(Ok(100), |v| v.parse().map_err(|e| format!("--count: {e}")))?;
    let parallel: usize = arg(&args, "--parallel").map_or(Ok(4), |v| v.parse().map_err(|e| format!("--parallel: {e}")))?;
    let n: usize = arg(&args, "--n").map_or(Ok(8), |v| v.parse().map_err(|e| format!("--n: {e}")))?;
    let write_rows: usize = arg(&args, "--write-rows").map_or(Ok(6), |v| v.parse().map_err(|e| format!("--write-rows: {e}")))?;
    let max_list: usize = arg(&args, "--max-list").map_or(Ok(200), |v| v.parse().map_err(|e| format!("--max-list: {e}")))?;
    let kingdom = TrueKingdom::load(&default_dir())?;
    let contract = kingdom.relations().check().map_err(|d| format!("the kingdom check failed: {:?}", &d[..d.len().min(3)]))?;
    let layout = GeneLayout::from_program_size(&contract, kingdom.root, n).map_err(|e| e.to_string())?;
    let spec = SeedSpec::new(&kingdom, &contract, &layout);
    if let Some(i) = arg(&args, "--prompt-only") {
        let i: usize = i.parse().map_err(|e| format!("--prompt-only: {e}"))?;
        let p = spec.prompt(&phrase(i), write_rows, max_list);
        println!("{p}");
        eprintln!("# prompt {} bytes, {} frames, row {} words", p.len(), spec.frames.len(), layout.width);
        return Ok(());
    }
    let calls = out.join("calls");
    std::fs::create_dir_all(&calls).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<(usize, Result<Call, String>)>> = std::sync::Mutex::new(Vec::new());
    // the first call alone writes the prompt cache (the prefix is shared); the rest read it
    let first = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if first < count {
        let r = call_claude(&spec.prompt(&phrase(first), write_rows, max_list), &calls, first);
        eprintln!("# call {first:03} (cache warm-up) done at {:.0} s: {}", started.elapsed().as_secs_f64(), r.as_ref().map_or_else(|e| e.clone(), |c| format!("{:.0} ms, ${:.4}", c.ms, c.cost)));
        results.lock().expect("results").push((first, r));
    }
    std::thread::scope(|s| {
        for _ in 0..parallel {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= count {
                    break;
                }
                let prompt = spec.prompt(&phrase(i), write_rows, max_list);
                let r = call_claude(&prompt, &calls, i);
                eprintln!("# call {i:03} done at {:.0} s: {}", started.elapsed().as_secs_f64(), r.as_ref().map_or_else(|e| e.clone(), |c| format!("{:.0} ms, ${:.4}", c.ms, c.cost)));
                results.lock().expect("results").push((i, r));
            });
        }
    });
    let mut results = results.into_inner().expect("results");
    results.sort_by_key(|(i, _)| *i);
    let mut seeds = std::fs::File::create(out.join("seeds.tsv")).map_err(|e| e.to_string())?;
    let mut summary = std::fs::File::create(out.join("seeds_summary.tsv")).map_err(|e| e.to_string())?;
    writeln!(summary, "seed\tcost_usd\tcall_ms\tnamed\thashed\tfilled\tignored_lines\tnodes").map_err(|e| e.to_string())?;
    let (mut cost, mut ms, mut failed) = (0.0, 0.0, 0);
    let const_len = kingdom.const_lens();
    for (i, r) in &results {
        match r {
            Ok(c) => {
                let (gene, counts) = spec.decode(&c.response, 0x5714_0000 + *i as u64);
                let nodes = express(&contract, &layout, &gene, &const_len).map_err(|e| format!("our software has a bug: seed {i} does not express: {e}"))?.nodes.len();
                writeln!(seeds, "{}", gene.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")).map_err(|e| e.to_string())?;
                writeln!(summary, "{i}\t{:.5}\t{:.0}\t{}\t{}\t{}\t{}\t{nodes}", c.cost, c.ms, counts.named, counts.hashed, counts.filled, counts.ignored_lines).map_err(|e| e.to_string())?;
                cost += c.cost;
                ms += c.ms;
            }
            Err(e) => {
                failed += 1;
                eprintln!("{e}");
            }
        }
    }
    println!("# {} calls, {failed} failed, cost ${cost:.4}, summed call time {:.0} s, wall {:.0} s", results.len(), ms / 1e3, started.elapsed().as_secs_f64());
    if failed > 0 {
        return Err(format!("{failed} calls failed"));
    }
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
