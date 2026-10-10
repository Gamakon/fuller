//! Hallucino-genetic programming, generation 0 (`docs/PLAN_hgp_gram.md`,
//! the smallest experiment, step 1): `--n` isolated calls to Sonnet through
//! the Claude Code CLI with no tools (bare mode cannot use the OAuth login), each asked for a complete
//! Gram kernel under the role's contract and one strategy; every prompt and
//! reply saved verbatim, the kernel text extracted (fences stripped) and
//! written as `k<NN>.wgsl` for the ladder (`wgsl_gram_ladder`, one process
//! per kernel). Nothing here judges a kernel.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_gram_hgp -- gen0 [--n 32] [--seed 1] [--out logs/HGP/001/gen0] [--model sonnet]

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use fuller::wgsl::llm::{call, contract, strip, task, STRATEGIES};

fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}Z", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

/// The strategies, one per call in turn; the diversity source, since the
/// CLI exposes no sampling temperature.


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
