//! SEED A v24 POPULATION FROM MODELS, AND NOTHING ELSE: the model-seeding
//! step of phylu's `wgsl_fit --seed-from model --seed-models ...` (phylu
//! examples/wgsl_fit.rs, the launch seeds), run alone so a fresh population
//! can be made for a problem without a fit and without the GPU.
//!
//! The method is wgsl_fit's: the problem's contract (`llm::contract`: its
//! contract.md, its reference kernel, the output rule); seed `i` (0-based, in
//! the order the `--models` mix lists) gets strategy `i` of the contract's
//! `## Strategies` (cycling) and the variation key "population seed <seed>,
//! launch kernel <i>, be unlike the obvious kernel"; every call of a batch
//! goes out at once (one thread a call; `claude -p` for sonnet, opus and
//! fable, `codex exec` for codex: `llm::call`); a seed whose call fails or
//! whose reply does not compile is asked again of the same model, up to three
//! calls a seed in all; a retry after a reply that did not compile quotes the
//! compile error ("Your previous reply did not compile: ...").
//!
//! `--only seed21,seed23,seed28` redoes only those seeds into an existing
//! `--out`: the seeds and calls already there are kept, each named seed's first
//! call quotes its last compile error from checks.tsv, its attempts are
//! numbered on from the ones recorded (no call file is overwritten),
//! checks.tsv is appended to, and models.tsv is rewritten with every seed that
//! compiles (the old and the new). The same --models and --seed as the first
//! run give each seed its model and variation key.
//!
//! Out, in `--out`: `seedNN.wgsl` for every seed whose reply compiles
//! (`harness::kernel_info`: naga parses and validates it and its bindings meet
//! the problem's schema, on the host; nothing runs), `models.tsv` (seed,
//! model, form "raw": the layout phylu's `--seed-from saved --seed-dir` reads),
//! `calls/` (every reply as returned, `llm::save_call`), `checks.tsv` (every
//! call: seed, attempt, model, seconds, the check's verdict).
//!
//! Run: cargo run --release --example v24_seed -- --problem <dir> --out <dir>
//!      --models sonnet:16,opus:8,codex:4,fable:4 --seed <n> [--only seedNN,...]

use std::path::PathBuf;

use fuller::wgsl::harness::{kernel_info, Problem};
use fuller::wgsl::llm;

/// One model call in flight: its thread, the reply (stripped) and its seconds.
type Call = std::thread::JoinHandle<Result<(String, f64), String>>;

fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Result<String, String> { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().ok_or_else(|| format!("{flag} <value> is required")) };
    let problem_dir = PathBuf::from(value("--problem")?);
    let out = PathBuf::from(value("--out")?);
    let seed: u64 = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
    let mut plan: Vec<String> = Vec::new();
    for part in value("--models")?.split(',') {
        let (name, n) = part.split_once(':').ok_or_else(|| format!("--models: {part} is not model:count"))?;
        if !matches!(name, "sonnet" | "opus" | "fable" | "codex") {
            return Err(format!("--models: {name} is not sonnet, opus, fable or codex"));
        }
        let n: usize = n.parse().map_err(|e| format!("--models: {part}: {e}"))?;
        plan.extend(std::iter::repeat_n(name.to_string(), n));
    }
    let only: Option<Vec<usize>> = match args.iter().position(|a| a == "--only") {
        None => None,
        Some(_) => Some(
            value("--only")?
                .split(',')
                .map(|s| s.trim().strip_prefix("seed").and_then(|n| n.parse::<usize>().ok()).filter(|n| *n < plan.len()).ok_or_else(|| format!("--only: {s} is not seedNN of the {} seeds", plan.len())))
                .collect::<Result<_, _>>()?,
        ),
    };
    // Each seed: its model when accepted; the calls made of it so far; the
    // compile error of its last reply that did not compile.
    let mut accepted: Vec<Option<String>> = vec![None; plan.len()];
    let mut made: Vec<usize> = vec![0; plan.len()];
    let mut last_error: Vec<Option<String>> = vec![None; plan.len()];
    let mut checks = String::from("seed\tattempt\tmodel\tseconds\tcheck\n");
    let wanted_at_start: Vec<usize> = match &only {
        None => {
            if out.exists() {
                return Err(format!("{} exists: a seed population is written to a new directory only (or name the seeds to redo with --only)", out.display()));
            }
            (0..plan.len()).collect()
        }
        Some(redo) => {
            // The run so far: its models.tsv, its checks.tsv.
            let read = |f: &str| std::fs::read_to_string(out.join(f)).map_err(|e| format!("--only: {}: {e}", out.join(f).display()));
            for l in read("models.tsv")?.lines().skip(1) {
                let mut c = l.split('\t');
                let (Some(stem), Some(model)) = (c.next(), c.next()) else { continue };
                let i: usize = stem.strip_prefix("seed").and_then(|n| n.parse().ok()).ok_or_else(|| format!("--only: models.tsv line {l:?}"))?;
                if i >= plan.len() || plan[i] != model {
                    return Err(format!("--only: models.tsv says {stem} is {model}; --models gives {}", plan.get(i).map_or("no seed", |s| s.as_str())));
                }
                accepted[i] = Some(model.to_string());
            }
            checks = read("checks.tsv")?;
            for l in checks.lines().skip(1) {
                let c: Vec<&str> = l.split('\t').collect();
                let Some(i) = c.first().and_then(|s| s.strip_prefix("seed")).and_then(|n| n.parse::<usize>().ok()).filter(|i| *i < plan.len()) else { continue };
                if let Some(a) = c.get(1).and_then(|a| a.parse::<usize>().ok()) {
                    made[i] = made[i].max(a);
                }
                if let Some(e) = c.get(4).and_then(|v| v.strip_prefix("does not compile: ")) {
                    last_error[i] = Some(e.to_string());
                }
            }
            if !checks.ends_with('\n') {
                checks.push('\n');
            }
            for i in redo {
                if accepted[*i].is_some() {
                    return Err(format!("--only: seed{i:02} already compiles (models.tsv): it is not redone"));
                }
            }
            redo.clone()
        }
    };
    std::fs::create_dir_all(out.join("calls")).map_err(|e| format!("{}: {e}", out.display()))?;
    let problem = Problem::load(&problem_dir)?;
    let contract = llm::contract(Some(&problem))?;
    println!("[{}] {} seeds for {} from {} strategies; models {}{}", stamp(), wanted_at_start.len(), problem.spec.name, contract.strategies.len(), value("--models")?, if only.is_some() { format!("; redoing {:?}", wanted_at_start.iter().map(|i| format!("seed{i:02}")).collect::<Vec<_>>()) } else { String::new() });
    let mut wanted = wanted_at_start;
    for _ in 1..=3 {
        if wanted.is_empty() {
            break;
        }
        let handles: Vec<(usize, String, Call)> = wanted
            .iter()
            .map(|&i| {
                let (c, name) = (contract.clone(), plan[i].clone());
                let strategy = c.strategy(i).to_string();
                let phrase = format!("population seed {seed}, launch kernel {i}, be unlike the obvious kernel");
                let mut task = c.task(&strategy, &phrase);
                if let Some(e) = &last_error[i] {
                    task.push_str(&format!("\n\nYour previous reply did not compile: {e}. Write the whole kernel again with that fixed."));
                }
                let sent = task.clone();
                (i, sent, std::thread::spawn(move || llm::call(&name, &c.system, &task).map(|(r, _, secs)| (llm::strip(&r), secs))))
            })
            .collect();
        let mut still = Vec::new();
        for (i, sent, h) in handles {
            let name = &plan[i];
            made[i] += 1;
            let attempt = made[i];
            let verdict = match h.join() {
                Ok(Ok((text, secs))) => {
                    llm::save_call(&out.join("calls"), &format!("seed{i:02}_try{attempt}"), name, &contract.system, &sent, &text)?;
                    match kernel_info(&problem.spec, &text) {
                        Ok(_) => {
                            std::fs::write(out.join(format!("seed{i:02}.wgsl")), &text).map_err(|e| e.to_string())?;
                            accepted[i] = Some(name.clone());
                            format!("{secs:.0}\tcompiles")
                        }
                        Err(e) => {
                            still.push(i);
                            let first = e.lines().next().unwrap_or("").chars().take(160).collect::<String>();
                            last_error[i] = Some(first.clone());
                            format!("{secs:.0}\tdoes not compile: {first}")
                        }
                    }
                }
                Ok(Err(e)) => {
                    still.push(i);
                    format!("-\tthe call failed: {}", e.chars().take(160).collect::<String>())
                }
                Err(_) => {
                    still.push(i);
                    "-\tthe call's thread panicked".to_string()
                }
            };
            println!("[{}] seed{i:02} {name} attempt {attempt}: {}", stamp(), verdict.replace('\t', " s, "));
            checks.push_str(&format!("seed{i:02}\t{attempt}\t{name}\t{verdict}\n"));
        }
        wanted = still;
    }
    let models: String = accepted.iter().enumerate().filter_map(|(i, m)| m.as_ref().map(|m| format!("seed{i:02}\t{m}\traw\n"))).collect();
    std::fs::write(out.join("models.tsv"), format!("seed\tmodel\tform\n{models}")).map_err(|e| e.to_string())?;
    std::fs::write(out.join("checks.tsv"), checks).map_err(|e| e.to_string())?;
    let n = accepted.iter().filter(|m| m.is_some()).count();
    println!("[{}] {n} of {} seeds compile; {} without one after this run's calls: {:?}", stamp(), plan.len(), wanted.len(), wanted.iter().map(|i| format!("seed{i:02} {}", plan[*i])).collect::<Vec<_>>());
    Ok(())
}
