//! Virtual weight blocks (faxl-core's `genblock.fut`), baselines on the GPU through fuller's rust-wgpu path
//! (`fuller::futhark_rust`): the fork compiler builds faxl-core's draft, with `bench_entries.fut` appended, into a Rust
//! module run in this process on one wgpu device.
//!
//! ```text
//! cargo run --release --example virtual_blocks_bench -- WORK SET[:t] ...
//! ```
//!
//! For every SET (an input `WORK/SET.in` written by `make_inputs.py`): the outputs of `gen_b` and `read_b` are
//! written to `WORK/SET.gen.bin` and `WORK/SET.read.bin` (Futhark's binary format). With `:t` the set is also timed:
//! `PAIRS` interleaved pairs (the runtime's `p` command: the generator and the memory read alternate run by run on one
//! upload, at least 1 s of runs a side and at least `MIN_RUNS` runs, the side that goes first alternating pair by
//! pair), each run's GPU time from the device's timestamps. The per-run times go to `WORK/timing.tsv`; a summary line
//! a set goes to standard output.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuller::futhark_rust::{compile_module, prebuild_rt, Session};

const PAIRS: usize = 4;
const SIDE_NS: f64 = 1e9;
const MIN_RUNS: usize = 11;

fn utc() -> String {
    let s = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}Z", s / 3600 % 24, s / 60 % 60, s % 60)
}

/// The command records of a run: (tag, ok, payload), in command order. The runtime's set-up record (`ctx`) and its
/// opening, set-up and closing records (`start`, `ctx`, `end`) are dropped: a failed set-up is the run's failure.
fn records(raw: &[u8]) -> Result<Vec<(String, bool, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let nl = raw[i..].iter().position(|b| *b == b'\n').ok_or("a record with no end of line")? + i;
        let head = std::str::from_utf8(&raw[i..nl]).map_err(|_| "a record head that is not text")?;
        let f: Vec<&str> = head.split(' ').collect();
        if f.len() != 6 {
            return Err(format!("a record head of {} words: {head:?}", f.len()));
        }
        let n: usize = f[5].parse().map_err(|_| format!("a payload length {:?}", f[5]))?;
        let (tag, ok, payload) = (f[0], f[2] == "ok", &raw[nl + 1..nl + 1 + n]);
        match tag {
            "ctx" if !ok => return Err(format!("the runtime's set-up failed: {}", String::from_utf8_lossy(payload))),
            "start" | "ctx" | "end" => {}
            _ => out.push((tag.to_string(), ok, payload.to_vec())),
        }
        i = nl + 1 + n;
    }
    Ok(out)
}

/// The GPU nanoseconds of each timed run in one side of a `p` payload (the first word is the warm-up).
fn side_times(words: &str) -> Result<Vec<f64>, String> {
    words.split(' ').skip(1).map(|w| w.parse::<f64>().map_err(|_| format!("a GPU time {w:?} (a run with no GPU pass?)"))).collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn run(work: &Path, sets: &[String]) -> Result<(), String> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments/virtual_blocks");
    let read = |p: PathBuf| std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()));
    let source = format!("{}\n{}", read(here.join("genblock.fut"))?, read(here.join("bench_entries.fut"))?);
    println!("[{}] building the runtime and compiling the draft (fork compiler {})", utc(), fuller::futhark_rust::compiler().display());
    let rt = prebuild_rt(&work.join("rust_rt"))?;
    let (dylib, stages) = compile_module(&source, &work.join("rust_build"), &rt, "virtual_blocks", Duration::from_secs(900))?;
    println!("[{}] compiled: futhark {:.1} s, rustc {:.1} s", utc(), stages.futhark, stages.cc);
    let session = Session::new(&rt, Duration::from_secs(120))?;
    let mut tsv = String::from("set\tpair\tyard_first\tside\trun\tgpu_ns\n");
    for spec in sets {
        let (name, timed) = match spec.strip_suffix(":t") {
            Some(n) => (n, true),
            None => (spec.as_str(), false),
        };
        let input = std::fs::read(work.join(format!("{name}.in"))).map_err(|e| format!("{name}.in: {e}"))?;
        let raw = session.run(&dylib, "o gen_b\no read_b\n", &[], 0, 0, Some(&input))?;
        let recs = records(&raw)?;
        if recs.len() != 2 {
            let what: Vec<String> = recs.iter().map(|r| format!("{} {} {}", r.0, r.1, String::from_utf8_lossy(&r.2[..r.2.len().min(200)]))).collect();
            return Err(format!("{name}: {} records for 2 output commands: {what:?}", recs.len()));
        }
        for (rec, side) in recs.iter().zip(["gen", "read"]) {
            if !rec.1 {
                return Err(format!("{name} {side}: {}", String::from_utf8_lossy(&rec.2)));
            }
            std::fs::write(work.join(format!("{name}.{side}.bin")), &rec.2).map_err(|e| e.to_string())?;
        }
        println!("[{}] {name}: outputs written", utc());
        if !timed {
            continue;
        }
        let (mut gen, mut rd) = (Vec::new(), Vec::new());
        for pair in 0..PAIRS {
            let yard_first = pair % 2 == 0;
            let cmd = format!("p gen_b read_b {SIDE_NS} {MIN_RUNS} {}\n", u8::from(yard_first));
            let recs = records(&session.run(&dylib, &cmd, &[], 0, 0, Some(&input))?)?;
            let [rec] = recs.as_slice() else {
                return Err(format!("{name} pair {pair}: {} records for one pair command", recs.len()));
            };
            let text = String::from_utf8_lossy(&rec.2).to_string();
            if !rec.1 {
                return Err(format!("{name} pair {pair}: {text}"));
            }
            let (a, b) = text.split_once('|').ok_or("a pair payload with no |")?;
            for (side, words, acc) in [("gen", a, &mut gen), ("read", b, &mut rd)] {
                for (i, t) in side_times(words)?.into_iter().enumerate() {
                    writeln!(tsv, "{name}\t{pair}\t{}\t{side}\t{i}\t{t:.1}", u8::from(yard_first)).map_err(|e| e.to_string())?;
                    acc.push(t);
                }
            }
        }
        for v in [&mut gen, &mut rd] {
            v.sort_by(f64::total_cmp);
        }
        let (gm, rm) = (quantile(&gen, 0.5), quantile(&rd, 0.5));
        println!(
            "[{}] {name}: gen_b GPU us p10/p50/p90 {:.1}/{:.1}/{:.1} ({} runs); read_b {:.1}/{:.1}/{:.1} ({} runs); gen/read at p50 {:.2}",
            utc(),
            quantile(&gen, 0.1) / 1e3,
            gm / 1e3,
            quantile(&gen, 0.9) / 1e3,
            gen.len(),
            quantile(&rd, 0.1) / 1e3,
            rm / 1e3,
            quantile(&rd, 0.9) / 1e3,
            rd.len(),
            gm / rm
        );
    }
    std::fs::write(work.join("timing.tsv"), tsv).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: virtual_blocks_bench WORK SET[:t] ...");
        std::process::exit(2);
    }
    if let Err(e) = run(Path::new(&args[0]), &args[1..]) {
        eprintln!("[{}] virtual_blocks_bench: {e}", utc());
        std::process::exit(1);
    }
}
