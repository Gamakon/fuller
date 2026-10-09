//! The oracle on the device (`docs/PLAN_wgsl_lineage.md` §4): every
//! hand-written kernel in `kingdoms/wgsl/oracle/` is run three ways and
//! compared with its hand-written expectation:
//!
//!   original  — the kernel's own text on the device (the authority);
//!   rebuilt   — the text the scaffold rebuilds from the chromosome, on
//!               the device;
//!   interp    — the folded chromosome in the reference interpreter
//!               (`src/wgsl/interp.rs`), where the kernel is one
//!               invocation.
//!
//! One workgroup is dispatched. Lanes are compared as bits; `"nan"` in an
//! expectation accepts any NaN; `ok*` means the device agreed except on
//! lanes the expectation lists as a known platform deviation from the
//! spec (printed). A difference between the original and the
//! expectation is a finding about the device or the expectation; between
//! the interpreter and the original, about the reader or the interpreter;
//! between the rebuilt text and the original, about the rebuild.
//!
//! With `--generate N [--seed S]` the generator's programs (`src/wgsl/
//! generator.rs`) are run after the hand-written set: for each, the
//! original on the device, the interpreter on the folded chromosome
//! (compared with the original lane by lane, any NaN equal to any NaN,
//! since WGSL fixes no payload), and the rebuilt text on the device
//! (compared with the original bit for bit). The summary counts the
//! refusal classes the programs meant to reach against the ones the
//! decision reported, and prints the first differing program in full.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_oracle -- [--dir kingdoms/wgsl/oracle] [--generate N] [--seed S]

use std::collections::BTreeMap;
use std::path::PathBuf;

use fuller::wgsl::generator::generate;
use fuller::wgsl::oracle::{bindings, check, device::Device, entry_name, hand_kernels, interpret};
use fuller::wgsl::{chromosome, read, round_trip, ChromosomeOptions, Interp, Invocation, Memory, Value};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1)).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("kingdoms/wgsl/oracle"));
    let generate_n: usize = args.iter().position(|a| a == "--generate").and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(0);
    let seed0: u64 = args.iter().position(|a| a == "--seed").and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(1);
    let device = Device::new()?;
    println!("device: {}", device.name);
    hand_set(&device, &dir)?;
    if generate_n > 0 {
        generated(&device, generate_n, seed0)?;
    }
    Ok(())
}

/// The distance in units of the last place between two finite f32 bit
/// patterns of the same sign (`None` for anything else).
fn ulps(a: u32, b: u32) -> Option<u32> {
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    if !x.is_finite() || !y.is_finite() || (x.is_sign_negative() != y.is_sign_negative() && x != 0.0 && y != 0.0) {
        return None;
    }
    let key = |w: u32| if w & 0x8000_0000 != 0 { !(w & 0x7fff_ffff) } else { w | 0x8000_0000 };
    Some(key(a).abs_diff(key(b)))
}

/// Lanes where `a` and `b` differ. With `tolerance` (the interpreter
/// against the device): any NaN equals any NaN, and a float lane within
/// `max_ulps` counts as contraction (Metal fuses `a * b + c`, the
/// interpreter does not) and is reported separately, not as a difference.
/// Without it (the rebuilt text against the original, same compiler): bits.
fn lane_differences(a: &Value, b: &Value, tolerance: Option<u32>) -> (Vec<(usize, u32, u32)>, usize) {
    let (wa, wb) = (fuller::wgsl::oracle::to_words(a), fuller::wgsl::oracle::to_words(b));
    let mut out = Vec::new();
    let mut contracted = 0usize;
    for i in 0..wa.len().max(wb.len()) {
        let (x, y) = (wa.get(i).copied(), wb.get(i).copied());
        let same = match (x, y, tolerance) {
            (Some(x), Some(y), _) if x == y => true,
            (Some(x), Some(y), Some(max_ulps)) => {
                if f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan() {
                    true
                } else if ulps(x, y).is_some_and(|u| u <= max_ulps) {
                    contracted += 1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !same {
            out.push((i, x.unwrap_or(0), y.unwrap_or(0)));
        }
    }
    (out, contracted)
}

/// The float lanes the interpreter may differ on by contraction: Metal
/// fuses a multiply and an add into one rounding and the interpreter
/// rounds twice (wgpu leaves Metal's default fast math on); a few such
/// fusions along one value make a few ulps.
const CONTRACTION_ULPS: u32 = 8;

fn generated(device: &Device, n: usize, seed0: u64) -> Result<(), String> {
    let mut intended: BTreeMap<&str, usize> = BTreeMap::new();
    let mut reached: BTreeMap<String, usize> = BTreeMap::new();
    let (mut agree, mut interp_differs, mut rebuilt_differs, mut errors, mut shared_programs, mut contracted_lanes, mut contracted_programs) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    let mut all_notes: Vec<String> = Vec::new();
    for seed in seed0..seed0 + n as u64 {
        let g = generate(seed);
        for c in &g.intended {
            *intended.entry(c).or_default() += 1;
        }
        let kernel = read(&g.source).map_err(|e| format!("seed {seed}: {e}
{}", g.source))?;
        let chromosomes: Vec<_> = kernel.functions.iter().map(|f| chromosome(f, &ChromosomeOptions::default())).collect::<Result<_, _>>().map_err(|e| format!("seed {seed}: {e}"))?;
        let entry = entry_name(&kernel)?;
        let main = chromosomes.iter().find(|c| c.function == entry).ok_or("no entry chromosome")?;
        if main.folded.filled > 0 {
            shared_programs += 1;
            *reached.entry("share".into()).or_default() += 1;
        }
        for (r, k) in &main.refused {
            *reached.entry(r.clone()).or_default() += k;
        }
        let mut memory = Memory::default();
        memory.set("buffer.xs", Value::Vec(g.xs.iter().map(|&x| Value::F32(x)).collect()));
        memory.set("buffer.ks", Value::Vec(g.ks.iter().map(|&x| Value::U32(x)).collect()));
        memory.set("uniform.n", Value::U32(g.n));
        memory.set("buffer.out", Value::Vec(vec![Value::F32(0.0); 8]));
        memory.set("buffer.cnt", Value::Vec(vec![Value::U32(0); 8]));
        memory.set("buffer.ys", Value::Vec(g.ys.iter().map(|&x| Value::F32(x)).collect()));
        let binds = bindings(&kernel.module)?;
        let original = match device.run(&g.source, &entry, &memory, &binds) {
            Ok(m) => m,
            Err(e) => {
                errors += 1;
                if failures.len() < 5 {
                    failures.push(format!("seed {seed}: original: {e}\n{}", g.source));
                }
                continue;
            }
        };
        // An unused binding is not read back: keep its initial content.
        let interp = Interp::new(&kernel, &chromosomes, memory.clone(), Invocation::default()).and_then(|mut it| it.run(&entry).map(|_| it.memory));
        let roots: Vec<Vec<String>> = chromosomes.iter().map(|c| fuller::homeotic::unfold(&c.folded)).collect::<Result<_, _>>()?;
        let rebuilt = round_trip(&kernel, &roots).and_then(|r| device.run(&r.wgsl, &entry, &memory, &binds));
        let mut notes = Vec::new();
        let mut program_contracted = 0usize;
        match interp {
            Err(e) => notes.push(format!("interp: {e}")),
            Ok(m) => {
                for loc in ["buffer.out", "buffer.cnt", "buffer.ys"] {
                    let (d, c) = lane_differences(m.get(loc).unwrap(), original.get(loc).unwrap(), Some(CONTRACTION_ULPS));
                    program_contracted += c;
                    if let Some((i, a, b)) = d.first() {
                        notes.push(format!("interp {loc}[{i}]: interp 0x{a:08x} ({}), device 0x{b:08x} ({}); {} lanes", f32::from_bits(*a), f32::from_bits(*b), d.len()));
                    }
                }
            }
        }
        let interp_bad = !notes.is_empty();
        let mut rebuilt_bad = false;
        match rebuilt {
            Err(e) => {
                rebuilt_bad = true;
                notes.push(format!("rebuilt: {e}"));
            }
            Ok(m) => {
                for loc in ["buffer.out", "buffer.cnt", "buffer.ys"] {
                    let (d, _) = lane_differences(m.get(loc).unwrap(), original.get(loc).unwrap(), None);
                    if let Some((i, a, b)) = d.first() {
                        rebuilt_bad = true;
                        notes.push(format!("rebuilt {loc}[{i}]: rebuilt 0x{a:08x}, original 0x{b:08x}; {} lanes", d.len()));
                    }
                }
            }
        }
        contracted_lanes += program_contracted;
        if program_contracted > 0 {
            contracted_programs += 1;
        }
        if interp_bad {
            interp_differs += 1;
        }
        if rebuilt_bad {
            rebuilt_differs += 1;
        }
        if notes.is_empty() {
            agree += 1;
        } else {
            all_notes.push(format!("seed {seed}: {}", notes.join("; ")));
            if failures.len() < 5 {
                failures.push(format!("seed {seed} (intended {:?}, shared {}, refused {:?}):\n  {}\n{}", g.intended, main.folded.filled, main.refused, notes.join("\n  "), g.source));
            }
        }
    }
    println!("\ngenerated: {n} programs from seed {seed0}: {agree} agree three ways, {interp_differs} interpreter differs, {rebuilt_differs} rebuilt differs, {errors} failed to run; {shared_programs} programs share; {contracted_lanes} lanes in {contracted_programs} programs within {CONTRACTION_ULPS} ulps (contraction)");
    for n in &all_notes {
        println!("  {n}");
    }
    println!("{:<24} {:>9} {:>9}   (reached = the decision's count; hoist, let_after_store and out_of_range are checked by the agreement, not counted)", "class", "intended", "reached");
    let mut classes: Vec<String> = intended.keys().map(|k| k.to_string()).chain(reached.keys().cloned()).collect();
    classes.sort();
    classes.dedup();
    for c in classes {
        println!("{:<24} {:>9} {:>9}", c, intended.get(c.as_str()).copied().unwrap_or(0), reached.get(&c).copied().unwrap_or(0));
    }
    if !failures.is_empty() {
        for f in &failures {
            println!("\ndifference:\n{f}");
        }
        return Err("generated programs differ".into());
    }
    Ok(())
}

fn hand_set(device: &Device, dir: &std::path::Path) -> Result<(), String> {
    let kernels = hand_kernels(dir)?;
    println!("{:<26} {:<10} {:<10} {:<10} {:<12}  what", "kernel", "original", "rebuilt", "interp", "shared/refused");
    let (mut ok, mut total) = (0usize, 0usize);
    for k in &kernels {
        total += 1;
        let entry = entry_name(&k.kernel)?;
        let mut memory = Memory::default();
        for (loc, v) in &k.expectation.inputs {
            memory.set(loc, v.clone());
        }
        let verdict = |r: Result<Memory, String>, on_device: bool| -> (String, Vec<String>) {
            match r {
                Ok(m) => {
                    let (d, known) = check(&m, &k.expectation, on_device);
                    let mut notes = d.clone();
                    notes.extend(known.iter().map(|n| format!("known deviation: {n}")));
                    (if !d.is_empty() { "DIFFERS".to_string() } else if !known.is_empty() { "ok*".to_string() } else { "ok".to_string() }, notes)
                }
                Err(e) => ("ERROR".to_string(), vec![e]),
            }
        };
        let binds = bindings(&k.kernel.module)?;
        let original = verdict(device.run(&k.source, &entry, &memory, &binds), true);
        let roots: Vec<Vec<String>> = k.chromosomes.iter().map(|c| fuller::homeotic::unfold(&c.folded)).collect::<Result<_, _>>()?;
        let rebuilt = verdict(round_trip(&k.kernel, &roots).and_then(|r| device.run(&r.wgsl, &entry, &memory, &binds)), true);
        let interp = if k.expectation.interpreter { verdict(interpret(k), false) } else { ("skipped".to_string(), Vec::new()) };
        let main = k.chromosomes.iter().find(|c| c.function == entry).ok_or("no entry chromosome")?;
        let refused: usize = main.refused.values().sum();
        let good = original.0.starts_with("ok") && rebuilt.0.starts_with("ok") && interp.0 != "DIFFERS" && interp.0 != "ERROR";
        if good {
            ok += 1;
        }
        println!("{:<26} {:<10} {:<10} {:<10} {:<12}  {}", k.name, original.0, rebuilt.0, interp.0, format!("{}/{}", main.folded.filled, refused), k.expectation.what);
        for (who, (_, notes)) in [("original", &original), ("rebuilt", &rebuilt), ("interp", &interp)] {
            for n in notes {
                println!("    {who}: {n}");
            }
        }
        if !main.refused.is_empty() {
            println!("    refused by reason: {:?}", main.refused);
        }
    }
    println!("\n{ok} of {total} hand-written kernels agree three ways with their expectation on {}", device.name);
    if ok == total {
        Ok(())
    } else {
        Err(format!("{} kernels differ", total - ok))
    }
}
