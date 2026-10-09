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
use fuller::wgsl::{chromosome, read, rebuild, round_trip, ChromosomeOptions, Interp, Invocation, Memory, Value};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1)).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("kingdoms/wgsl/oracle"));
    let generate_n: usize = args.iter().position(|a| a == "--generate").and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(0);
    let seed0: u64 = args.iter().position(|a| a == "--seed").and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(1);
    if let Some(seed) = args.iter().position(|a| a == "--show").and_then(|i| args.get(i + 1)).and_then(|n| n.parse::<u64>().ok()) {
        let g = generate(seed);
        println!("// seed {seed}, intended {:?}, xs {:?}, ks {:?}, ys {:?}, n {}\n{}", g.intended, g.xs, g.ks, g.ys, g.n, g.source);
        return Ok(());
    }
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

/// How the interpreter's lanes relate to the device's. wgpu leaves Metal's
/// fast math on, so the device fuses `a * b + c` into one rounding, may
/// reassociate, treats signed zeros loosely and assumes no NaN or
/// infinity arises (`0.0 / x` folds to `0.0`): the spec's IEEE answer and
/// the device's differ in those cases, and the interpreter follows the
/// spec. The classes are reported, never folded into agreement silently.
#[derive(Default, Debug, Clone)]
struct Lanes {
    exact: usize,
    /// Finite, within 8 ulps: contraction.
    near: usize,
    /// Finite, within 64 ulps: reassociation along a longer value.
    far: usize,
    /// A NaN or an infinity on one side only, or both NaN (payloads differ):
    /// fast math's no-NaN assumption.
    non_finite: usize,
    /// `-0.0` against `0.0`.
    signed_zero: usize,
    /// Everything else: a real difference (lane, interp word, device word).
    differ: Vec<(usize, u32, u32)>,
}

const NEAR_ULPS: u32 = 8;
const FAR_ULPS: u32 = 64;

fn classify(interp: &Value, device: &Value) -> Lanes {
    let (wa, wb) = (fuller::wgsl::oracle::to_words(interp), fuller::wgsl::oracle::to_words(device));
    let mut l = Lanes::default();
    for i in 0..wa.len().max(wb.len()) {
        let (Some(&x), Some(&y)) = (wa.get(i), wb.get(i)) else {
            l.differ.push((i, wa.get(i).copied().unwrap_or(0), wb.get(i).copied().unwrap_or(0)));
            continue;
        };
        let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
        if x == y {
            l.exact += 1;
        } else if fx.is_nan() || fy.is_nan() || fx.is_infinite() || fy.is_infinite() {
            l.non_finite += 1;
        } else if fx == 0.0 && fy == 0.0 {
            l.signed_zero += 1;
        } else if let Some(u) = ulps(x, y) {
            if u <= NEAR_ULPS {
                l.near += 1;
            } else if u <= FAR_ULPS {
                l.far += 1;
            } else {
                l.differ.push((i, x, y));
            }
        } else {
            l.differ.push((i, x, y));
        }
    }
    l
}

/// Lanes where two device results differ, bit for bit (same compiler).
fn bit_differences(a: &Value, b: &Value) -> Vec<(usize, u32, u32)> {
    let (wa, wb) = (fuller::wgsl::oracle::to_words(a), fuller::wgsl::oracle::to_words(b));
    (0..wa.len().max(wb.len())).filter(|&i| wa.get(i) != wb.get(i)).map(|i| (i, wa.get(i).copied().unwrap_or(0), wb.get(i).copied().unwrap_or(0))).collect()
}

fn generated(device: &Device, n: usize, seed0: u64) -> Result<(), String> {
    let mut intended: BTreeMap<&str, usize> = BTreeMap::new();
    let mut reached: BTreeMap<String, usize> = BTreeMap::new();
    let (mut agree, mut interp_differs, mut rebuilt_differs, mut gate_refused, mut errors, mut shared_programs, mut non_finite_programs, mut non_finite_differs) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut lanes = Lanes::default();
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
        // The structural gate is reported on its own; the rebuilt text runs
        // on the device whether or not the gate accepted it.
        if let Err(e) = round_trip(&kernel, &roots) {
            gate_refused += 1;
            all_notes.push(format!("seed {seed}: gate: {}", e.lines().collect::<Vec<_>>().join(" | ")));
        }
        let rebuilt = rebuild(&kernel, &roots).and_then(|r| device.run(&r.wgsl, &entry, &memory, &binds));
        let mut notes = Vec::new();
        match interp {
            Err(e) => notes.push(format!("interp: {e}")),
            Ok(m) => {
                for loc in ["buffer.out", "buffer.cnt", "buffer.ys"] {
                    let l = classify(m.get(loc).unwrap(), original.get(loc).unwrap());
                    lanes.exact += l.exact;
                    lanes.near += l.near;
                    lanes.far += l.far;
                    lanes.non_finite += l.non_finite;
                    lanes.signed_zero += l.signed_zero;
                    if let Some((i, a, b)) = l.differ.first() {
                        notes.push(format!("interp {loc}[{i}]: interp 0x{a:08x} ({}), device 0x{b:08x} ({}); {} lanes", f32::from_bits(*a), f32::from_bits(*b), l.differ.len()));
                    }
                }
            }
        }
        // A program with a NaN or an infinity among its inputs: the device's
        // fast math may fold `s - s` or `s / s` to a constant where the spec
        // (and the interpreter) has NaN, and the result can flip a branch or
        // an index; counted apart, not held to agree.
        let non_finite = g.non_finite_inputs();
        if non_finite {
            non_finite_programs += 1;
            if !notes.is_empty() {
                non_finite_differs += 1;
                all_notes.push(format!("seed {seed} (non-finite inputs): {}", notes.join("; ")));
                notes.clear();
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
                    let d = bit_differences(m.get(loc).unwrap(), original.get(loc).unwrap());
                    if let Some((i, a, b)) = d.first() {
                        rebuilt_bad = true;
                        notes.push(format!("rebuilt {loc}[{i}]: rebuilt 0x{a:08x}, original 0x{b:08x}; {} lanes", d.len()));
                    }
                }
            }
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
    println!("\ngenerated: {n} programs from seed {seed0}: {agree} agree three ways, {interp_differs} interpreter differs, {rebuilt_differs} rebuilt differs on the device, {gate_refused} refused by the structural gate, {errors} failed to run; {shared_programs} programs share; {non_finite_programs} programs have non-finite inputs, the interpreter differs on {non_finite_differs} of them (fast math, not counted)");
    println!("interpreter lanes against the device: {} exact, {} within {NEAR_ULPS} ulps (contraction), {} within {FAR_ULPS} ulps, {} non-finite on one side (fast math), {} signed zero", lanes.exact, lanes.near, lanes.far, lanes.non_finite, lanes.signed_zero);
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
