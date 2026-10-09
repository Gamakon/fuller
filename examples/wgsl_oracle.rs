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
//!   cargo run --release --features gpu,wgsl --example wgsl_oracle -- [--dir kingdoms/wgsl/oracle]

use std::path::PathBuf;

use fuller::wgsl::oracle::{bindings, check, device::Device, entry_name, hand_kernels, interpret};
use fuller::wgsl::{round_trip, Memory};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1)).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("kingdoms/wgsl/oracle"));
    let device = Device::new()?;
    println!("device: {}", device.name);
    let kernels = hand_kernels(&dir)?;
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
