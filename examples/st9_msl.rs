//! ST9: the Metal source of a kernel exactly as wgpu 30 hands it to Apple's compiler on this machine
//! (`fuller::wgsl::target::metal_source`: the pipeline constants applied, MSL 4.0, bounds checks Restrict, naga's loop
//! bounding), so the shipped shader and a kernel of the kingdom can be compared line by line. CPU only.
//!
//!   cargo run --release --features gpu,wgsl --example st9_msl -- IN.wgsl OUT.metal [NAME=VALUE ...]
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (input, output) = match (args.first(), args.get(1)) {
        (Some(i), Some(o)) => (i, o),
        _ => return Err("st9_msl IN.wgsl OUT.metal [NAME=VALUE ...]".into()),
    };
    let mut constants: Vec<(String, f64)> = Vec::new();
    for a in &args[2..] {
        let (k, v) = a.split_once('=').ok_or_else(|| format!("{a}: want NAME=VALUE"))?;
        constants.push((k.to_string(), v.parse().map_err(|e| format!("{a}: {e}"))?));
    }
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let module = fuller::wgsl::naga_shape::parse_unscanned(&text)?;
    let info = fuller::wgsl::target::Target::current().validate(&module, &text)?;
    let refs: Vec<(&str, f64)> = constants.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let msl = fuller::wgsl::target::metal_source(&module, &info, "main", &refs)?;
    std::fs::write(output, &msl).map_err(|e| format!("{output}: {e}"))?;
    println!("{output}\t{} bytes\t{} lines\tloop_init {}", msl.len(), msl.lines().count(), msl.matches("loop_init").count());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
