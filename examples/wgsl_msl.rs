//! A WGSL kernel as naga 0.20 writes it for Metal (the translation wgpu
//! hands the Metal compiler): to read what an operation becomes there.
//! CPU only.
//!
//! `cargo run --release --features gpu,wgsl --example wgsl_msl -- <kernel.wgsl>`

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: wgsl_msl <kernel.wgsl>")?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let module = fuller::wgsl::naga_shape::parse(&text)?;
    // (The recorded target's capabilities and subgroup operations: no device.)
    let info = fuller::wgsl::target::Target::recorded().validate(&module, &text)?;
    // (Metal 2.4, as wgpu asks on this machine, and bindings as written.)
    let options = naga::back::msl::Options { lang_version: (2, 4), fake_missing_bindings: true, ..Default::default() };
    let (msl, _) = naga::back::msl::write_string(&module, &info, &options, &naga::back::msl::PipelineOptions::default()).map_err(|e| format!("{e:?}"))?;
    println!("{msl}");
    Ok(())
}
