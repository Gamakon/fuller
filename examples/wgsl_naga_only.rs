//! The control for the rebuild: a kernel through naga's own front end and
//! writer with NO reading, folding or rebuilding in between. If this text
//! behaves differently from the shipped source on the device, the
//! difference is naga's writer (or what the engine's source rewrites expect
//! of the text), not the kingdom's rebuild.
//!
//!   cargo run --release --features wgsl --example wgsl_naga_only -- <in.wgsl[+more.wgsl]> <out.wgsl>

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        return Err("usage: wgsl_naga_only <in.wgsl[+more]> <out.wgsl>".into());
    };
    let mut src = String::new();
    for part in input.split('+') {
        src.push_str(&std::fs::read_to_string(part).map_err(|e| format!("read {part}: {e}"))?);
    }
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("validation: {}", e.emit_to_string(&src)))?;
    let text = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).map_err(|e| format!("write: {e}"))?;
    std::fs::write(output, &text).map_err(|e| format!("write {output}: {e}"))?;
    println!("{output}: {} bytes, {} functions, {} entry points", text.len(), module.functions.len(), module.entry_points.len());
    Ok(())
}
