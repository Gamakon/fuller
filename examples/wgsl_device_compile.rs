//! Step 5, first half: do the original kernel and its REBUILT text both
//! compile on this machine's device? For every file pair, every entry point:
//! create the shader module and a compute pipeline (auto layout) on the wgpu
//! device, under an error scope, and report the driver's verdict. No
//! dispatch, no buffers, so there is no result to invent: the question is
//! only whether the backend's shader compiler accepts what naga validated.
//!
//!   cargo run --release --features gpu,wgsl --example wgsl_device_compile -- <original.wgsl[+more.wgsl]>=<rebuilt.wgsl> ...
//!
//! Each argument is `original=rebuilt`; an original assembled from several
//! files is `a.wgsl+b.wgsl`, concatenated as the engine does.

use std::borrow::Cow;
use std::time::Instant;

fn read_concat(spec: &str) -> Result<String, String> {
    let mut src = String::new();
    for part in spec.split('+') {
        src.push_str(&std::fs::read_to_string(part).map_err(|e| format!("read {part}: {e}"))?);
    }
    Ok(src)
}

fn entry_points(src: &str) -> Result<Vec<String>, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    Ok(module.entry_points.iter().map(|e| e.name.clone()).collect())
}

/// Compile `src` for `entry` on `device`; `Ok(ms)` or the driver's message.
fn compile(device: &wgpu::Device, label: &str, src: &str, entry: &str) -> Result<f64, String> {
    let t = Instant::now();
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });
    if let Some(e) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!("shader module: {e}"));
    }
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: entry,
        compilation_options: Default::default(),
    });
    if let Some(e) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!("pipeline {entry}: {e}"));
    }
    Ok(t.elapsed().as_secs_f64() * 1e3)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: wgsl_device_compile <original[+more]>=<rebuilt> ...".into());
    }
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok_or("no adapter")?;
    let info = adapter.get_info();
    println!("device: {} ({:?})", info.name, info.backend);
    // The adapter's own limits: these kernels bind up to 21 storage buffers,
    // past the default 8, and the engine requests the adapter's limits too.
    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("wgsl-device-compile"), required_features: wgpu::Features::empty(), required_limits: adapter.limits() },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;
    println!("{:<28} {:<20} {:>12} {:>12}  verdict", "kernel", "entry point", "original ms", "rebuilt ms");
    let (mut pairs, mut ok) = (0usize, 0usize);
    for arg in &args {
        let (orig_spec, rebuilt_path) = arg.split_once('=').ok_or_else(|| format!("{arg}: expected original=rebuilt"))?;
        let orig = read_concat(orig_spec)?;
        let rebuilt = std::fs::read_to_string(rebuilt_path).map_err(|e| format!("read {rebuilt_path}: {e}"))?;
        let name = std::path::Path::new(rebuilt_path).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        let entries = entry_points(&orig)?;
        let rebuilt_entries = entry_points(&rebuilt)?;
        if entries != rebuilt_entries {
            println!("{name:<28} entry points differ: {entries:?} vs {rebuilt_entries:?}");
        }
        for entry in &entries {
            pairs += 1;
            let a = compile(&device, &format!("{name}:original:{entry}"), &orig, entry);
            let b = compile(&device, &format!("{name}:rebuilt:{entry}"), &rebuilt, entry);
            let verdict = match (&a, &b) {
                (Ok(_), Ok(_)) => {
                    ok += 1;
                    "both compile".to_string()
                }
                (Ok(_), Err(e)) => format!("REBUILT FAILS: {e}"),
                (Err(e), Ok(_)) => format!("ORIGINAL FAILS: {e}"),
                (Err(e), Err(f)) => format!("BOTH FAIL: {e} / {f}"),
            };
            let ms = |r: &Result<f64, String>| r.as_ref().map(|m| format!("{m:.1}")).unwrap_or_else(|_| "-".into());
            println!("{name:<28} {entry:<20} {:>12} {:>12}  {verdict}", ms(&a), ms(&b));
        }
    }
    println!("\n{ok} of {pairs} (kernel, entry point) pairs compile both ways on {}", info.name);
    Ok(())
}
