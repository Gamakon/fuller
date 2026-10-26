//! The portability probe for Futhark's WebGPU backend.
//!
//! Futhark's experimental WebGPU backend (diku-dk/futhark PR #2140, branch
//! `webgpu`) emits WGSL. Nothing in that backend is wgpu-specific, so the WGSL
//! it emits *should* run on wgpu-native / Metal -- but that has never been
//! checked, and three things in the emitted text are known hazards:
//!
//!   * the prelude opens with an unconditional `enable f16;`, so the device
//!     must have `SHADER_F16`;
//!   * the integer runtime uses `alias i8 = i32; alias i16 = i32;` and
//!     `alias i64 = vec2<i32>;`, and `i8`/`i16`/`i64` are WGSL *reserved*
//!     words (Firefox's Tint rejects them; naga has `i64` in its builtin
//!     identifier list, so it may reject the redefinition too);
//!   * the workgroup size arrives as WGSL `override` (pipeline-overridable)
//!     constants rather than literals, because Futhark specialises the block
//!     size at run time.
//!
//! This probe answers all three, and the rest of the question "does a Futhark
//! WebGPU shader compile on this machine's device", in one run:
//!
//!   cargo run --release --features gpu,wgsl --example futhark_wgsl_probe -- \
//!       prelude.wgsl kernel.wgsl ...
//!
//! For each file it reports three verdicts:
//!
//!   naga     the front end parsed it (naga's own parser, no kingdom rules);
//!   valid    naga's validator accepted it, and it fits this device's limits
//!             (workgroup size, workgroup memory) -- the same limits the
//!             wgsl_true harness enforces;
//!   device   the driver accepted the shader module and built a compute
//!             pipeline for every compute entry point, once with the module's
//!             default constants and once with each override set explicitly.
//!
//! The last one is the verdict that matters: naga accepting a module proves
//! nothing about whether Metal will build a pipeline from it.

use std::borrow::Cow;
use std::time::Instant;

/// The default value of a pipeline-overridable constant, as a f64, or None if
/// the module gives it no default.
fn override_default(module: &naga::Module, ov: &naga::Override) -> Option<f64> {
    let init = ov.init?;
    match module.global_expressions[init] {
        naga::Expression::Literal(lit) => Some(match lit {
            naga::Literal::F64(v) => v,
            naga::Literal::F32(v) => f64::from(v),
            naga::Literal::F16(v) => f64::from(v),
            naga::Literal::U32(v) => f64::from(v),
            naga::Literal::U16(v) => f64::from(v),
            naga::Literal::U64(v) => v as f64,
            naga::Literal::I32(v) => f64::from(v),
            naga::Literal::I16(v) => f64::from(v),
            naga::Literal::I64(v) => v as f64,
            naga::Literal::Bool(v) => f64::from(u8::from(v)),
            naga::Literal::AbstractInt(v) => v as f64,
            naga::Literal::AbstractFloat(v) => v,
        }),
        _ => None,
    }
}

/// The name a pipeline constant is keyed by: the `@id` if the override has
/// one, else the identifier.
fn override_key(ov: &naga::Override) -> Option<String> {
    match ov.id {
        Some(id) => Some(id.to_string()),
        None => ov.name.clone(),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: futhark_wgsl_probe [--override name=value ...] <file.wgsl> ...".into());
    }
    // `--override name=value` forces a pipeline constant, which is how the
    // workgroup size gets its value: Futhark emits `override x: i32 = 0;`
    // and the host sets it per dispatch.
    let mut forced: Vec<(String, f64)> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--override" {
            let v = it.next().ok_or("--override needs a name=value")?;
            let (k, val) = v.split_once('=').ok_or("--override needs a name=value")?;
            let val: f64 = val.parse().map_err(|_| "--override needs a number")?;
            forced.push((k.to_string(), val));
        } else {
            files.push(a);
        }
    }
    if files.is_empty() {
        return Err("usage: futhark_wgsl_probe [--override name=value ...] <file.wgsl> ...".into());
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    println!("device: {} ({:?})", info.name, info.backend);
    println!(
        "features: SHADER_F16={}  limits: max_storage_buffers_per_stage={}  max_compute_invocations_per_workgroup={}",
        adapter.features().contains(wgpu::Features::SHADER_F16),
        adapter.limits().max_storage_buffers_per_shader_stage,
        adapter.limits().max_compute_invocations_per_workgroup
    );
    let required = adapter.features() & wgpu::Features::SHADER_F16;
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("futhark-wgsl-probe"),
        required_features: required,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("request_device: {e}"))?;

    let target = fuller::wgsl::target::Target::current();
    let mut files_ok = 0usize;
    let mut files_bad = 0usize;

    for path in &files {
        let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        println!("\n=== {path} ({} bytes)", src.len());

        // 1. naga's front end, with no kingdom rules in the way.
        let module = match fuller::wgsl::naga_shape::parse_unscanned(&src) {
            Ok(m) => {
                println!("  naga     OK  ({} functions, {} entry points, {} overrides)", m.functions.len(), m.entry_points.len(), m.overrides.len());
                m
            }
            Err(e) => {
                println!("  naga     FAIL\n{e}");
                files_bad += 1;
                continue;
            }
        };

        // 2. naga's validator plus this device's limits.
        match target.validate(&module, &src) {
            Ok(_) => println!("  valid    OK"),
            Err(e) => {
                println!("  valid    FAIL\n{e}");
                files_bad += 1;
                continue;
            }
        }

        // The overrides, so the report says what they are.
        let mut keys: Vec<(String, f64)> = Vec::new();
        for ov in module.overrides.iter() {
            if let Some(k) = override_key(ov.1) {
                let d = override_default(&module, ov.1).unwrap_or(64.0);
                keys.push((k, d));
            }
        }
        // Futhark emits `override x: i32 = 0;` and lets the host set the
        // value per dispatch. A workgroup size of zero is not a legal
        // pipeline, so the "overrides" run replaces a zero default with a
        // positive one -- what the host does -- unless the caller forced a
        // value.
        let tuned: Vec<(String, f64)> = keys
            .iter()
            .map(|(k, v)| {
                let forced_value = forced.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
                let value = forced_value.unwrap_or(if *v <= 0.0 { 64.0 } else { *v });
                (k.clone(), value)
            })
            .collect();
        if !keys.is_empty() {
            let listed: Vec<String> = keys.iter().map(|(k, v)| format!("{k}={v}")).collect();
            println!("  overrides: {}", listed.join(", "));
        }

        // 3. The driver: a shader module and a pipeline per compute entry
        //    point, with the module's own defaults and then with every
        //    override set explicitly (what Futhark's host does at run time).
        let entries: Vec<String> = module
            .entry_points
            .iter()
            .filter(|ep| ep.stage == naga::ShaderStage::Compute)
            .map(|ep| ep.name.clone())
            .collect();
        if entries.is_empty() {
            println!("  device   (no compute entry points)");
            files_ok += 1;
            continue;
        }
        for entry in &entries {
            let mut verdicts: Vec<String> = Vec::new();
            for constants in [vec![], tuned.clone()] {
                let label = if constants.is_empty() {
                    "defaults".to_string()
                } else {
                    format!(
                        "overrides[{}]",
                        constants
                            .iter()
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                };
                let t = Instant::now();
                let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(path),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&src)),
                });
                let pairs: Vec<(&str, f64)> =
                    constants.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(path),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &pairs,
                        ..Default::default()
                    },
                    cache: None,
                });
                let verdict = match pollster::block_on(scope.pop()) {
                    None => format!("OK ({:.1} ms)", t.elapsed().as_secs_f64() * 1e3),
                    Some(e) => format!("FAIL: {e}"),
                };
                verdicts.push(format!("{label}: {verdict}"));
            }
            let good = verdicts.iter().all(|v| v.contains("OK"));
            if good {
                files_ok += 1;
            } else {
                files_bad += 1;
            }
            println!("  device   {entry:<24} {}", verdicts.join("  |  "));
        }
    }

    println!("\n{files_ok} file(s) fully accepted, {files_bad} rejected.");
    Ok(())
}
