//! The oracle (`docs/PLAN_wgsl_lineage.md` §4): kernels with inputs and
//! expected outputs, run through the three implementations and compared.
//!
//! A kernel's expectation file (`<name>.expect.json` beside `<name>.wgsl`,
//! `kingdoms/wgsl/oracle/`) is written BY HAND, never captured from a run:
//! the inputs of every binding (its initial content), the expected content
//! of the read-write bindings afterwards, whether the interpreter can run
//! it (one invocation) and one line saying what it tests. The original
//! kernel on the device is the authority; the interpreter on the folded
//! chromosome and the rebuilt text on the device are checked against it
//! and against the expectation. Values are compared as bits; a lane whose
//! expectation is `"nan"` accepts any NaN (WGSL fixes neither sign nor
//! payload). Where this platform is KNOWN to deviate from the spec, the
//! expectation lists the lanes under `device_deviates` and the device
//! comparison reports them as known rather than as a difference, while the
//! interpreter is still held to the spec. Measured 2026-10-09 on Apple M3
//! Max (Metal, naga 0.20): integer `x / 0` gives `-1` (all ones) and
//! `x % 0` gives `x`, where WGSL says `x` and `0`; naga's MSL output does
//! not guard integer division.
//!
//! The device parts compile under `gpu`; reading expectations and running
//! the interpreter need only `wgsl`.

use std::collections::BTreeMap;
use std::path::Path;

use naga::{AddressSpace, Module, TypeInner};

use super::chromosome::{chromosome, ChromosomeOptions, WgslChromosome};
use super::interp::{Interp, Invocation, Memory, Value};
use super::reader::{read, Kernel};
use super::versions;

/// What a binding is, from the module's globals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    /// The reader's location name (`buffer.xs`, `uniform.n`).
    pub location: String,
    pub uniform: bool,
    pub read_write: bool,
    /// Element kind: `f32`, `u32` or `i32`; and whether it is an array.
    pub elem: &'static str,
    pub array: bool,
}

/// Every bound global of `module`, in (group, binding) order.
pub fn bindings(module: &Module) -> Result<Vec<Binding>, String> {
    let mut out = Vec::new();
    for (h, g) in module.global_variables.iter() {
        let Some(rb) = &g.binding else { continue };
        let (uniform, read_write) = match g.space {
            AddressSpace::Uniform => (true, false),
            AddressSpace::Storage { access } => (false, access.contains(naga::StorageAccess::STORE)),
            other => return Err(format!("oracle: binding in {other:?} space is not supported")),
        };
        let (inner, array) = match &module.types[g.ty].inner {
            TypeInner::Array { base, .. } => (&module.types[*base].inner, true),
            other => (other, false),
        };
        let scalar = match inner {
            TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } => *s,
            other => return Err(format!("oracle: binding of type {other:?} is not supported")),
        };
        let elem = match (scalar.kind, scalar.width) {
            (naga::ScalarKind::Float, 4) => "f32",
            (naga::ScalarKind::Uint, 4) => "u32",
            (naga::ScalarKind::Sint, 4) => "i32",
            other => return Err(format!("oracle: scalar {other:?} is not supported")),
        };
        out.push(Binding { group: rb.group, binding: rb.binding, location: versions::global_location(module, h), uniform, read_write, elem, array });
    }
    out.sort_by_key(|b| (b.group, b.binding));
    Ok(out)
}

/// A value as 32-bit words, as the device holds it.
pub fn to_words(v: &Value) -> Vec<u32> {
    match v {
        Value::F32(x) => vec![x.to_bits()],
        Value::U32(x) => vec![*x],
        Value::I32(x) => vec![*x as u32],
        Value::Bool(b) => vec![u32::from(*b)],
        Value::Vec(items) => items.iter().flat_map(to_words).collect(),
    }
}

/// Words back into a value of the binding's element kind.
pub fn from_words(words: &[u32], elem: &str, array: bool) -> Result<Value, String> {
    let one = |w: u32| -> Result<Value, String> {
        Ok(match elem {
            "f32" => Value::F32(f32::from_bits(w)),
            "u32" => Value::U32(w),
            "i32" => Value::I32(w as i32),
            other => return Err(format!("oracle: element kind {other}")),
        })
    };
    if array {
        Ok(Value::Vec(words.iter().map(|&w| one(w)).collect::<Result<_, _>>()?))
    } else {
        one(*words.first().ok_or("oracle: no word for a scalar binding")?)
    }
}

/// One lane of an expectation: a number, `"nan"`, `"inf"`, `"-inf"`, or a
/// hexadecimal bit pattern `"0x…"`.
#[derive(Debug, Clone, PartialEq)]
pub enum Lane {
    Exact(u32),
    AnyNan,
}

fn lane(elem: &str, v: &serde_json::Value) -> Result<Lane, String> {
    match v {
        serde_json::Value::String(s) => match s.as_str() {
            "nan" => Ok(Lane::AnyNan),
            "inf" => Ok(Lane::Exact(f32::INFINITY.to_bits())),
            "-inf" => Ok(Lane::Exact(f32::NEG_INFINITY.to_bits())),
            hex if hex.starts_with("0x") => u32::from_str_radix(&hex[2..], 16).map(Lane::Exact).map_err(|e| format!("oracle: {hex}: {e}")),
            other => Err(format!("oracle: lane {other:?}")),
        },
        serde_json::Value::Number(n) => Ok(Lane::Exact(match elem {
            "f32" => (n.as_f64().ok_or("oracle: lane is not a number")? as f32).to_bits(),
            "u32" => n.as_u64().ok_or_else(|| format!("oracle: {n} is not a u32"))? as u32,
            "i32" => n.as_i64().ok_or_else(|| format!("oracle: {n} is not an i32"))? as i32 as u32,
            other => return Err(format!("oracle: element kind {other}")),
        })),
        other => Err(format!("oracle: lane {other}")),
    }
}

/// A hand-written expectation.
#[derive(Debug, Clone)]
pub struct Expectation {
    pub what: String,
    pub interpreter: bool,
    /// Initial content per location.
    pub inputs: BTreeMap<String, Value>,
    /// Expected content per read-write location, as lanes.
    pub expected: BTreeMap<String, Vec<Lane>>,
    /// Lanes per location where the device is known to deviate from the
    /// spec (the interpreter is still checked against the spec there).
    pub device_deviates: BTreeMap<String, Vec<usize>>,
}

fn values(obj: &serde_json::Value) -> Result<BTreeMap<String, (String, Vec<serde_json::Value>)>, String> {
    let mut out = BTreeMap::new();
    for (loc, spec) in obj.as_object().ok_or("oracle: not an object")? {
        let (elem, lanes) = spec.as_object().and_then(|o| o.iter().next()).ok_or_else(|| format!("oracle: {loc} needs {{\"f32\": [...]}}"))?;
        let lanes = lanes.as_array().ok_or_else(|| format!("oracle: {loc} lanes"))?.clone();
        out.insert(loc.clone(), (elem.clone(), lanes));
    }
    Ok(out)
}

pub fn read_expectation(path: &Path) -> Result<Expectation, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let what = json["what"].as_str().unwrap_or("").to_string();
    let interpreter = json["interpreter"].as_bool().unwrap_or(false);
    let mut inputs = BTreeMap::new();
    for (loc, (elem, lanes)) in values(&json["inputs"])? {
        let words: Vec<u32> = lanes.iter().map(|l| lane(&elem, l)).collect::<Result<Vec<_>, _>>()?.into_iter().map(|l| match l {
            Lane::Exact(w) => w,
            Lane::AnyNan => f32::NAN.to_bits(),
        }).collect();
        let array = !loc.starts_with("uniform.") || words.len() > 1;
        inputs.insert(loc, from_words(&words, &elem, array)?);
    }
    let mut expected = BTreeMap::new();
    for (loc, (elem, lanes)) in values(&json["expected"])? {
        expected.insert(loc, lanes.iter().map(|l| lane(&elem, l)).collect::<Result<_, _>>()?);
    }
    let mut device_deviates = BTreeMap::new();
    if let Some(obj) = json.get("device_deviates").and_then(|v| v.as_object()) {
        for (loc, lanes) in obj {
            let lanes: Vec<usize> = lanes.as_array().ok_or_else(|| format!("oracle: device_deviates {loc}"))?.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect();
            device_deviates.insert(loc.clone(), lanes);
        }
    }
    Ok(Expectation { what, interpreter, inputs, expected, device_deviates })
}

/// The lanes where `got` differs from `want` (index, got word, want).
pub fn differences(got: &Value, want: &[Lane]) -> Vec<(usize, u32, Lane)> {
    let words = to_words(got);
    let mut out = Vec::new();
    for (i, w) in want.iter().enumerate() {
        let g = words.get(i).copied();
        let ok = match (g, w) {
            (Some(g), Lane::Exact(x)) => g == *x,
            (Some(g), Lane::AnyNan) => f32::from_bits(g).is_nan(),
            (None, _) => false,
        };
        if !ok {
            out.push((i, g.unwrap_or(0), w.clone()));
        }
    }
    if words.len() != want.len() {
        out.push((want.len(), words.len() as u32, Lane::Exact(want.len() as u32)));
    }
    out
}

/// One hand-written kernel: its source, read kernel, chromosomes and
/// expectation.
pub struct HandKernel {
    pub name: String,
    pub source: String,
    pub kernel: Kernel,
    pub chromosomes: Vec<WgslChromosome>,
    pub expectation: Expectation,
}

/// Every `<name>.wgsl` with a `<name>.expect.json` in `dir`, read and folded.
pub fn hand_kernels(dir: &Path) -> Result<Vec<HandKernel>, String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".wgsl")).map(str::to_string))
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let source = std::fs::read_to_string(dir.join(format!("{name}.wgsl"))).map_err(|e| format!("{name}: {e}"))?;
        let expectation = read_expectation(&dir.join(format!("{name}.expect.json")))?;
        let kernel = read(&source).map_err(|e| format!("{name}: {e}"))?;
        let chromosomes = kernel.functions.iter().map(|f| chromosome(f, &ChromosomeOptions::default())).collect::<Result<_, _>>().map_err(|e| format!("{name}: {e}"))?;
        out.push(HandKernel { name, source, kernel, chromosomes, expectation });
    }
    Ok(out)
}

/// The entry point's name (the one entry point of an oracle kernel).
pub fn entry_name(kernel: &Kernel) -> Result<String, String> {
    let eps: Vec<&String> = kernel.functions.iter().filter(|f| f.entry_point).map(|f| &f.name).collect();
    match eps.as_slice() {
        [one] => Ok((*one).clone()),
        _ => Err(format!("oracle: {} entry points, want 1", eps.len())),
    }
}

/// Run the interpreter on a hand kernel: the memory after, or the error.
pub fn interpret(k: &HandKernel) -> Result<Memory, String> {
    let mut memory = Memory::default();
    for (loc, v) in &k.expectation.inputs {
        memory.set(loc, v.clone());
    }
    let entry = entry_name(&k.kernel)?;
    let mut it = Interp::new(&k.kernel, &k.chromosomes, memory, Invocation::default())?;
    it.run(&entry)?;
    Ok(it.memory)
}

/// Compare a memory with the expectation: the differences as text (the
/// first lane per location), and separately the differences that sit on
/// lanes the device is known to deviate on, when `device` is true.
pub fn check(memory: &Memory, expectation: &Expectation, device: bool) -> (Vec<String>, Vec<String>) {
    let mut out = Vec::new();
    let mut known = Vec::new();
    for (loc, want) in &expectation.expected {
        match memory.get(loc) {
            None => out.push(format!("{loc}: missing")),
            Some(got) => {
                let d = differences(got, want);
                let known_lanes = expectation.device_deviates.get(loc).cloned().unwrap_or_default();
                let (expected_dev, real): (Vec<_>, Vec<_>) = d.into_iter().partition(|(i, _, _)| device && known_lanes.contains(i));
                if let Some((i, g, w)) = real.first() {
                    out.push(format!("{loc}[{i}]: got 0x{g:08x} ({}), want {w:?} ({} lanes differ)", f32::from_bits(*g), real.len()));
                }
                if let Some((i, g, w)) = expected_dev.first() {
                    known.push(format!("{loc}[{i}]: device gives 0x{g:08x}, the spec {w:?} ({} known lanes)", expected_dev.len()));
                }
            }
        }
    }
    (out, known)
}

#[cfg(feature = "gpu")]
pub mod device {
    //! The device side: compile a kernel, bind its buffers from a `Memory`,
    //! dispatch one workgroup, read every read-write binding back.

    use std::borrow::Cow;

    use super::{bindings, from_words, to_words, Binding};
    use crate::wgsl::interp::{Memory, Value};

    /// A compiled kernel with its bind group and the buffers (binding,
    /// buffer, words) behind it.
    type Prepared = (wgpu::ComputePipeline, wgpu::BindGroup, Vec<(Binding, wgpu::Buffer, usize)>);

    pub struct Device {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub name: String,
    }

    impl Device {
        pub fn new() -> Result<Device, String> {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok_or("no adapter")?;
            let name = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor { label: Some("wgsl-oracle"), required_features: wgpu::Features::empty(), required_limits: adapter.limits() },
                None,
            ))
            .map_err(|e| format!("request_device: {e}"))?;
            Ok(Device { device, queue, name })
        }

        /// Time `source`: compile and bind as [`Device::run`] does, then
        /// `repeats` submissions of `dispatches` workgroups each, the wall
        /// clock around submit-and-wait; the MEDIAN milliseconds per
        /// submission and the device's MINIMUM. Outputs are not read back;
        /// every workgroup runs the same invocation, which is idempotent for
        /// the oracle's kernels. At one invocation per workgroup this measures
        /// dispatch overhead, not the kernel; a real workload needs the
        /// engine's buffers (phylu's `kernel_time`).
        pub fn time(&self, source: &str, entry: &str, memory: &Memory, original: &[Binding], dispatches: u32, repeats: usize) -> Result<(f64, f64), String> {
            let (pipeline, bind, buffers) = self.prepare(source, entry, memory, original)?;
            let mut times = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(dispatches, 1, 1);
                }
                let t = std::time::Instant::now();
                self.queue.submit(Some(enc.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            for (_, buf, _) in &buffers {
                buf.destroy();
            }
            self.device.poll(wgpu::Maintain::Poll);
            times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            Ok((times[times.len() / 2], times[0]))
        }

        /// Dispatch overhead (`docs/PLAN_population_dag.md` task 2): one
        /// submission holding `passes` compute passes, each one workgroup of
        /// `source` (`variants` pipelines compiled from it and cycled, so a
        /// pipeline switch is paid every pass as banded kernels would pay
        /// it); the median and minimum wall-clock milliseconds per
        /// submission over `repeats`.
        pub fn time_passes(&self, source: &str, entry: &str, memory: &Memory, original: &[Binding], passes: usize, variants: usize, repeats: usize) -> Result<(f64, f64), String> {
            let mut prepared = Vec::with_capacity(variants.max(1));
            for _ in 0..variants.max(1) {
                prepared.push(self.prepare(source, entry, memory, original)?);
            }
            let mut times = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
                for i in 0..passes {
                    let (pipeline, bind, _) = &prepared[i % prepared.len()];
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                let t = std::time::Instant::now();
                self.queue.submit(Some(enc.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            for (_, _, buffers) in &prepared {
                for (_, buf, _) in buffers {
                    buf.destroy();
                }
            }
            self.device.poll(wgpu::Maintain::Poll);
            times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            Ok((times[times.len() / 2], times[0]))
        }

        /// Compile, bind and upload: the pipeline, its bind group and the
        /// buffers (binding, buffer, words).
        fn prepare(&self, source: &str, entry: &str, memory: &Memory, original: &[Binding]) -> Result<Prepared, String> {
            let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| format!("validation: {}", e.emit_to_string(source)))?;
            let own: Vec<Binding> = bindings(&module)?;
            if own.len() != original.len() {
                return Err(format!("oracle: {} bindings, the original has {}", own.len(), original.len()));
            }
            // wgpu's automatic layout holds only the globals the entry point
            // USES; an unused binding must not be bound.
            let ep = module.entry_points.iter().position(|e| e.name == entry).ok_or_else(|| format!("oracle: no entry point {entry}"))?;
            let fi = info.get_entry_point(ep);
            let used: std::collections::BTreeSet<(u32, u32)> = module
                .global_variables
                .iter()
                .filter(|(h, g)| g.binding.is_some() && !fi[*h].is_empty())
                .map(|(_, g)| g.binding.as_ref().map(|b| (b.group, b.binding)).expect("filtered"))
                .collect();
            let mut binds = Vec::with_capacity(own.len());
            for (a, b) in own.iter().zip(original) {
                if (a.group, a.binding, a.uniform, a.read_write, a.elem, a.array) != (b.group, b.binding, b.uniform, b.read_write, b.elem, b.array) {
                    return Err(format!("oracle: binding {}/{} differs from the original: {a:?} vs {b:?}", a.group, a.binding));
                }
                if used.contains(&(b.group, b.binding)) {
                    binds.push(b.clone());
                }
            }
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(entry), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)) });
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(format!("shader module: {e}"));
            }
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: None, module: &shader, entry_point: entry, compilation_options: Default::default() });
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(format!("pipeline {entry}: {e}"));
            }
            let mut buffers = Vec::new();
            let mut entries = Vec::new();
            for b in &binds {
                if b.group != 0 {
                    return Err(format!("oracle: binding group {} is not supported (one group)", b.group));
                }
                let v = memory.get(&b.location).ok_or_else(|| format!("oracle: memory has no {} for binding {}", b.location, b.binding))?;
                let mut words = to_words(v);
                if b.uniform {
                    while !words.len().is_multiple_of(4) {
                        words.push(0);
                    }
                }
                if words.is_empty() {
                    return Err(format!("oracle: {} is empty", b.location));
                }
                let usage = if b.uniform { wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST } else { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC };
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(&b.location), size: (words.len() * 4) as u64, usage, mapped_at_creation: false });
                self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&words));
                buffers.push((b.clone(), buf, words.len()));
            }
            for (b, buf, _) in &buffers {
                entries.push(wgpu::BindGroupEntry { binding: b.binding, resource: buf.as_entire_binding() });
            }
            let layout = pipeline.get_bind_group_layout(0);
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(entry), layout: &layout, entries: &entries });
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(format!("bind group: {e}"));
            }
            Ok((pipeline, bind, buffers))
        }

        /// Compile `source`, bind every binding from `memory` (its initial
        /// content), dispatch `(1, 1, 1)` on `entry`, and return the memory
        /// with every read-write binding replaced by what the device wrote.
        /// `original` names the bindings (a rebuilt text's writer may rename
        /// a global, `out2` → `out2_`; the (group, binding) slots are the
        /// identity and must agree in kind).
        pub fn run(&self, source: &str, entry: &str, memory: &Memory, original: &[Binding]) -> Result<Memory, String> {
            let (pipeline, bind, buffers) = self.prepare(source, entry, memory, original)?;
            let reads: Vec<(usize, wgpu::Buffer)> = buffers
                .iter()
                .enumerate()
                .filter(|(_, (b, _, _))| b.read_write)
                .map(|(i, (b, _, n))| {
                    let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(&format!("{}-read", b.location)), size: (*n * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
                    (i, read)
                })
                .collect();
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            for (i, read) in &reads {
                enc.copy_buffer_to_buffer(&buffers[*i].1, 0, read, 0, (buffers[*i].2 * 4) as u64);
            }
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            self.queue.submit(Some(enc.finish()));
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(format!("submit: {e}"));
            }
            let mut out = memory.clone();
            for (i, read) in &reads {
                let slice = read.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                self.device.poll(wgpu::Maintain::Wait);
                rx.recv().map_err(|e| format!("map_async channel: {e}"))?.map_err(|e| format!("map_async: {e}"))?;
                let words: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
                read.unmap();
                let b = &buffers[*i].0;
                let v: Value = from_words(&words, b.elem, b.array)?;
                out.set(&b.location, v);
            }
            for (_, buf, _) in &buffers {
                buf.destroy();
            }
            for (_, read) in &reads {
                read.destroy();
            }
            self.device.poll(wgpu::Maintain::Poll);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kingdoms/wgsl/oracle")
    }

    #[test]
    fn the_hand_written_kernels_read_fold_and_the_interpreter_meets_every_expectation() {
        let kernels = hand_kernels(&oracle_dir()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(kernels.len(), 10);
        let mut ran = 0;
        for k in &kernels {
            assert!(!k.expectation.what.is_empty(), "{}: says what it tests", k.name);
            if !k.expectation.interpreter {
                continue;
            }
            let memory = interpret(k).unwrap_or_else(|e| panic!("{}: {e}", k.name));
            let (diffs, known) = check(&memory, &k.expectation, false);
            assert!(diffs.is_empty() && known.is_empty(), "{}: {diffs:?} {known:?}", k.name);
            ran += 1;
        }
        assert_eq!(ran, 9, "one kernel is device-only");
    }

    #[test]
    fn the_hand_written_set_reaches_the_refusal_classes_and_shares_where_it_may() {
        let kernels = hand_kernels(&oracle_dir()).unwrap();
        let by_name = |n: &str| kernels.iter().find(|k| k.name == n).unwrap();
        let main_of = |k: &HandKernel| k.chromosomes.iter().find(|c| c.function == entry_name(&k.kernel).unwrap()).unwrap().clone();
        let c = main_of(by_name("adv_loop_store"));
        assert!(c.refused.get("lineage_differs").copied().unwrap_or(0) >= 1, "{:?}", c.refused);
        let c = main_of(by_name("adv_dynamic_index_repeat"));
        assert!(c.refused.get("lineage_differs").copied().unwrap_or(0) >= 1, "{:?}", c.refused);
        assert!(c.folded.filled >= 1, "ys[j] * ys[j] + ys[j] shares within a version");
        let c = main_of(by_name("adv_branch_let"));
        assert!(c.folded.filled >= 1, "v * v + 1.0 shares inside the arm: {:?}", c.refused);
        assert!(c.placements[0].as_ref().unwrap().path.len() >= 3, "placed inside the arm: {:?}", c.placements[0]);
    }

    #[test]
    fn bindings_and_words_round_trip() {
        let src = r#"
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(1) fn main() { out[0] = i32(xs[0]) + i32(n); }
"#;
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let b = bindings(&module).unwrap();
        assert_eq!(b.iter().map(|b| (b.location.as_str(), b.uniform, b.read_write, b.elem, b.array)).collect::<Vec<_>>(), vec![("buffer.xs", false, false, "f32", true), ("uniform.n", true, false, "u32", false), ("buffer.out", false, true, "i32", true)]);
        let v = Value::Vec(vec![Value::F32(-0.0), Value::F32(f32::NAN)]);
        let w = to_words(&v);
        assert_eq!(w[0], 0x8000_0000);
        assert!(from_words(&w, "f32", true).unwrap().bits_eq(&v));
        assert_eq!(differences(&Value::Vec(vec![Value::F32(1.0), Value::F32(f32::NAN)]), &[Lane::Exact(1.0f32.to_bits()), Lane::AnyNan]), vec![]);
        assert_eq!(differences(&Value::Vec(vec![Value::F32(-0.0)]), &[Lane::Exact(0)]).len(), 1, "-0.0 is not 0.0 in bits");
    }
}
