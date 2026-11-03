//! THE IN-PROCESS EVALUATION HOST: the batch driver of `fuller::futhark_batch`, in this process, on one device.
//!
//! A generation's programs are one generated module, compiled by `rustc` into a dylib that links THIS runtime as a
//! dylib (so the module, the host and wgpu are one copy in the process), and loaded with `dlopen`. [`run`] executes
//! the batch driver's commands on it and writes the batch driver's results records, byte for byte the format
//! `futhark_batch` parses:
//!
//! * commands, one a line: `q ENTRY` (the quality entry on every stored case), `t ENTRY RUNS` (`RUNS + 1` runs on the
//!   timing input, the first a warm-up, each run's GPU nanoseconds), `o ENTRY` (once on the timing input, the
//!   output), `k ENTRY CASE START N CAP TOL` (THE K-MEANS LOOP on a row range of the resident sample, [`kmeans_loop`]);
//!   entries are named, not numbered;
//! * records `TAG INDEX STATUS NANOSECONDS PIPELINE_NANOSECONDS BYTES\n` then the payload; TAG is `start`, `ctx`,
//!   `u` (the resident sample's upload, once a session: its nanoseconds, 0 when it was resident already), `q`, `t`,
//!   `p`, `o`, `k` or `end`.
//!
//! A command that fails writes its failure record and the context is made again for the next command, as the
//! driver's process stops and a new one runs the commands after it. A wait past the bound, or a lost device, also
//! drops the device. A panic in a generated module is caught and is that command's failure.
//!
//! fuller reaches this through the C entry points below only (`futhark_rust_rt_*`), as it reaches the shim: fuller
//! never links this crate.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::values::{ElemType, HostValue, Scalar};
use crate::{EntryDesc, Error, FutharkCtx, ProgramDesc, Value};

/// What a generation's dylib exports, by the Rust ABI (`futhark_rust_module`): its entries and its program.
#[derive(Clone, Copy)]
pub struct ModuleExport {
    pub entries: &'static [EntryDesc],
    pub program: &'static ProgramDesc,
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}
const RTLD_NOW: c_int = 0x2;
const RTLD_LOCAL: c_int = 0x4;

fn dl_error() -> String {
    // SAFETY: dlerror returns a thread-local C string or null.
    let e = unsafe { dlerror() };
    if e.is_null() {
        "(no message)".to_string()
    } else {
        // SAFETY: a non-null dlerror result is a NUL-terminated string.
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

/// Load a generation's dylib (once per path: a loaded module is never unloaded, its code may be in a state).
fn load(path: &Path) -> Result<ModuleExport, Error> {
    let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| Error::new("a module path with a NUL byte"))?;
    // SAFETY: dlopen of a path; the library's initialisers are rustc's.
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    if h.is_null() {
        return Err(Error::new(format!("dlopen {}: {}", path.display(), dl_error())));
    }
    let sym = CString::new("futhark_rust_module").map_err(|_| Error::new("a symbol name with a NUL byte"))?;
    // SAFETY: dlsym on a handle dlopen returned.
    let f = unsafe { dlsym(h, sym.as_ptr()) };
    if f.is_null() {
        return Err(Error::new(format!("{} has no futhark_rust_module: {}", path.display(), dl_error())));
    }
    // SAFETY: the generation crate defines `futhark_rust_module` as `fn() -> ModuleExport`, compiled by the same
    // rustc against this same runtime dylib, so the Rust ABI and the type agree.
    let f: fn() -> ModuleExport = unsafe { std::mem::transmute::<*mut c_void, fn() -> ModuleExport>(f) };
    Ok(f())
}

/// A device for the host: the default adapter, Futhark's requirements (its limits as the adapter allows, `SHADER_F16`)
/// and `TIMESTAMP_QUERY` for the GPU timing, as the shim's device for Futhark's runtime.
fn new_device() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| Error::new(format!("Could not get WebGPU adapter: {e}")))?;
    let mut features = wgpu::Features::SHADER_F16;
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let desc = wgpu::DeviceDescriptor { label: Some("futhark-rust-rt host"), required_features: features, required_limits: adapter.limits(), ..Default::default() };
    pollster::block_on(adapter.request_device(&desc)).map_err(|e| Error::new(format!("Could not get WebGPU device: {e}")))
}

/// THE RESIDENT SAMPLE (Andrew, 2026-11-03): the k-means sample uploaded ONCE for the run and kept on the device
/// across every generation's run and module; a `k` command reads it as the entry's first argument. It is a buffer
/// the session owns (`DevMem::from_buffer`: never in a context's free list), dropped with the device, and replaced
/// only when a run brings another sample (its fingerprint differs).
struct Resident {
    buffer: wgpu::Buffer,
    elem: ElemType,
    shape: Vec<i64>,
    fingerprint: u64,
}

impl Resident {
    /// The value a context reads the resident buffer through.
    fn value(&self) -> Result<Value, Error> {
        Ok(Value::Array(crate::values::AnyArray { elem: self.elem, shape: self.shape.clone(), mem: crate::DevMem::from_buffer(self.buffer.clone())? }))
    }
}

/// FNV-1a of a sample's bytes: whether the resident sample is this one.
fn fingerprint(h: &HostValue) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for b in h.bytes.iter().chain(h.elem.binary_name().as_bytes()).chain(h.shape.iter().flat_map(|d| d.to_le_bytes().to_vec()).collect::<Vec<u8>>().iter()) {
        x ^= u64::from(*b);
        x = x.wrapping_mul(0x0000_0100_0000_01b3);
    }
    x
}

/// THE SAMPLE'S ROWS ON THE HOST (the advisor's round-3 review, N3): the resident sample's `f32` values, parsed from
/// its bytes ONCE a session and kept beside the device's copy, so a `k` command slices its case's rows from them.
/// Before 2026-11-03 every `k` command converted the whole sample (5,000 x 1,536 floats, 30 MB) to slice its case:
/// 2 pairs x 2 sides x 6 cases x 24 programs = 576 conversions a generation. Replaced only by another sample (its
/// fingerprint differs); it outlives the device, which a lost context makes again without it.
struct SampleRows {
    fingerprint: u64,
    floats: std::sync::Arc<Vec<f32>>,
}

/// A session: a device, kept while it works, the modules loaded, the resident sample and its rows on the host.
pub struct Session {
    device: Option<(wgpu::Device, wgpu::Queue)>,
    modules: HashMap<PathBuf, ModuleExport>,
    wait: Duration,
    devices_made: u64,
    resident: Option<Resident>,
    sample_rows: Option<SampleRows>,
    /// How many times a sample's rows were parsed (a measurement aid: once a session for one sample).
    sample_parses: u64,
}

impl Session {
    pub fn new(wait: Duration) -> Session {
        Session { device: None, modules: HashMap::new(), wait, devices_made: 0, resident: None, sample_rows: None, sample_parses: 0 }
    }

    /// The sample's rows as `f32` ([`SampleRows`]): parsed now when no sample's rows are kept, or another sample's
    /// (`fp` its fingerprint), else the kept ones, shared.
    fn sample_rows(&mut self, sample: &HostValue, fp: u64) -> Result<std::sync::Arc<Vec<f32>>, Error> {
        if sample.elem != ElemType::F32 || sample.shape.len() != 2 {
            return Err(Error::new("the sample is not a [m][d]f32"));
        }
        if let Some(r) = self.sample_rows.as_ref().filter(|r| r.fingerprint == fp) {
            return Ok(r.floats.clone());
        }
        let floats: Vec<f32> = sample.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        self.sample_parses += 1;
        let floats = std::sync::Arc::new(floats);
        self.sample_rows = Some(SampleRows { fingerprint: fp, floats: floats.clone() });
        Ok(floats)
    }

    /// How many times a sample's rows were parsed in this session.
    pub fn sample_parses(&self) -> u64 {
        self.sample_parses
    }

    /// The resident sample as `sample` (its fingerprint `fp`; uploaded now when it is not, or is another sample):
    /// the value a context reads it through, and the nanoseconds of the upload (0 when it was resident already).
    fn resident_sample(&mut self, ctx: &FutharkCtx, sample: &HostValue, fp: u64) -> Result<(Value, u128), Error> {
        if self.resident.as_ref().is_none_or(|r| r.fingerprint != fp) {
            let t = Instant::now();
            let size = (sample.bytes.len() as u64).div_ceil(4) * 4;
            let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor { label: Some("futhark resident sample"), size: size.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut bytes = sample.bytes.clone();
            bytes.resize(size as usize, 0);
            ctx.queue().write_buffer(&buffer, 0, &bytes);
            ctx.sync()?;
            self.resident = Some(Resident { buffer, elem: sample.elem, shape: sample.shape.clone(), fingerprint: fp });
            let r = self.resident.as_ref().ok_or_else(|| Error::new("our software has a bug: no resident sample"))?;
            return Ok((r.value()?, t.elapsed().as_nanos()));
        }
        let r = self.resident.as_ref().ok_or_else(|| Error::new("our software has a bug: no resident sample"))?;
        Ok((r.value()?, 0))
    }

    fn ctx(&mut self, timing: bool) -> Result<FutharkCtx, Error> {
        if self.device.is_none() {
            self.device = Some(new_device()?);
            self.devices_made += 1;
        }
        let (d, q) = self.device.as_ref().ok_or_else(|| Error::new("our software has a bug: no device"))?;
        let ctx = FutharkCtx::new(d.clone(), q.clone())?;
        ctx.set_wait_timeout(Some(self.wait));
        ctx.set_timing(timing)?;
        Ok(ctx)
    }

    pub fn devices_made(&self) -> u64 {
        self.devices_made
    }
}

fn unix_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// The records being written.
struct Records {
    out: Vec<u8>,
}

impl Records {
    fn record(&mut self, tag: &str, index: i64, ok: bool, ns: u128, pipeline_ns: u128, payload: &[u8]) {
        self.out.extend_from_slice(format!("{tag} {index} {} {ns} {pipeline_ns} {}\n", if ok { "ok" } else { "err" }, payload.len()).as_bytes());
        self.out.extend_from_slice(payload);
    }
}

/// The inputs of a run: every case's values (case-major, `n_in` values a case) and the timing input's.
pub struct RunInputs<'a> {
    pub cases: &'a [u8],
    pub n_cases: usize,
    pub n_in: usize,
    pub timing: Option<&'a [u8]>,
}

fn find<'m>(m: &'m ModuleExport, name: &str) -> Result<&'m EntryDesc, Error> {
    m.entries.iter().find(|e| e.name == name).ok_or_else(|| Error::new(format!("no such entry point {name}")))
}

fn upload(ctx: &FutharkCtx, vs: &[HostValue]) -> Result<Vec<Value>, Error> {
    vs.iter().map(|h| Value::from_host(ctx, h)).collect()
}

/// THE K-MEANS LOOP ON THE DEVICE-RESIDENT SAMPLE (Andrew, 2026-11-03; fuller `futhark_sig::kmeans_converged`): the
/// sample (`[m][d]f32`, the first value of the run's timing input) is uploaded ONCE for the session and kept
/// ([`Resident`]) across every run and module, and each `k` command runs UltraDim's loop on one case, a row range
/// `start..start+n` of it with the case's starting centres (the LAST input value of `cases[case]`, the k-means
/// signature's `centres`): for up to `cap` iterations the harness assigns each point to its nearest
/// centre ([`crate::kmeans_harness::assign`], on the host, over the sample's rows it holds), the entry is called on
/// (sample, start, n, centres) with the centres ON THE DEVICE (the previous call's output, never uploaded again; the
/// entry slices the range itself, `kmeans_converged::range_entry`), the new centres are read back (k x d f32, 16-300
/// KB) for the harness, and the Euclidean inertia's change under `tol` stops the loop (the cap stops it with
/// `converged` 0). The program's own time is the sum of the GPU-timed passes of its calls. The payload is
/// [`kmeans_payload`]: a line `ITERATIONS CONVERGED FINISHED NANOSECONDS START_INERTIA FINAL_INERTIA`, then, when the
/// run finished (every step a finite `[k][d]f32`), the final centres and THE ASSIGNMENTS THE LOOP STOPPED ON (made
/// against the centres the last step was given, as the stopping rule reads them: fuller's `converge` defines the
/// final clustering so, and its `run_of_k_payload` reads these rather than assigning to the final centres, which is
/// another partition on a run that did not converge), else the failure's text.
fn kmeans_loop(ctx: &FutharkCtx, e: &EntryDesc, case: &KmeansCase) -> Result<Vec<u8>, Error> {
    let KmeansCase { points, d, sample_v, starts_h, start, n, cap, tol } = *case;
    if !ctx.timing() {
        return Err(Error::new("a k-means command in a run with no GPU timing"));
    }
    let f32s = |h: &HostValue| -> Vec<f32> { h.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let k = match (starts_h.elem, starts_h.shape.as_slice()) {
        (ElemType::F32, [k, dd]) if *dd as usize == d => *k as usize,
        _ => return Err(Error::new(format!("the starting centres are not a [k][{d}]f32"))),
    };
    let rows = |v: &[f32]| -> Vec<Vec<f32>> { v.chunks_exact(d).map(<[f32]>::to_vec).collect() };
    let mut centroids = rows(&f32s(starts_h));
    let start_inertia = crate::kmeans_harness::euclidean_inertia(points, &centroids, &crate::kmeans_harness::assign(points, d, &centroids), d);
    let mut centres_v = Value::from_host(ctx, starts_h)?;
    ctx.sync()?;
    let (mut prev, mut current) = (f32::INFINITY, f32::NAN);
    let (mut iterations, mut ns, mut converged) = (0u32, 0f64, false);
    let mut failure = String::new();
    let mut final_h: Option<HostValue> = None;
    let mut assignments: Vec<usize> = Vec::new();
    for it in 0..cap {
        iterations = it + 1;
        // The harness's assignment against the centres the step is given (UltraDim's loop assigns before the update).
        assignments = crate::kmeans_harness::assign(points, d, &centroids);
        let args = [sample_v.clone(), Value::Scalar(Scalar::I64(start as i64)), Value::Scalar(Scalar::I64(n as i64)), centres_v.clone()];
        let m0 = ctx.timed_passes();
        let outs = match (e.call)(ctx, &args).and_then(|o| ctx.sync().map(|()| o)) {
            Ok(o) => o,
            Err(x) => {
                failure = format!("iteration {iterations}: the step failed: {x}");
                break;
            }
        };
        let m1 = ctx.timed_passes();
        ns += (m0..m1).map(|p| ctx.pass_ns(p).unwrap_or(0.0)).sum::<f64>();
        let Some(out) = outs.into_iter().next() else {
            failure = format!("iteration {iterations}: the step returned no value");
            break;
        };
        let h = out.to_host(ctx)?;
        if h.elem != ElemType::F32 || h.shape != [k as i64, d as i64] {
            failure = format!("iteration {iterations}: the centres are not a [{k}][{d}]f32: {:?} {:?}", h.elem, h.shape);
            break;
        }
        let vals = f32s(&h);
        if !vals.iter().all(|x| x.is_finite()) {
            failure = format!("iteration {iterations}: a centroid is infinite or NaN");
            break;
        }
        centroids = rows(&vals);
        final_h = Some(h);
        // The next step takes this output as its centres, on the device: nothing is uploaded again.
        centres_v = out;
        current = crate::kmeans_harness::euclidean_inertia(points, &centroids, &assignments, d);
        if (prev - current).abs() < tol {
            converged = true;
            break;
        }
        prev = current;
    }
    let finished = failure.is_empty();
    let final_inertia = if current.is_finite() { current } else { start_inertia };
    // The cap is at least 1, so a finished run has its last step's centres; the starts stand in only if it has not.
    let centres = final_h.as_ref().unwrap_or(starts_h);
    Ok(kmeans_payload(iterations, converged, ns, start_inertia, final_inertia, if finished { Ok((centres, &assignments)) } else { Err(&failure) }))
}

/// THE PAYLOAD OF A `k` RECORD (fuller's `kmeans_converged::run_of_k_payload` reads it): the line `ITERATIONS
/// CONVERGED FINISHED NANOSECONDS START_INERTIA FINAL_INERTIA`, then for a finished run its final centres
/// (`[k][d]f32`) and the assignments the loop stopped on (`[n]i32`), both in Futhark's binary format, else the
/// failure's text.
fn kmeans_payload(iterations: u32, converged: bool, ns: f64, start_inertia: f32, final_inertia: f32, end: Result<(&HostValue, &[usize]), &str>) -> Vec<u8> {
    let mut payload = format!("{iterations} {} {} {ns:.1} {start_inertia} {final_inertia}\n", u8::from(converged), u8::from(end.is_ok())).into_bytes();
    match end {
        Ok((centres, assignments)) => {
            centres.write_binary(&mut payload);
            let assigned = HostValue { elem: ElemType::I32, shape: vec![assignments.len() as i64], bytes: assignments.iter().flat_map(|a| (*a as i32).to_le_bytes()).collect() };
            assigned.write_binary(&mut payload);
        }
        Err(failure) => payload.extend_from_slice(failure.as_bytes()),
    }
    payload
}

/// One `k` command's case: the case's rows on the host (a slice of the session's [`SampleRows`]) and their width,
/// the sample on the device, the case's starting centres, its row range, the loop's cap and tolerance.
#[derive(Clone, Copy)]
struct KmeansCase<'a> {
    points: &'a [f32],
    d: usize,
    sample_v: &'a Value,
    starts_h: &'a HostValue,
    start: usize,
    n: usize,
    cap: u32,
    tol: f32,
}

/// What every command of a run reads: the cases' values, the timing input's, their counts, the resident sample the
/// `k` commands share (the session's, through this run's context) and its rows on the host.
struct RunState<'a> {
    cases: &'a [HostValue],
    timing: Option<&'a [HostValue]>,
    n_cases: usize,
    n_in: usize,
    resident: Option<&'a Value>,
    sample_rows: Option<&'a [f32]>,
}

/// One command: its payload, or its failure.
fn command(ctx: &FutharkCtx, m: &ModuleExport, line: &str, st: RunState) -> Result<(&'static str, Vec<u8>), (&'static str, Error)> {
    let RunState { cases, timing, n_cases, n_in, resident, sample_rows } = st;
    let w: Vec<&str> = line.split_whitespace().collect();
    match w.as_slice() {
        ["k", entry, case, start, n, cap, tol] => {
            let tag = "k";
            let bad = |what: &str| (tag, Error::new(format!("a k-means command with a bad {what}: {line:?}")));
            let case: usize = case.parse().map_err(|_| bad("case"))?;
            let start: usize = start.parse().map_err(|_| bad("start"))?;
            let n: usize = n.parse().map_err(|_| bad("row count"))?;
            let cap: u32 = cap.parse().map_err(|_| bad("cap"))?;
            let tol: f32 = tol.parse().map_err(|_| bad("tolerance"))?;
            if cap == 0 || tol.is_nan() || tol <= 0.0 {
                return Err(bad("cap or tolerance"));
            }
            let e = find(m, entry).map_err(|x| (tag, x))?;
            let sample_h = timing.and_then(|t| t.first()).ok_or((tag, Error::new("a k-means command with no sample (the timing input)")))?;
            if n_in == 0 || case >= n_cases {
                return Err(bad("case index"));
            }
            let starts_h = cases.get(case * n_in + n_in - 1).ok_or_else(|| bad("case index"))?;
            let sample_v = resident.ok_or((tag, Error::new("our software has a bug: a k-means command with no resident sample")))?;
            let rows = sample_rows.ok_or((tag, Error::new("our software has a bug: a k-means command with no sample rows on the host")))?;
            let (m, d) = match (sample_h.elem, sample_h.shape.as_slice()) {
                (ElemType::F32, [m, d]) => (*m as usize, *d as usize),
                _ => return Err((tag, Error::new("the sample is not a [m][d]f32"))),
            };
            if start + n > m || n == 0 || rows.len() != m * d {
                return Err((tag, Error::new(format!("the range {start}..{} is not within the sample's {m} rows", start + n))));
            }
            // The case's rows: a slice of the session's parsed sample, nothing converted or copied.
            let points = &rows[start * d..(start + n) * d];
            kmeans_loop(ctx, e, &KmeansCase { points, d, sample_v, starts_h, start, n, cap, tol }).map(|p| (tag, p)).map_err(|x| (tag, x))
        }
        ["q", entry] => {
            let tag = "q";
            let e = find(m, entry).map_err(|x| (tag, x))?;
            let args = upload(ctx, cases).map_err(|x| (tag, Error::new(format!("an input did not upload: {x}"))))?;
            ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after the inputs: {x}"))))?;
            let outs = (e.call)(ctx, &args).map_err(|x| (tag, Error::new(format!("the quality entry failed: {x}"))))?;
            ctx.sync().map_err(|x| (tag, Error::new(format!("the quality entry failed: {x}"))))?;
            if outs.len() != n_cases {
                return Err((tag, Error::new(format!("the quality entry returned {} values for {n_cases} cases", outs.len()))));
            }
            let mut payload = Vec::new();
            for o in &outs {
                o.to_host(ctx).map_err(|x| (tag, Error::new(format!("an output did not read back: {x}"))))?.write_binary(&mut payload);
            }
            Ok((tag, payload))
        }
        ["t", entry, runs] => {
            let tag = "t";
            let runs: usize = runs.parse().map_err(|_| (tag, Error::new("a time command with no runs")))?;
            let timing = timing.ok_or((tag, Error::new("a time command with no timing input")))?;
            if !ctx.timing() {
                return Err((tag, Error::new("a time command in a run with no GPU timing")));
            }
            let e = find(m, entry).map_err(|x| (tag, x))?;
            let mut words = Vec::new();
            let args = upload(ctx, timing).map_err(|x| (tag, Error::new(format!("an input did not upload: {x}"))))?;
            ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after the inputs: {x}"))))?;
            for _ in 0..=runs {
                let m0 = ctx.timed_passes();
                let outs = (e.call)(ctx, &args).map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                ctx.sync().map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                drop(outs);
                ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after a run: {x}"))))?;
                let m1 = ctx.timed_passes();
                if m1 == m0 {
                    words.push("none".to_string());
                } else {
                    let ns: f64 = (m0..m1).map(|p| ctx.pass_ns(p).unwrap_or(0.0)).sum();
                    words.push(format!("{ns:.1}"));
                }
            }
            drop(args);
            Ok((tag, words.join(" ").into_bytes()))
        }
        // `p E Y SIDE_NS MIN FIRST`: an INTERLEAVED PAIR (fuller::runner::interleaved_runs). Entries E and Y alternate
        // run by run on one upload of the timing input, Y first when FIRST is 1: a warm-up each, then
        // ceil(SIDE_NS / the slower side's first timed run's wall time) runs each, at least MIN
        // (fuller::futhark_batch::pair_runs). The payload is E's run times, `|`, Y's run times, each as `t`'s words.
        ["p", entry, yard, side_ns, min_runs, first] => {
            let tag = "p";
            let side_ns: f64 = side_ns.parse().map_err(|_| (tag, Error::new("a pair command with no side time")))?;
            let min_runs: usize = min_runs.parse().map_err(|_| (tag, Error::new("a pair command with no least runs")))?;
            let yard_first = match *first {
                "1" => true,
                "0" => false,
                _ => return Err((tag, Error::new("a pair command whose first side is not 0 or 1"))),
            };
            if side_ns.is_nan() || side_ns <= 0.0 || min_runs < 1 {
                return Err((tag, Error::new("a pair command with no side time or no runs")));
            }
            let timing = timing.ok_or((tag, Error::new("a pair command with no timing input")))?;
            if !ctx.timing() {
                return Err((tag, Error::new("a pair command in a run with no GPU timing")));
            }
            let sides = [find(m, entry).map_err(|x| (tag, x))?, find(m, yard).map_err(|x| (tag, x))?];
            let args = upload(ctx, timing).map_err(|x| (tag, Error::new(format!("an input did not upload: {x}"))))?;
            ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after the inputs: {x}"))))?;
            let mut words: [Vec<String>; 2] = [Vec::new(), Vec::new()];
            let mut first_wall = [0f64; 2];
            let mut need: Option<usize> = None;
            let mut k = 0usize;
            while need.is_none_or(|n| words[0].len() < n + 1 || words[1].len() < n + 1) {
                // Side 1 (the yardstick) on the even calls when it goes first, else on the odd ones.
                let side = usize::from(k.is_multiple_of(2) == yard_first);
                k += 1;
                let w0 = Instant::now();
                let m0 = ctx.timed_passes();
                let outs = (sides[side].call)(ctx, &args).map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                ctx.sync().map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                drop(outs);
                ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after a run: {x}"))))?;
                let m1 = ctx.timed_passes();
                let wall = w0.elapsed().as_nanos() as f64;
                words[side].push(if m1 == m0 { "none".to_string() } else { format!("{:.1}", (m0..m1).map(|p| ctx.pass_ns(p).unwrap_or(0.0)).sum::<f64>()) });
                if words[side].len() == 2 {
                    first_wall[side] = wall;
                }
                if need.is_none() && words[0].len() >= 2 && words[1].len() >= 2 {
                    let slower = first_wall[0].max(first_wall[1]);
                    let n = if slower > 0.0 { (side_ns / slower).ceil() } else { 0.0 };
                    need = Some(if n > min_runs as f64 { n as usize } else { min_runs });
                }
            }
            drop(args);
            Ok((tag, format!("{}|{}", words[0].join(" "), words[1].join(" ")).into_bytes()))
        }
        ["o", entry] => {
            let tag = "o";
            let timing = timing.ok_or((tag, Error::new("an output command with no timing input")))?;
            let e = find(m, entry).map_err(|x| (tag, x))?;
            let args = upload(ctx, timing).map_err(|x| (tag, Error::new(format!("an input did not upload: {x}"))))?;
            let outs = (e.call)(ctx, &args).map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
            ctx.sync().map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
            let mut payload = Vec::new();
            outs.first()
                .ok_or((tag, Error::new("the timed entry returned no value")))?
                .to_host(ctx)
                .map_err(|x| (tag, Error::new(format!("the output did not read back: {x}"))))?
                .write_binary(&mut payload);
            Ok((tag, payload))
        }
        // `s`: the context's counts so far (a measurement aid: launches, launches skipped as empty, timed passes,
        // pipelines made and reused, submissions).
        ["s"] => {
            let st = ctx.stats();
            Ok((
                "s",
                format!("launches {} skipped {} passes {} pipelines {} cached {} submits {}", st.launches, st.skipped_launches, ctx.timed_passes(), st.pipelines_created, st.pipelines_cached, st.submits).into_bytes(),
            ))
        }
        _ => Err(("q", Error::new(format!("an unknown command {line:?}")))),
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "(no message)".into())
}

/// Run `commands` on the module at `path`: the batch driver's results records.
pub fn run(s: &mut Session, path: &Path, commands: &str, inputs: &RunInputs) -> Vec<u8> {
    let mut r = Records { out: Vec::new() };
    r.record("start", -1, true, unix_ns(), 0, &[]);
    let cases = match HostValue::read_all(inputs.cases) {
        Ok(v) if v.len() == inputs.n_cases * inputs.n_in => v,
        Ok(v) => {
            r.record("start", -1, false, 0, 0, format!("{} case values, wanted {}", v.len(), inputs.n_cases * inputs.n_in).as_bytes());
            return r.out;
        }
        Err(e) => {
            r.record("start", -1, false, 0, 0, e.message().as_bytes());
            return r.out;
        }
    };
    let timing = match inputs.timing.map(HostValue::read_all) {
        Some(Ok(v)) => Some(v),
        Some(Err(e)) => {
            r.record("start", -1, false, 0, 0, e.message().as_bytes());
            return r.out;
        }
        None => None,
    };
    let timed = commands.lines().any(|l| l.starts_with("t ") || l.starts_with("p ") || l.starts_with("k "));
    if !s.modules.contains_key(path) {
        match load(path) {
            Ok(m) => {
                s.modules.insert(path.to_path_buf(), m);
            }
            Err(e) => {
                r.record("ctx", -1, false, 0, 0, e.message().as_bytes());
                return r.out;
            }
        }
    }
    let m = s.modules[path];
    let program = m.program;
    // The device's set-up, and the program's own WGSL module: one shared module for every member, so a module that
    // does not validate fails them all (the driver's set-up fails, and the batch is halved).
    // It is checked as it will be made: specialised by the program's own macro values in this context.
    let setup = s.ctx(timed).and_then(|ctx| {
        let p = crate::Program::new(&ctx, program)?;
        ctx.module_check(p.wgsl()).map(|()| ctx)
    });
    let mut ctx = match setup {
        Ok(c) => Some(c),
        Err(e) => {
            r.record("ctx", -1, false, 0, 0, e.message().as_bytes());
            return r.out;
        }
    };
    r.record("ctx", -1, true, unix_ns(), 0, &[]);
    // THE RESIDENT SAMPLE of the `k` commands: the session's, uploaded now only when it is not there yet (the first
    // run of the session, or a device made again), as the `u` record says (its nanoseconds; 0 when resident).
    let mut resident: Option<Value> = None;
    // The sample's fingerprint, once a run (30 MB hashed), for the device's copy and the host's rows alike; the rows
    // parsed once a session ([`SampleRows`]). The `u` record's payload: the sample's bytes, then whether this run
    // parsed the rows (`rows_parsed 1`) or found them kept (`rows_parsed 0`).
    let mut sample_fp = 0u64;
    let mut sample_rows: Option<std::sync::Arc<Vec<f32>>> = None;
    if commands.lines().any(|l| l.starts_with("k ")) {
        let Some(sample) = timing.as_ref().and_then(|t| t.first()) else {
            r.record("u", -1, false, 0, 0, b"a k-means run with no sample (the timing input)");
            return r.out;
        };
        sample_fp = fingerprint(sample);
        let parses_before = s.sample_parses();
        let rows = match s.sample_rows(sample, sample_fp) {
            Ok(rows) => rows,
            Err(e) => {
                r.record("u", -1, false, 0, 0, e.message().as_bytes());
                return r.out;
            }
        };
        let parsed = s.sample_parses() - parses_before;
        sample_rows = Some(rows);
        let got = match ctx.as_ref() {
            Some(c) => s.resident_sample(c, sample, sample_fp),
            None => Err(Error::new("our software has a bug: no context for the upload")),
        };
        match got {
            Ok((v, ns)) => {
                r.record("u", -1, true, ns, 0, format!("{} rows_parsed {parsed}", sample.bytes.len()).as_bytes());
                resident = Some(v);
            }
            Err(e) => {
                r.record("u", -1, false, 0, 0, e.message().as_bytes());
                return r.out;
            }
        }
    }
    let mut index = 0i64;
    for line in commands.lines().filter(|l| !l.trim().is_empty()) {
        let c = match ctx.take() {
            Some(c) => c,
            None => match s.ctx(timed) {
                Ok(c) => {
                    // A context made again after a device was: the resident sample is uploaded again for it.
                    if resident.is_none() && commands.lines().any(|l| l.starts_with("k ")) {
                        if let Some(sample) = timing.as_ref().and_then(|t| t.first()) {
                            match s.resident_sample(&c, sample, sample_fp) {
                                Ok((v, ns)) => {
                                    r.record("u", -1, true, ns, 0, format!("{} rows_parsed 0", sample.bytes.len()).as_bytes());
                                    resident = Some(v);
                                }
                                Err(e) => {
                                    r.record("u", index, false, 0, 0, e.message().as_bytes());
                                    index += 1;
                                    continue;
                                }
                            }
                        }
                    }
                    c
                }
                Err(e) => {
                    r.record("q", index, false, 0, 0, format!("the context could not be made again: {e}").as_bytes());
                    index += 1;
                    continue;
                }
            },
        };
        let p0 = c.stats().pipeline_ns;
        let t0 = Instant::now();
        // One bound on the whole command, as the driver's `alarm(bound)`: every wait respects the deadline, and a
        // command that ends past it is a failure too.
        c.set_deadline(Some(t0 + s.wait));
        let got = catch_unwind(AssertUnwindSafe(|| command(&c, &m, line, RunState { cases: &cases, timing: timing.as_deref(), n_cases: inputs.n_cases, n_in: inputs.n_in, resident: resident.as_ref(), sample_rows: sample_rows.as_deref().map(Vec::as_slice) })));
        c.set_deadline(None);
        let ns = t0.elapsed().as_nanos();
        let over = t0.elapsed() >= s.wait;
        let got = match got {
            Ok(Ok((tag, _))) if over => Ok(Err((tag, Error::device(format!("the command ran past its bound of {} s", s.wait.as_secs()))))),
            Ok(Err((tag, e))) if e.is_device_failure() && over => Ok(Err((tag, Error::device(format!("the command ran past its bound of {} s: {}", s.wait.as_secs(), e.message()))))),
            g => g,
        };
        let pns = c.stats().pipeline_ns - p0;
        match got {
            Ok(Ok((tag, payload))) => {
                r.record(tag, index, true, ns, pns, &payload);
                ctx = Some(c);
            }
            Ok(Err((tag, e))) => {
                r.record(tag, index, false, 0, pns, e.message().as_bytes());
                // A new context for the next command, as a new driver process; a device that failed is made again too,
                // and the resident sample with it.
                drop(c);
                if e.is_device_failure() {
                    resident = None;
                    s.device = None;
                    s.resident = None;
                }
            }
            Err(p) => {
                let tag = match line.split_whitespace().next() {
                    Some("t") => "t",
                    Some("p") => "p",
                    Some("o") => "o",
                    Some("k") => "k",
                    _ => "q",
                };
                r.record(tag, index, false, 0, pns, format!("the generated module panicked: {}", panic_text(&*p)).as_bytes());
                resident = None;
                drop(c);
                s.device = None;
                s.resident = None;
            }
        }
        index += 1;
    }
    r.record("end", index, true, unix_ns(), 0, &[]);
    r.out
}

// ---- the C entry points fuller calls ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::default());
}

/// A new session; `wait_seconds` bounds every wait for the device.
#[no_mangle]
pub extern "C" fn futhark_rust_rt_session_new(wait_seconds: u32) -> *mut Session {
    Box::into_raw(Box::new(Session::new(Duration::from_secs(u64::from(wait_seconds.max(1))))))
}

/// # Safety
/// `s` came from [`futhark_rust_rt_session_new`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn futhark_rust_rt_session_free(s: *mut Session) {
    if !s.is_null() {
        // SAFETY: the caller's contract.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// How many devices the session has made (a device is made again after a lost one or a wait past the bound).
///
/// # Safety
/// `s` came from [`futhark_rust_rt_session_new`].
#[no_mangle]
pub unsafe extern "C" fn futhark_rust_rt_devices_made(s: *const Session) -> u64 {
    // SAFETY: the caller's contract.
    unsafe { s.as_ref() }.map(Session::devices_made).unwrap_or(0)
}

/// The arguments of [`futhark_rust_rt_run`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RunArgs {
    /// The generation's dylib, NUL-terminated.
    pub module_path: *const c_char,
    /// The commands, one a line, NUL-terminated.
    pub commands: *const c_char,
    /// Every case's values, Futhark's binary format.
    pub cases: *const u8,
    pub cases_len: usize,
    pub n_cases: u32,
    pub n_in: u32,
    /// The timing input's values, or null.
    pub timing: *const u8,
    pub timing_len: usize,
}

/// Run the commands on the module: the records are returned in a buffer of `*out_len` bytes, freed with
/// [`futhark_rust_rt_free_bytes`]. Null on a misuse of the API, with [`futhark_rust_rt_last_error`].
///
/// # Safety
/// Every pointer is valid for its length; the strings are NUL-terminated; `s` came from
/// [`futhark_rust_rt_session_new`]; `timing` may be null (no timing input).
#[no_mangle]
pub unsafe extern "C" fn futhark_rust_rt_run(s: *mut Session, args: *const RunArgs, out_len: *mut usize) -> *mut u8 {
    let fail = |msg: &str| {
        LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg.replace('\0', " ")).unwrap_or_default());
        std::ptr::null_mut()
    };
    // SAFETY: the caller's contract, for every dereference below.
    let Some(s) = (unsafe { s.as_mut() }) else { return fail("a null session") };
    let Some(a) = (unsafe { args.as_ref() }) else { return fail("null arguments") };
    let RunArgs { module_path, commands, cases, cases_len, n_cases, n_in, timing, timing_len } = *a;
    if module_path.is_null() || commands.is_null() || cases.is_null() || out_len.is_null() {
        return fail("a null argument");
    }
    let path = PathBuf::from(unsafe { CStr::from_ptr(module_path) }.to_string_lossy().into_owned());
    let commands = unsafe { CStr::from_ptr(commands) }.to_string_lossy().into_owned();
    let cases = unsafe { std::slice::from_raw_parts(cases, cases_len) };
    let timing = if timing.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(timing, timing_len) }) };
    let inputs = RunInputs { cases, n_cases: n_cases as usize, n_in: n_in as usize, timing };
    let out = run(s, &path, &commands, &inputs).into_boxed_slice();
    unsafe { *out_len = out.len() };
    Box::into_raw(out).cast::<u8>()
}

/// # Safety
/// `p` and `len` are a buffer [`futhark_rust_rt_run`] returned, not freed before.
#[no_mangle]
pub unsafe extern "C" fn futhark_rust_rt_free_bytes(p: *mut u8, len: usize) {
    if !p.is_null() {
        // SAFETY: the caller's contract: this is the boxed slice `run` leaked.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, len)) });
    }
}

/// The last misuse's message, valid until the next call on this thread.
#[no_mangle]
pub extern "C" fn futhark_rust_rt_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finished `k` record's payload carries its header line, then the final centres and the assignments the loop
    /// stopped on as a `[n]i32`, in Futhark's binary format (fuller's `run_of_k_payload` wants exactly those two
    /// values); an unfinished one carries its header and the failure's text. No device.
    #[test]
    fn a_k_records_payload_carries_the_final_centres_and_the_loops_assignments() {
        let centres = HostValue { elem: ElemType::F32, shape: vec![2, 3], bytes: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|x| x.to_le_bytes()).collect() };
        let p = kmeans_payload(7, false, 1234.5, 10.0, 2.5, Ok((&centres, &[1usize, 0, 1, 1])));
        let nl = p.iter().position(|b| *b == b'\n').expect("a header line");
        assert_eq!(std::str::from_utf8(&p[..nl]).expect("text"), "7 0 1 1234.5 10 2.5");
        let values = HostValue::read_all(&p[nl + 1..]).expect("two values");
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], centres);
        assert_eq!(values[1], HostValue { elem: ElemType::I32, shape: vec![4], bytes: [1i32, 0, 1, 1].iter().flat_map(|x| x.to_le_bytes()).collect() });
        let f = kmeans_payload(3, false, 0.0, 10.0, 10.0, Err("iteration 3: a centroid is infinite or NaN"));
        let nl = f.iter().position(|b| *b == b'\n').expect("a header line");
        assert_eq!(std::str::from_utf8(&f[..nl]).expect("text"), "3 0 0 0.0 10 10");
        assert_eq!(&f[nl + 1..], b"iteration 3: a centroid is infinite or NaN");
    }

    /// THE SAMPLE'S ROWS ARE PARSED ONCE A SESSION (the advisor's round-3 review, N3): the dbpedia sample's size
    /// (5,000 x 1,536 f32, 30 MB) asked for twice gives the one parsed vector (the same allocation, one parse); another
    /// sample is parsed again; a sample that is not a `[m][d]f32` is refused. The parse is timed and the saving
    /// stated at the first run's count: 576 conversions a generation (2 pairs x 2 sides x 6 cases x 24 programs).
    /// No device.
    #[test]
    fn the_samples_rows_are_parsed_once_a_session_and_shared_by_every_k_command() {
        let (m, d) = (5000usize, 1536usize);
        let bytes: Vec<u8> = (0..m * d).flat_map(|i| ((i % 977) as f32 / 977.0).to_le_bytes()).collect();
        let sample = HostValue { elem: ElemType::F32, shape: vec![m as i64, d as i64], bytes };
        let fp = fingerprint(&sample);
        let mut s = Session::new(Duration::from_secs(1));
        let t = Instant::now();
        let a = s.sample_rows(&sample, fp).expect("the rows");
        let parse = t.elapsed();
        let t = Instant::now();
        let b = s.sample_rows(&sample, fp).expect("the rows again");
        let reuse = t.elapsed();
        assert!(std::sync::Arc::ptr_eq(&a, &b), "the second request must share the first parse");
        assert_eq!(s.sample_parses(), 1);
        assert_eq!(a.len(), m * d);
        assert_eq!(a[d + 1], (1537 % 977) as f32 / 977.0);
        assert_eq!(&a[d..2 * d], &sample.bytes[4 * d..8 * d].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>()[..]);
        let other = HostValue { elem: ElemType::F32, shape: vec![2, 2], bytes: vec![0; 16] };
        let c = s.sample_rows(&other, fingerprint(&other)).expect("another sample's rows");
        assert_eq!((c.len(), s.sample_parses()), (4, 2), "another sample is parsed again");
        assert!(s.sample_rows(&sample, fp).is_ok() && s.sample_parses() == 3, "the first sample again, after another: parsed again (one sample is kept)");
        let wrong = HostValue { elem: ElemType::I32, shape: vec![4], bytes: vec![0; 16] };
        assert!(s.sample_rows(&wrong, fingerprint(&wrong)).unwrap_err().message().contains("[m][d]f32"));
        let per_generation = 2 * 2 * 6 * 24;
        eprintln!("the sample's rows ({} MB): parsed once in {:.1} ms, found kept in {:.3} ms; before, {per_generation} parses a generation = {:.2} s of host time a generation", sample.bytes.len() / 1_000_000, parse.as_secs_f64() * 1e3, reuse.as_secs_f64() * 1e3, parse.as_secs_f64() * per_generation as f64);
    }
}
