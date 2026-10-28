//! THE IN-PROCESS EVALUATION HOST: the batch driver of `fuller::futhark_batch`, in this process, on one device.
//!
//! A generation's programs are one generated module, compiled by `rustc` into a dylib that links THIS runtime as a
//! dylib (so the module, the host and wgpu are one copy in the process), and loaded with `dlopen`. [`run`] executes
//! the batch driver's commands on it and writes the batch driver's results records, byte for byte the format
//! `futhark_batch` parses:
//!
//! * commands, one a line: `q ENTRY` (the quality entry on every stored case), `t ENTRY RUNS` (`RUNS + 1` runs on the
//!   timing input, the first a warm-up, each run's GPU nanoseconds), `o ENTRY` (once on the timing input, the
//!   output); entries are named, not numbered;
//! * records `TAG INDEX STATUS NANOSECONDS PIPELINE_NANOSECONDS BYTES\n` then the payload; TAG is `start`, `ctx`,
//!   `q`, `t`, `o` or `end`.
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

use crate::values::HostValue;
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

/// A session: a device, kept while it works, and the modules loaded.
pub struct Session {
    device: Option<(wgpu::Device, wgpu::Queue)>,
    modules: HashMap<PathBuf, ModuleExport>,
    wait: Duration,
    devices_made: u64,
}

impl Session {
    pub fn new(wait: Duration) -> Session {
        Session { device: None, modules: HashMap::new(), wait, devices_made: 0 }
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

/// One command: its payload, or its failure.
fn command(ctx: &FutharkCtx, m: &ModuleExport, line: &str, cases: &[HostValue], timing: Option<&[HostValue]>, n_cases: usize) -> Result<(&'static str, Vec<u8>), (&'static str, Error)> {
    let w: Vec<&str> = line.split_whitespace().collect();
    match w.as_slice() {
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
            for _ in 0..=runs {
                let m0 = ctx.timed_passes();
                let args = upload(ctx, timing).map_err(|x| (tag, Error::new(format!("an input did not upload: {x}"))))?;
                ctx.sync().map_err(|x| (tag, Error::new(format!("the sync after the inputs: {x}"))))?;
                let outs = (e.call)(ctx, &args).map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                ctx.sync().map_err(|x| (tag, Error::new(format!("the timed entry failed: {x}"))))?;
                drop(args);
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
            Ok((tag, words.join(" ").into_bytes()))
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
    let timed = commands.lines().any(|l| l.starts_with("t "));
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
    let mut index = 0i64;
    for line in commands.lines().filter(|l| !l.trim().is_empty()) {
        let c = match ctx.take() {
            Some(c) => c,
            None => match s.ctx(timed) {
                Ok(c) => c,
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
        let got = catch_unwind(AssertUnwindSafe(|| command(&c, &m, line, &cases, timing.as_deref(), inputs.n_cases)));
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
                // A new context for the next command, as a new driver process; a device that failed is made again too.
                drop(c);
                if e.is_device_failure() {
                    s.device = None;
                }
            }
            Err(p) => {
                let tag = match line.split_whitespace().next() {
                    Some("t") => "t",
                    Some("o") => "o",
                    _ => "q",
                };
                r.record(tag, index, false, 0, pns, format!("the generated module panicked: {}", panic_text(&*p)).as_bytes());
                drop(c);
                s.device = None;
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
