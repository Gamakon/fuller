//! THE RUST-WGPU PATH (`docs/PLAN_futhark_rust_wgpu.md`): Futhark programs compiled to a Rust module (WGSL kernels and
//! a host over wgpu-rs, by the `rust-wgpu` backend of our Futhark fork) and run IN THIS PROCESS, on one device, by the
//! runtime crate `futhark-rust-rt`. No C, no shim, no process a candidate: what evolution scores is what ships.
//!
//! * [`prebuild_rt`] builds the runtime once, as a dylib, into a run's own directory (`futhark-rust-rt/prebuild_rt.sh`);
//! * [`compile_module`] compiles one batch's Futhark source: the fork compiler writes `prog.rs`, `rustc` builds it into
//!   a dylib that links that runtime (one copy of the runtime, wgpu and libstd in the process);
//! * [`Session`] opens the runtime with `dlopen` and runs the batch driver's commands on a module, in this process,
//!   returning the batch driver's records (`crate::futhark_batch` parses them as it parses the driver's).
//!
//! fuller reaches the runtime through its C entry points only, as it reaches the shim: fuller never links it, so
//! fuller's own wgpu and the runtime's never meet.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::futhark_exec::output_with_timeout;
use crate::futhark_wgpu_host::CompileStages;

/// The fork's compiler: `$FUTHARK_RUST_WGPU`, or [`default_compiler`].
pub fn compiler() -> PathBuf {
    std::env::var_os("FUTHARK_RUST_WGPU").map(PathBuf::from).unwrap_or_else(default_compiler)
}

/// Where `futhark-rust-backend/build.sh` installs the fork, built from the committed patch series
/// (`futhark-rust-backend/patches`): `$HOME/.local/share/fuller/futhark-rust/bin/futhark`, a path that survives a
/// reboot. Until 2026-10-31 the default was a build under `/private/tmp` with the first 6 of the 9 patches, so a run
/// without `FUTHARK_RUST_WGPU`/`FUTHARK_BRANCH` compiled with a compiler that fails 373 corpus programs.
/// [`installed_series`] names the series the build holds.
pub fn default_compiler() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share/fuller/futhark-rust/bin/futhark")
}

/// The patch series a compiler was built from: `build.sh` copies it to `bin/patches/` beside the compiler. Each
/// patch's file name and bytes, sorted by name.
pub fn installed_series(compiler: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let dir = compiler.parent().ok_or_else(|| format!("{} has no directory", compiler.display()))?.join("patches");
    series_in(&dir)
}

/// The committed patch series (`futhark-rust-backend/patches`), as [`installed_series`] reads one.
pub fn committed_series() -> Result<Vec<(String, Vec<u8>)>, String> {
    series_in(&Path::new(env!("CARGO_MANIFEST_DIR")).join("futhark-rust-backend/patches"))
}

fn series_in(dir: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let p = e.map_err(|e| format!("read {}: {e}", dir.display()))?.path();
        if p.extension().is_some_and(|x| x == "patch") {
            let name = p.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("{} is not a UTF-8 name", p.display()))?.to_string();
            out.push((name, std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?));
        }
    }
    out.sort();
    Ok(out)
}

/// The compile-failure corpus (`tests/corpus/futhark_compile_failures`): every program `index.tsv` lists (checked to
/// be exactly the corpus's `.fut` files), then the minimal reproducers (one a kind).
#[cfg(test)]
pub(crate) fn compile_failure_corpus() -> Vec<PathBuf> {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_compile_failures");
    let index = std::fs::read_to_string(corpus.join("index.tsv")).expect("the corpus index");
    let mut files: Vec<PathBuf> = index.lines().skip(1).map(|l| corpus.join(l.split('\t').next().expect("a file column"))).collect();
    let mut on_disk: Vec<PathBuf> = std::fs::read_dir(&corpus).expect("the corpus").map(|e| e.expect("an entry").path()).filter(|p| p.extension().is_some_and(|x| x == "fut")).collect();
    files.sort();
    on_disk.sort();
    assert_eq!(files, on_disk, "index.tsv must list exactly the corpus's programs");
    assert_eq!(files.len(), 373, "the corpus holds 373 programs");
    let mut minimal: Vec<PathBuf> = std::fs::read_dir(corpus.join("minimal")).expect("the minimal reproducers").map(|e| e.expect("an entry").path()).collect();
    minimal.sort();
    assert_eq!(minimal.len(), 3, "one minimal reproducer a kind");
    files.extend(minimal);
    files
}

/// The runtime crate.
pub fn rt_crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("futhark-rust-rt")
}

const PREBUILD_TIMEOUT: Duration = Duration::from_secs(1800);

/// Build the runtime once into `dir` (the run's own copy). Returns `dir`.
pub fn prebuild_rt(dir: &Path) -> Result<PathBuf, String> {
    // The lock is waited for outside the build's bound: a wait is another holder's build, not a hang of ours.
    let lock = crate::futhark_wgpu_host::CargoLock::take()?;
    let out = output_with_timeout(lock.mark(Command::new(rt_crate_dir().join("prebuild_rt.sh")).arg(dir).env_remove("CARGO_TARGET_DIR")), PREBUILD_TIMEOUT, None)?;
    drop(lock);
    if !out.status.success() {
        return Err(format!("prebuilding the runtime failed:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(dir.to_path_buf())
}

/// The largest binding index of a generated module's kernels (`scalars_binding: N`, `binding_indices: &[..]`): Futhark
/// numbers bindings across the whole program, and wgpu's limit is 1000 (`crate::futhark_batch`).
pub fn max_binding_index(rust: &str) -> u64 {
    let mut top = 0;
    for l in rust.lines().map(str::trim) {
        if let Some(n) = l.strip_prefix("scalars_binding: ") {
            top = top.max(n.trim_end_matches(',').parse().unwrap_or(0));
        }
        if let Some(b) = l.strip_prefix("binding_indices: &[") {
            for w in b.trim_end_matches("],").split(',').map(str::trim).filter(|w| !w.is_empty()) {
                top = top.max(w.parse().unwrap_or(0));
            }
        }
    }
    top
}

/// The crate root of a generation: the module, and the one symbol the host looks up.
const LIB_RS: &str = "//! One batch of programs, generated by Futhark's rust-wgpu backend (fuller::futhark_rust).\n\
#[path = \"prog.rs\"]\npub mod prog;\n\n\
/// The module's entry points and program, for the in-process host (`futhark_rust_rt::host`).\n\
#[no_mangle]\npub fn futhark_rust_module() -> futhark_rust_rt::host::ModuleExport {\n    \
futhark_rust_rt::host::ModuleExport { entries: prog::ENTRIES, program: &prog::PROGRAM }\n}\n";

/// Compile `source` (a Futhark program) in `dir` into a generation dylib against the runtime in `rt` (a
/// [`prebuild_rt`] directory). `tag` names the crate (unique in the process: the host keeps every module it loads).
/// The stages: `futhark` is the fork compiler, `cc` is rustc.
pub fn compile_module(source: &str, dir: &Path, rt: &Path, tag: &str, timeout: Duration) -> Result<(PathBuf, CompileStages), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write {}: {e}", fut.display()))?;
    let mut stages = CompileStages::default();
    let t = Instant::now();
    let out = output_with_timeout(Command::new(compiler()).arg("rust-wgpu").arg(&fut).arg("-o").arg(dir.join("prog")), timeout, None)?;
    stages.futhark = t.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(format!("futhark rust-wgpu failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    let lib = dir.join("lib.rs");
    std::fs::write(&lib, LIB_RS).map_err(|e| format!("write {}: {e}", lib.display()))?;
    let flags = std::fs::read_to_string(rt.join("rustc_flags")).map_err(|e| format!("the runtime's rustc flags ({}): {e}", rt.display()))?;
    let crate_name = format!("futhark_gen_{tag}");
    let dylib = dir.join(format!("lib{crate_name}.dylib"));
    let mut cmd = Command::new("rustc");
    cmd.args(["--edition", "2021", "--crate-type", "dylib", "--crate-name", &crate_name, "-C", "opt-level=1", "-D", "warnings"])
        .arg("-C")
        .arg(format!("metadata={crate_name}"));
    for f in flags.lines().filter(|l| !l.trim().is_empty()) {
        let (a, b) = f.split_once(' ').ok_or_else(|| format!("a rustc flag line {f:?}"))?;
        cmd.arg(a).arg(b);
    }
    cmd.arg(&lib).arg("-o").arg(&dylib);
    let t = Instant::now();
    let out = output_with_timeout(&mut cmd, timeout, None)?;
    stages.cc = t.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(format!("rustc of the generated module failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok((dylib, stages))
}

// ---- the runtime's C entry points ------------------------------------------------------------------------------

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}
const RTLD_NOW: c_int = 0x2;
const RTLD_LOCAL: c_int = 0x4;

/// `futhark_rust_rt::host::RunArgs`, the C ABI's argument struct (the two definitions are the contract; a GPU test
/// runs the path end to end).
#[repr(C)]
struct RunArgs {
    module_path: *const c_char,
    commands: *const c_char,
    cases: *const u8,
    cases_len: usize,
    n_cases: u32,
    n_in: u32,
    timing: *const u8,
    timing_len: usize,
}

type SessionNew = extern "C" fn(u32) -> *mut c_void;
type SessionFree = unsafe extern "C" fn(*mut c_void);
type Run = unsafe extern "C" fn(*mut c_void, *const RunArgs, *mut usize) -> *mut u8;
type FreeBytes = unsafe extern "C" fn(*mut u8, usize);
type LastError = extern "C" fn() -> *const c_char;
type DevicesMade = unsafe extern "C" fn(*const c_void) -> u64;

/// The runtime's entry points, from one dylib (opened once per process and path, never closed).
#[derive(Clone, Copy)]
struct Rt {
    session_new: SessionNew,
    session_free: SessionFree,
    run: Run,
    free_bytes: FreeBytes,
    last_error: LastError,
    devices_made: DevicesMade,
}

fn dl_error() -> String {
    // SAFETY: dlerror returns a thread-local C string or null.
    let e = unsafe { dlerror() };
    if e.is_null() {
        "(no message)".into()
    } else {
        // SAFETY: a non-null dlerror result is NUL-terminated.
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

static OPENED: Mutex<Option<HashMap<PathBuf, Rt>>> = Mutex::new(None);

fn open(dir: &Path) -> Result<Rt, String> {
    let mut g = OPENED.lock().unwrap_or_else(|p| p.into_inner());
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(rt) = map.get(dir) {
        return Ok(*rt);
    }
    let path = dir.join("libfuthark_rust_rt.dylib");
    let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| "a runtime path with a NUL byte")?;
    // SAFETY: dlopen of the runtime dylib prebuild_rt wrote.
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    if h.is_null() {
        return Err(format!("dlopen {}: {}", path.display(), dl_error()));
    }
    let sym = |name: &str| -> Result<*mut c_void, String> {
        let n = CString::new(name).map_err(|_| "a symbol name with a NUL byte")?;
        // SAFETY: dlsym on the handle dlopen returned.
        let p = unsafe { dlsym(h, n.as_ptr()) };
        if p.is_null() {
            Err(format!("the runtime has no {name}: {}", dl_error()))
        } else {
            Ok(p)
        }
    };
    // SAFETY: each symbol is the runtime's `#[no_mangle] extern "C"` function of exactly this signature
    // (futhark-rust-rt/src/host.rs).
    let rt = unsafe {
        Rt {
            session_new: std::mem::transmute::<*mut c_void, SessionNew>(sym("futhark_rust_rt_session_new")?),
            session_free: std::mem::transmute::<*mut c_void, SessionFree>(sym("futhark_rust_rt_session_free")?),
            run: std::mem::transmute::<*mut c_void, Run>(sym("futhark_rust_rt_run")?),
            free_bytes: std::mem::transmute::<*mut c_void, FreeBytes>(sym("futhark_rust_rt_free_bytes")?),
            last_error: std::mem::transmute::<*mut c_void, LastError>(sym("futhark_rust_rt_last_error")?),
            devices_made: std::mem::transmute::<*mut c_void, DevicesMade>(sym("futhark_rust_rt_devices_made")?),
        }
    };
    map.insert(dir.to_path_buf(), rt);
    Ok(rt)
}

/// A session of the runtime: one device (made again only after a failure of the device), and the modules loaded.
pub struct Session {
    rt: Rt,
    s: *mut c_void,
}

impl Session {
    /// A session of the runtime in `rt` (a [`prebuild_rt`] directory); `wait` bounds every wait for the device.
    pub fn new(rt: &Path, wait: Duration) -> Result<Session, String> {
        let rt = open(rt)?;
        let s = (rt.session_new)(u32::try_from(wait.as_secs().max(1)).map_err(|_| "a wait bound too large")?);
        if s.is_null() {
            return Err("the runtime made no session".into());
        }
        Ok(Session { rt, s })
    }

    /// Run the batch driver's `commands` on the module `dylib`: the records (`crate::futhark_batch`'s format).
    pub fn run(&self, dylib: &Path, commands: &str, cases: &[u8], n_cases: usize, n_in: usize, timing: Option<&[u8]>) -> Result<Vec<u8>, String> {
        #[cfg(test)]
        let _turn = crate::gpu_test_lock::futhark_gpu_turn("futhark_rust::Session::run");
        let path = CString::new(dylib.as_os_str().as_encoded_bytes()).map_err(|_| "a module path with a NUL byte")?;
        let cmds = CString::new(commands).map_err(|_| "commands with a NUL byte")?;
        let args = RunArgs {
            module_path: path.as_ptr(),
            commands: cmds.as_ptr(),
            cases: cases.as_ptr(),
            cases_len: cases.len(),
            n_cases: u32::try_from(n_cases).map_err(|_| "too many cases")?,
            n_in: u32::try_from(n_in).map_err(|_| "too many inputs")?,
            timing: timing.map_or(std::ptr::null(), <[u8]>::as_ptr),
            timing_len: timing.map_or(0, <[u8]>::len),
        };
        let mut len = 0usize;
        // SAFETY: the session is live; every pointer in `args` outlives the call.
        let p = unsafe { (self.rt.run)(self.s, &args, &mut len) };
        if p.is_null() {
            // SAFETY: the runtime's last error is a NUL-terminated string.
            let e = unsafe { CStr::from_ptr((self.rt.last_error)()) }.to_string_lossy().into_owned();
            return Err(format!("the runtime refused the run: {e}"));
        }
        // SAFETY: the runtime returned `len` bytes at `p`, freed below with its own function.
        let v = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
        // SAFETY: as above.
        unsafe { (self.rt.free_bytes)(p, len) };
        Ok(v)
    }

    /// How many devices the session has made.
    pub fn devices_made(&self) -> u64 {
        // SAFETY: the session is live.
        unsafe { (self.rt.devices_made)(self.s) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session came from session_new and is not used again.
        unsafe { (self.rt.session_free)(self.s) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fuller_futhark_rust_{name}_{}", std::process::id()));
        if d.exists() {
            std::fs::remove_dir_all(&d).expect("remove a stale scratch directory");
        }
        d
    }

    #[cfg(feature = "gpu")]
    /// One `o` record's payload (the entry's output in Futhark's binary format), and the `t` record's per-run
    /// nanoseconds, from a run's records.
    fn payloads(raw: &[u8]) -> Vec<(String, bool, u64, Vec<u8>)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            let nl = raw[i..].iter().position(|b| *b == b'\n').expect("a record's end of line") + i;
            let head = std::str::from_utf8(&raw[i..nl]).expect("a text record");
            let f: Vec<&str> = head.split(' ').collect();
            let n: usize = f[5].parse().expect("a payload length");
            out.push((f[0].to_string(), f[2] == "ok", f[3].parse().expect("nanoseconds"), raw[nl + 1..nl + 1 + n].to_vec()));
            i = nl + 1 + n;
        }
        out
    }

    /// THE BENCHMARKS' GATE, on the GPU: every case of the four integer benchmarks (reduce, scan, histogram,
    /// pathfinder), through the rust-wgpu path (the published program compiled to Rust, run in this process), gives
    /// output bytes identical to Futhark's C backend (the corpus's `.out`, written by `futhark c`) and to the shim path
    /// (the branch compiler's webgpu C over the shim, its own process). The k-means step is not repeatable on the GPU
    /// on either path (f32 atomics), so on its corpus cases each path's every run is held to a derived bound from the
    /// exact step on the CPU ([`kmeans_bound`]); its bit gate is on three inputs
    /// of multiples of 1/256, whose output is one in any order of the additions: the rust path gives the shim's bytes
    /// and each repeats itself, three runs each. It prints each benchmark's GPU time
    /// a call on both paths (the fastest of 20 runs after a warm-up, GPU timestamps) and the rust path's wall time a
    /// call. (The fork compiler, rustc, the branch compiler, cc, the GPU; run under the GPU lock.)
    #[cfg(feature = "gpu")]
    #[test]
    fn on_the_gpu_the_benchmarks_equal_futhark_c_and_the_shim() {
        let branch = crate::futhark_wgpu_host::branch_compiler();
        if !compiler().is_file() || !branch.is_file() {
            eprintln!("skipped: the fork compiler ({}) or the branch compiler ({}) is not installed", compiler().display(), branch.display());
            return;
        }
        let _gpu = crate::gpu_test_lock::acquire();
        let base = scratch("benchmarks");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).expect("the shim");
        let session = Session::new(&rt, Duration::from_secs(120)).expect("a session");
        let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems");
        let runs = 20;
        let mut table = String::from("benchmark\tcases\tbit-identical to C\tbit-identical to the shim\trust GPU us/call\tshim GPU us/call\trust wall us/call\trust launches/call (last case)\tshim dispatches/call (last case)\n");
        // Futhark's binary format allows whitespace after a value, and an executable's standard output ends with a
        // newline: the values are compared, the trailing whitespace is not.
        let v = |b: &[u8]| b.trim_ascii_end().to_vec();
        for name in ["reduce", "scan", "histogram", "pathfinder", "kmeans"] {
            let dir = corpus.join(name);
            let yard = std::fs::read_to_string(dir.join("yardstick.fut")).expect("the yardstick");
            let (dylib, _) = compile_module(&yard, &base.join(format!("{name}_rust")), &rt, &format!("bench_{name}"), Duration::from_secs(600)).expect("the rust-wgpu compile");
            let exe = crate::futhark_wgpu_host::compile_program_prebuilt(&yard, &base.join(format!("{name}_shim")), &shim).expect("the shim compile");
            // The k-means corpus cases are real embeddings the bound cannot judge (`kmeans_bound_inputs`): the device
            // gate for k-means runs on the uniform inputs built to be judgeable; the corpus cases are the runner's.
            let mut inputs: Vec<PathBuf> = if name == "kmeans" { kmeans_bound_inputs(&base.join("kmeans_bound_inputs")) } else { std::fs::read_dir(&dir).expect("the corpus").map(|e| e.expect("an entry").path()).filter(|p| p.extension().is_some_and(|x| x == "in")).collect() };
            inputs.sort();
            let (mut same_c, mut same_shim) = (0, 0);
            let (mut rust_gpu, mut shim_gpu, mut rust_wall) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let (mut rust_launches, mut shim_dispatches) = (0.0, 0.0);
            let mut kmeans_repeatable = 0;
            let mut kmeans_worst_seen = 0f64;
            // The answer each case is held to: the stored one, or (k-means, which stores none: it is scored on its
            // objective) the yardstick's through `futhark c`.
            let c_exe = (name == "kmeans").then(|| crate::futhark_sig::compile_c(&yard, &base.join(format!("{name}_c")), ).expect("futhark c on the yardstick"));
            for input in &inputs {
                let stdin = std::fs::read(input).expect("a case's input");
                let want = match &c_exe {
                    Some(exe) => crate::futhark_sig::write_binary(&[&crate::futhark_sig::run_exe(exe, &stdin).expect("the yardstick's run")]),
                    None => std::fs::read(input.with_extension("out")).expect("a case's output"),
                };
                let raw = session.run(&dylib, &format!("o main\ns\nt main {runs}\ns\n"), &[], 0, 0, Some(&stdin)).expect("the rust path's run");
                let recs = payloads(&raw);
                let o = recs.iter().find(|r| r.0 == "o").expect("an output record");
                assert!(o.1, "{name} {}: the rust path failed: {}", input.display(), String::from_utf8_lossy(&o.3));
                let t = recs.iter().find(|r| r.0 == "t").expect("a time record");
                assert!(t.1, "{name} {}: the rust path's timing failed: {}", input.display(), String::from_utf8_lossy(&t.3));
                // A run with no GPU pass (a case so small the program launches nothing) is `none`: no time to take.
                let times: Vec<f64> = String::from_utf8_lossy(&t.3).split(' ').skip(1).filter(|w| *w != "none").map(|w| w.parse().expect("a GPU time")).collect();
                rust_gpu = rust_gpu.min(times.iter().cloned().fold(f64::INFINITY, f64::min) / 1e3);
                rust_wall = rust_wall.min(t.2 as f64 / (runs + 1) as f64 / 1e3);
                let (shim_out, _) = crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, false, 1).expect("the shim path's run");
                let (_, report) = crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, true, runs).expect("the shim path's timed run");
                // Launches a call on each path: the rust path's counts across its timed runs, the shim's dispatches
                // across its process's runs (`-r runs`, and a warm-up).
                let counts: Vec<String> = recs.iter().filter(|r| r.0 == "s").map(|r| String::from_utf8_lossy(&r.3).to_string()).collect();
                let launches = |c: &str| c.split(' ').nth(1).and_then(|w| w.parse::<u64>().ok()).unwrap_or(0);
                if let [a, b] = counts.as_slice() {
                    rust_launches = (launches(b) - launches(a)) as f64 / (runs + 1) as f64;
                }
                shim_dispatches = report.dispatches as f64 / (runs + 1) as f64;
                // A case with no GPU pass has no GPU time on the shim either (an error there, not a time).
                if let Ok(ns) = report.fastest_run_ns(runs) {
                    shim_gpu = shim_gpu.min(ns / 1e3);
                }
                if v(&shim_out) != v(&o.3) {
                    eprintln!("{name} {}: the shim's {} bytes against the rust path's {}: first difference at {:?}", input.display(), shim_out.len(), o.3.len(), shim_out.iter().zip(&o.3).position(|(a, b)| a != b));
                }
                same_c += usize::from(v(&o.3) == v(&want));
                same_shim += usize::from(v(&o.3) == v(&shim_out));
                if name == "kmeans" {
                    // The k-means step's cluster sums are f32 additions done with atomics: not repeatable on either
                    // path (on_the_gpu_the_kmeans_step_s_repeatability_on_each_path), so a corpus case has no one
                    // output to compare. Three equal runs do not make a case repeatable (measured 2026-11-01:
                    // train_n200_d4_k5_s2 repeated three times on the rust path and not on the shim, and in the merge
                    // run the shim repeated train_n200_d4_k5_s1 three times while the rust path gave another value 1
                    // ulp away). Counted and printed here; the exact gate is on inputs whose answer is one in any
                    // order of the additions (kmeans_exact_input, below).
                    let shim3: Vec<Vec<u8>> = (0..3).map(|_| v(&crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, false, 1).expect("the shim's run").0)).collect();
                    if shim3.iter().all(|x| *x == shim3[0]) {
                        kmeans_repeatable += 1;
                    }
                    // Where rounding happens: each path's every run within kmeans_bound of the exact step (f64, CPU).
                    let (exact, bound) = kmeans_bound(&stdin);
                    let mut runs3 = vec![v(&o.3)];
                    for _ in 0..2 {
                        let raw = session.run(&dylib, "o main\n", &[], 0, 0, Some(&stdin)).expect("the rust path's run");
                        runs3.push(v(&payloads(&raw).into_iter().find(|r| r.0 == "o" && r.1).expect("an output").3));
                    }
                    for (path, outs) in [("rust", &runs3), ("shim", &shim3)] {
                        for (run, out) in outs.iter().enumerate() {
                            let worst = kmeans_worst(out, &exact, &bound);
                            kmeans_worst_seen = kmeans_worst_seen.max(worst);
                            assert!(worst <= 1.0, "kmeans {}: the {path} path's run {run} is {worst} times the bound from the exact step", input.display());
                        }
                    }
                }
            }
            if name == "kmeans" {
                // THE K-MEANS GATE, exact: points and centres multiples of 1/256 in [0, 1), so every distance, every
                // cluster sum and every count is exact in f32 in any order of the additions, and the step's output is
                // one set of bits on the device: the rust path and the shim must each give it, three runs each. Not
                // futhark c's bits: the device's f32 division (a centre is a sum over a count) is not correctly
                // rounded (WGSL allows 2.5 ulp), measured 2026-11-01: 1 ulp from C on several centres.
                for (n, d, k, seed) in [(200usize, 4usize, 5usize, 1u64), (500, 8, 10, 3), (777, 6, 7, 4)] {
                    let stdin = kmeans_exact_input(n, d, k, seed);
                    let want = v(&crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, false, 1).expect("the shim's run").0);
                    for run in 0..3 {
                        let raw = session.run(&dylib, "o main\n", &[], 0, 0, Some(&stdin)).expect("the rust path's run");
                        let r = payloads(&raw).into_iter().find(|r| r.0 == "o" && r.1).expect("an output").3;
                        assert_eq!(v(&r), want, "kmeans exact n {n} d {d} k {k}, run {run}: the rust path differs from the shim's first run");
                        let shim = v(&crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, false, 1).expect("the shim's run").0);
                        assert_eq!(shim, want, "kmeans exact n {n} d {d} k {k}, run {run}: the shim does not repeat itself");
                    }
                }
            }
            table.push_str(&format!("{name}\t{}\t{same_c}\t{same_shim}\t{rust_gpu:.1}\t{shim_gpu:.1}\t{rust_wall:.1}\t{rust_launches:.1}\t{shim_dispatches:.1}\n", inputs.len()));
            if name == "kmeans" {
                table.push_str(&format!("kmeans: {kmeans_repeatable} of {} cases gave one output over three shim runs; every run of both paths within the bound of the exact step (worst {kmeans_worst_seen:.3} of it); 3 exact-input cases one output on both paths, three runs each\n", inputs.len()));
            } else {
                assert_eq!(same_shim, inputs.len(), "{name}: the rust path and the shim path differ\n{table}");
                assert_eq!(same_c, inputs.len(), "{name}: the rust path and futhark c differ\n{table}");
            }
        }
        eprintln!("{table}devices made: {}", session.devices_made());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The 2-D f32 arrays of a Futhark binary value stream (`b`, version 2, rank 2, ` f32`, two i64 dimensions, the
    /// data), in order: (rows, columns, values). Trailing whitespace is allowed.
    fn f32_matrices(mut b: &[u8]) -> Vec<(usize, usize, Vec<f32>)> {
        let mut out = Vec::new();
        loop {
            b = b.trim_ascii_start();
            if b.is_empty() {
                return out;
            }
            assert!(b.len() >= 23 && b[0] == b'b' && b[1] == 2 && b[2] == 2 && &b[3..7] == b" f32", "not a [][]f32 value: {:?}", &b[..b.len().min(8)]);
            let dim = |i: usize| i64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes")) as usize;
            let (r, c) = (dim(7), dim(15));
            let vals: Vec<f32> = b[23..23 + 4 * r * c].chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().expect("4 bytes"))).collect();
            out.push((r, c, vals));
            b = &b[23 + 4 * r * c..];
        }
    }

    /// gamma_k = k u / (1 - k u), the bound on the relative error of k floating-point operations of unit roundoff u
    /// (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1).
    fn gamma(k: usize, u: f64) -> f64 {
        let ku = k as f64 * u;
        assert!(ku < 1.0);
        ku / (1.0 - ku)
    }

    /// THE K-MEANS STEP'S BOUND: the exact step on the CPU (f64) and, centre by centre and coordinate by coordinate, how
    /// far a device's f32 result may lie from it. `input` is the step's input (points, centres); returns the exact
    /// centres and the bound, each `k * d` values.
    ///
    /// Derivation (u = 2^-24, f32's unit roundoff):
    /// * MEMBERSHIP must be the exact one. A squared distance in f32 is d differences, d squares and a sum of d terms in
    ///   some order (a fused multiply-add only removes roundings): relative error at most gamma_{d+2}(u). So the
    ///   nearest centre is unambiguous when the exact nearest D1 and second nearest D2 satisfy
    ///   D2 - D1 > (gamma_{d+2}(u) + gamma_{d+2}(2^-53)) (D1 + D2) (the second term the f64 reference's own error).
    ///   An input that fails this for any point panics: it cannot be judged by a bound.
    /// * A CLUSTER SUM of m points, in ANY order (atomics, a tree, a fold): |S^ - S| <= gamma_{m-1}(u) sum |x_i|
    ///   (Higham §4.2). The count m is exact in f32 (m < 2^24).
    /// * THE DIVISION c^ = S^ (/) m on the device: WGSL allows 2.5 ulp for f32 `/`, and ulp(y) <= 2^-23 |y| for a
    ///   normal y, so |c^ - S^/m| <= 2.5 * 2^-23 |S^/m|.
    /// * Together, with e1 = gamma_{m-1}(u) sum |x_i| / m: |c^ - S/m| <= e1 + 2.5 * 2^-23 (|S/m| + e1), plus the f64
    ///   reference's own error gamma_{m+1}(2^-53) (sum |x_i| / m + |S/m|). An empty cluster's centre is exactly 0.
    fn kmeans_bound(input: &[u8]) -> (Vec<f64>, Vec<f64>) {
        let m = f32_matrices(input);
        assert_eq!(m.len(), 2, "a k-means input is points then centres");
        let ((n, d, pts), (k, d2, cen)) = (&m[0], &m[1]);
        assert_eq!(d, d2);
        let (n, d, k) = (*n, *d, *k);
        let (u32_, u64_) = (2f64.powi(-24), 2f64.powi(-53));
        let margin = gamma(d + 2, u32_) + gamma(d + 2, u64_);
        let mut sum = vec![0f64; k * d];
        let mut abs = vec![0f64; k * d];
        let mut count = vec![0usize; k];
        for i in 0..n {
            let p = &pts[i * d..(i + 1) * d];
            let dist: Vec<f64> = (0..k).map(|j| (0..d).map(|t| (f64::from(p[t]) - f64::from(cen[j * d + t])).powi(2)).sum()).collect();
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|a, b| dist[*a].total_cmp(&dist[*b]).then(a.cmp(b)));
            if k > 1 {
                let (d1, d2) = (dist[order[0]], dist[order[1]]);
                assert!(d2 - d1 > margin * (d1 + d2), "point {i}: its nearest centres are {d1} and {d2} apart by less than the f32 distance bound; the input cannot be judged by a bound");
            }
            let j = order[0];
            count[j] += 1;
            for t in 0..d {
                sum[j * d + t] += f64::from(p[t]);
                abs[j * d + t] += f64::from(p[t]).abs();
            }
        }
        let mut exact = vec![0f64; k * d];
        let mut bound = vec![0f64; k * d];
        for (j, &mc) in count.iter().enumerate() {
            if mc == 0 {
                continue;
            }
            for t in 0..d {
                let x = j * d + t;
                exact[x] = sum[x] / mc as f64;
                let e1 = gamma(mc.saturating_sub(1), u32_) * abs[x] / mc as f64;
                bound[x] = e1 + 2.5 * 2f64.powi(-23) * (exact[x].abs() + e1) + gamma(mc + 1, u64_) * (abs[x] / mc as f64 + exact[x].abs());
            }
        }
        (exact, bound)
    }

    /// The largest |c^ - c| / bound of a k-means output against [`kmeans_bound`]: at most 1 within the bound.
    fn kmeans_worst(output: &[u8], exact: &[f64], bound: &[f64]) -> f64 {
        let m = f32_matrices(output);
        assert_eq!(m.len(), 1, "a k-means output is one [k][d]f32");
        assert_eq!(m[0].2.len(), exact.len(), "the output's size");
        m[0].2.iter().zip(exact.iter().zip(bound)).map(|(g, (e, b))| {
            let gap = (f64::from(*g) - e).abs();
            if *b == 0.0 { if gap == 0.0 { 0.0 } else { f64::INFINITY } } else { gap / b }
        }).fold(0.0, f64::max)
    }

    /// THE INPUTS THE K-MEANS BOUND CAN JUDGE: uniform random f32 points and centres in [0, 1) at the shapes of the
    /// uniform cases of 2026-11-03 09:27Z (n x d x k: 200 x 4 x 5, 300 x 6 x 8, 400 x 7 x 6, 500 x 3 x 6, 2,000 x 4 x
    /// 2), from a fixed xorshift seed, written to `dir` as `<name>.in` (Futhark's binary format: points, then
    /// centres). The k-means corpus cases are real embeddings (the dbpedia sample, `docs/KMEANS_FITNESS.md`) and
    /// CANNOT be judged by the bound: measured 2026-11-03 on `train_n200_k4`, point 207's two nearest centres are
    /// 0.48775 and 0.48783 away, closer than the f32 distance margin, so no order of the additions is "wrong" there,
    /// which is why k-means is scored on its objective and not against a bound. The bound stays a tool of the device
    /// checks on inputs built to be judgeable.
    fn kmeans_bound_inputs(dir: &Path) -> Vec<PathBuf> {
        std::fs::create_dir_all(dir).expect("mkdir");
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = |count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 40) as f32 / 16_777_216.0
                })
                .collect()
        };
        let mut out = Vec::new();
        for (n, d, k) in [(200usize, 4usize, 5usize), (300, 6, 8), (400, 7, 6), (500, 3, 6), (2000, 4, 2)] {
            let points = crate::futhark_sig::Array::f32s(vec![n, d], draw(n * d));
            let centres = crate::futhark_sig::Array::f32s(vec![k, d], draw(k * d));
            let path = dir.join(format!("uniform_n{n}_d{d}_k{k}.in"));
            std::fs::write(&path, crate::futhark_sig::write_binary(&[&points, &centres])).expect("write an input");
            out.push(path);
        }
        out
    }

    /// THE K-MEANS BOUND HOLDS FOR FUTHARK C (CPU, no device): every input of [`kmeans_bound_inputs`] has an
    /// unambiguous membership under the f32 distance bound, and futhark c's output of the published step (the
    /// yardstick compiled here; IEEE division, 0.5 ulp) lies within [`kmeans_bound`] of the exact step. The GPU check
    /// (`on_the_gpu_the_benchmarks_equal_futhark_c_and_the_shim`) holds each device path to the same bound on the
    /// same inputs. The bound is a tool of the device checks, not of the fitness; the corpus cases (real embeddings)
    /// cannot be judged by it (see [`kmeans_bound_inputs`]).
    #[test]
    fn futhark_c_s_kmeans_steps_lie_within_the_bound_of_the_exact_step() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems/kmeans");
        let inputs = kmeans_bound_inputs(&scratch("kmeans_bound_inputs"));
        assert!(inputs.len() >= 5, "{} inputs", inputs.len());
        let yard = std::fs::read_to_string(dir.join("yardstick.fut")).expect("the yardstick");
        let exe = crate::futhark_sig::compile_c(&yard, &scratch("kmeans_bound_yardstick")).expect("futhark c on the yardstick");
        for input in &inputs {
            let stdin = std::fs::read(input).expect("an input");
            let (exact, bound) = kmeans_bound(&stdin);
            let out = crate::futhark_sig::run_exe(&exe, &stdin).expect("the yardstick's run");
            let worst = kmeans_worst(&crate::futhark_sig::write_binary(&[&out]), &exact, &bound);
            eprintln!("{}: futhark c at {worst:.3} of the bound", input.display());
            assert!(worst <= 1.0, "{}: futhark c's output is {worst} times the bound from the exact step", input.display());
        }
    }

    /// A k-means step's input (Futhark's binary format: `[n][d]f32` points, then `[k][d]f32` centres, the first k
    /// points as the published program starts) whose every value is a multiple of 1/256 in [0, 1), from a fixed
    /// seed. A squared distance is a sum of d <= 8 products of 8-bit fractions and a cluster sum at most n < 2^10 of
    /// them: each fits f32's 24 bits, so the step's output does not depend on the order of its additions (only the
    /// final division rounds).
    #[cfg(feature = "gpu")]
    fn kmeans_exact_input(n: usize, d: usize, k: usize, seed: u64) -> Vec<u8> {
        assert!(n < 1024 && d <= 8 && k <= n);
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let pts: Vec<f32> = (0..n * d)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 56) as f32 / 256.0
            })
            .collect();
        let array = |rows: usize, vals: &[f32]| -> Vec<u8> {
            let mut b = vec![b'b', 2, 2];
            b.extend_from_slice(b" f32");
            b.extend_from_slice(&(rows as i64).to_le_bytes());
            b.extend_from_slice(&(d as i64).to_le_bytes());
            vals.iter().for_each(|f| b.extend_from_slice(&f.to_le_bytes()));
            b
        };
        [array(n, &pts), array(k, &pts[..k * d])].concat()
    }

    /// THE A7 PROGRAMS on the GPU (`futhark-wgpu-shim/tests/s5/a7_*.fut`, which Futhark's webgpu backend does not
    /// compile: SetMem in a kernel loop, a register array sized at run time): both compile to Rust+WGSL and run in
    /// this process. Measured against `futhark c` on seeded inputs (`futhark dataset -s 7`, nine 64x64 f32 matrices):
    /// the count of bit-identical values and the largest gap in units in the last place, printed. Not asserted equal:
    /// f32 multiply-adds on Metal are contracted (fast math; the shim's S5 measured it), which is to be measured.
    /// (The fork compiler, rustc, the released futhark, the GPU; run under the GPU lock.)
    #[cfg(feature = "gpu")]
    #[test]
    fn on_the_gpu_the_a7_programs_run_through_the_rust_path() {
        if !compiler().is_file() {
            eprintln!("skipped: the fork compiler is not installed");
            return;
        }
        let _gpu = crate::gpu_test_lock::acquire();
        let base = scratch("a7");
        std::fs::create_dir_all(&base).expect("a scratch directory");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let session = Session::new(&rt, Duration::from_secs(120)).expect("a session");
        let mut ds = Command::new("futhark");
        ds.args(["dataset", "-s", "7", "-b"]);
        for _ in 0..9 {
            ds.args(["-g", "[64][64]f32"]);
        }
        let input = output_with_timeout(&mut ds, Duration::from_secs(60), None).expect("futhark dataset").stdout;
        let s5 = Path::new(env!("CARGO_MANIFEST_DIR")).join("futhark-wgpu-shim/tests/s5");
        for name in ["a7_reference", "a7_winner"] {
            let src = std::fs::read_to_string(s5.join(format!("{name}.fut"))).expect("the program");
            let c = crate::futhark_sig::compile_c(&src, &base.join(format!("{name}_c"))).expect("futhark c");
            let want = crate::futhark_exec::output_with_timeout_bytes(Command::new(&c).arg("-b"), Duration::from_secs(120), Some(&input)).expect("the C run").stdout;
            let (dylib, _) = compile_module(&src, &base.join(format!("{name}_rust")), &rt, &format!("a7_{name}"), Duration::from_secs(600)).expect("the rust-wgpu compile");
            let raw = session.run(&dylib, "o main\n", &[], 0, 0, Some(&input)).expect("the rust path's run");
            let recs = payloads(&raw);
            let failed: Vec<String> = recs.iter().filter(|r| !r.1).map(|r| format!("{}: {}", r.0, String::from_utf8_lossy(&r.3))).collect();
            let o = recs.into_iter().find(|r| r.0 == "o").unwrap_or_else(|| panic!("{name}: no output record; the failures: {failed:?}"));
            assert!(o.1, "{name}: the rust path failed: {}", String::from_utf8_lossy(&o.3));
            // The C program's standard output ends with a newline, which the binary format allows after a value.
            let want = want.trim_ascii_end();
            assert_eq!(o.3.len(), want.len(), "{name}: the output's size");
            // Past the 7-byte header and two dimensions: the f32 values.
            let (g, w) = (&o.3[23..], &want[23..]);
            let vals = |b: &[u8]| b.chunks(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
            let (g, w) = (vals(g), vals(w));
            let same = g.iter().zip(&w).filter(|(a, b)| a == b).count();
            let ulp = g.iter().zip(&w).map(|(a, b)| (i64::from(*a) - i64::from(*b)).abs()).max().unwrap_or(0);
            eprintln!("{name}: {same} of {} values bit-identical to futhark c, the largest gap {ulp} ulp", w.len());
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// THE K-MEANS STEP'S DETERMINISM on the GPU: each path (rust in this process, the shim in its own process) runs the
    /// published k-means step three times on each stored case, and the test prints whether each path gives the same
    /// bytes every time, and whether the two paths agree. The cluster sums are a histogram of f32 additions, which
    /// Futhark's GPU code does with atomics, so their order, and the f32 result, may change from run to run on either
    /// path: this is what the benchmarks' gate needs to know before it compares the paths. (The fork compiler,
    /// rustc, the branch compiler, cc, the GPU; run under the GPU lock.)
    #[cfg(feature = "gpu")]
    #[test]
    fn on_the_gpu_the_kmeans_step_s_repeatability_on_each_path() {
        let branch = crate::futhark_wgpu_host::branch_compiler();
        if !compiler().is_file() || !branch.is_file() {
            eprintln!("skipped: the fork compiler or the branch compiler is not installed");
            return;
        }
        let _gpu = crate::gpu_test_lock::acquire();
        let base = scratch("kmeans_repeat");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).expect("the shim");
        let session = Session::new(&rt, Duration::from_secs(120)).expect("a session");
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_problems/kmeans");
        let yard = std::fs::read_to_string(dir.join("yardstick.fut")).expect("the yardstick");
        let (dylib, _) = compile_module(&yard, &base.join("rust"), &rt, "kmeans_repeat", Duration::from_secs(600)).expect("the rust-wgpu compile");
        let exe = crate::futhark_wgpu_host::compile_program_prebuilt(&yard, &base.join("shim_build"), &shim).expect("the shim compile");
        let mut inputs: Vec<PathBuf> = std::fs::read_dir(&dir).expect("the corpus").map(|e| e.expect("an entry").path()).filter(|p| p.extension().is_some_and(|x| x == "in")).collect();
        inputs.sort();
        for input in &inputs {
            let stdin = std::fs::read(input).expect("a case's input");
            let rust: Vec<Vec<u8>> = (0..3)
                .map(|_| {
                    let raw = session.run(&dylib, "o main\n", &[], 0, 0, Some(&stdin)).expect("the rust path's run");
                    payloads(&raw).into_iter().find(|r| r.0 == "o" && r.1).expect("an output").3
                })
                .collect();
            let shim_out: Vec<Vec<u8>> = (0..3).map(|_| crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin, false, 1).expect("the shim's run").0.trim_ascii_end().to_vec()).collect();
            eprintln!(
                "{}: rust repeats itself {}, the shim repeats itself {}, the paths agree on the first run {}",
                input.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                rust.iter().all(|r| *r == rust[0]),
                shim_out.iter().all(|r| *r == shim_out[0]),
                rust[0] == shim_out[0]
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The prebuilt runtime loads in this process (dlopen: its libstd found by its rpath) and makes a session. No device:
    /// a session makes one only when a command runs.
    #[test]
    fn the_runtime_loads_in_process() {
        let base = scratch("load");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let s = Session::new(&rt, Duration::from_secs(5)).expect("a session");
        assert_eq!(s.devices_made(), 0);
        drop(s);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// TWO RUNTIMES IN ONE PROCESS (2026-11-01): a module binds to the runtime it was compiled against. The runtime's
    /// install name was `@rpath/libfuthark_rust_rt.dylib` in every run directory, and dyld binds a module's dependency
    /// of that name to the first runtime of that name loaded in the process: a second test's module ran the first
    /// test's runtime (and its wgpu) on the second session's device, and every pipeline it made failed
    /// (`NegativeWorkgroupSize`, 7 GPU tests run in one process). Each runtime's install name is now its own path, and
    /// a module compiled against runtime B names B's path, not A's. CPU only (`otool`), no device.
    #[test]
    fn a_module_names_the_runtime_it_was_compiled_against() {
        let base = scratch("two_runtimes");
        let a = prebuild_rt(&base.join("rt_a")).expect("runtime A");
        let b = prebuild_rt(&base.join("rt_b")).expect("runtime B");
        let otool = |args: &[&str], p: &Path| -> Vec<String> {
            let out = Command::new("otool").args(args).arg(p).output().expect("otool");
            assert!(out.status.success(), "otool {args:?} {}: {}", p.display(), String::from_utf8_lossy(&out.stderr));
            String::from_utf8_lossy(&out.stdout).lines().skip(1).map(|l| l.trim().split(" (compatibility").next().unwrap_or("").to_string()).collect()
        };
        let lib = |d: &Path| std::fs::canonicalize(d.join("libfuthark_rust_rt.dylib")).expect("the runtime dylib").display().to_string();
        let (la, lb) = (lib(&a), lib(&b));
        assert_eq!(otool(&["-D"], Path::new(&la)), vec![la.clone()], "runtime A's install name must be its own path");
        assert_eq!(otool(&["-D"], Path::new(&lb)), vec![lb.clone()], "runtime B's install name must be its own path");
        if !compiler().is_file() {
            panic!("the fork compiler {} is not built: run futhark-rust-backend/build.sh", compiler().display());
        }
        let (dylib, _) = compile_module("entry main (n: i64) : []i32 = replicate n 7\n", &base.join("m"), &b, "two_runtimes_m", Duration::from_secs(600)).expect("the module");
        let deps: Vec<String> = otool(&["-L"], &dylib).into_iter().filter(|d| d.ends_with("libfuthark_rust_rt.dylib")).collect();
        assert_eq!(deps, vec![lb], "the module must name runtime B by its path (runtime A is {la})");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// THE COMPILE-FAILURE CORPUS (`tests/corpus/futhark_compile_failures`): every distinct program a legal gene
    /// expressed that the rust-wgpu path once failed to compile (from the evolution runs' `bugs.log`, `index.tsv`
    /// names each one's error and first failing stage), and the minimal reproducers, compiles through the rust path:
    /// the fork compiler's `rust-wgpu`, then `rustc -D warnings` against the runtime. CPU only, no device.
    #[test]
    fn every_program_of_the_compile_failure_corpus_compiles_through_the_rust_path() {
        assert_eq!(compiler(), default_compiler(), "FUTHARK_RUST_WGPU overrides the default compiler, which this test checks");
        assert!(compiler().is_file(), "the default compiler {} is not built: run futhark-rust-backend/build.sh", compiler().display());
        let files = compile_failure_corpus();
        let base = scratch("compile_failures");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed = Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(f) = files.get(i) else { break };
                    let source = std::fs::read_to_string(f).expect("a corpus program");
                    if let Err(e) = compile_module(&source, &base.join(format!("p{i}")), &rt, &format!("cf{i}"), Duration::from_secs(600)) {
                        let tail: Vec<&str> = e.lines().filter(|l| !l.starts_with("Warning at") && !l.trim_start().starts_with("Unused name") && !l.trim().is_empty()).take(6).collect();
                        failed.lock().expect("the failure list").push(format!("{}: {}", f.display(), tail.join(" | ")));
                    }
                });
            }
        });
        let failed = failed.into_inner().expect("the failure list");
        eprintln!("{} of {} corpus programs compiled through the rust path", files.len() - failed.len(), files.len());
        assert!(failed.is_empty(), "{} of {} corpus programs did not compile:\n{}", failed.len(), files.len(), failed.join("\n"));
        let _ = std::fs::remove_dir_all(&base);
    }

    /// THE GRID OVER WEBGPU'S LIMIT (series patch 0011): a 1-D grid over the device's 65,535 blocks a dimension is
    /// dispatched as a rectangle by the runtime (`futhark_rust_rt::program::split_grid`), and every generated kernel
    /// reads its own block count from the last field of its scalars and returns the excess blocks. This checks the
    /// compiler's side on the CPU: every generated kernel's table entry names that field's offset, no built-in
    /// kernel does, and the WGSL computes the flat block id from `workgroup_id` and `num_workgroups`. The GPU side
    /// (the program's values on 2^24 and 2^25 elements against `futhark c`) is `tests/corpus/futhark_rust/large_grid.fut`
    /// under the gpu feature.
    #[test]
    fn every_generated_kernel_reads_its_block_count_and_a_flat_block_id() {
        assert!(compiler().is_file(), "the default compiler {} is not built: run futhark-rust-backend/build.sh", compiler().display());
        let source = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_rust/large_grid.fut")).expect("the program");
        let dir = scratch("large_grid_text");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let fut = dir.join("prog.fut");
        std::fs::write(&fut, &source).expect("write");
        let out = Command::new(compiler()).arg("rust-wgpu").arg(&fut).arg("-o").arg(dir.join("prog")).output().expect("futhark rust-wgpu");
        assert!(out.status.success(), "futhark rust-wgpu: {}", String::from_utf8_lossy(&out.stderr));
        let prog = std::fs::read_to_string(dir.join("prog.rs")).expect("prog.rs");
        let entries: Vec<&str> = prog.split("rt::KernelInfo {").skip(1).collect();
        let generated = entries.iter().filter(|e| e.contains("builtin: None")).count();
        let builtin = entries.len() - generated;
        assert!(generated >= 1, "no generated kernel in prog.rs");
        for e in &entries {
            let head = e.split("},").next().unwrap_or(e);
            if head.contains("builtin: None") {
                assert!(head.contains("grid_offset: Some("), "a generated kernel without its grid field: {head}");
            } else {
                assert!(head.contains("grid_offset: None"), "a built-in kernel with a grid field: {head}");
            }
        }
        assert!(prog.contains("@builtin(num_workgroups)"), "the kernels do not read num_workgroups");
        assert!(prog.contains("let futhark_block_id") && prog.contains("futhark_grid_x"), "the kernels do not compute the flat block id against their block count");
        eprintln!("{generated} generated kernels and {builtin} built-in kernels checked");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE GRID OVER WEBGPU'S LIMIT, ON THE GPU (series patch 0011): `tests/corpus/futhark_rust/large_grid.fut` on
    /// 2^24 and 2^25 elements (65,536 and 131,072 blocks of 256), both entries, through the rust path against
    /// `futhark c`, byte for byte; before the patch the dispatch failed validation at 65,536 blocks. (The fork
    /// compiler, rustc, cc, the GPU; run under the GPU lock.)
    #[cfg(feature = "gpu")]
    #[test]
    fn on_the_gpu_a_grid_over_the_blocks_per_dimension_limit_equals_futhark_c() {
        if !compiler().is_file() {
            eprintln!("skipped: the fork compiler ({}) is not installed", compiler().display());
            return;
        }
        let _gpu = crate::gpu_test_lock::acquire();
        let base = scratch("large_grid");
        std::fs::create_dir_all(&base).expect("a scratch directory");
        let rt = prebuild_rt(&base.join("rt")).expect("the runtime");
        let session = Session::new(&rt, Duration::from_secs(300)).expect("a session");
        let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/futhark_rust/large_grid.fut")).expect("the program");
        let c = crate::futhark_sig::compile_c(&src, &base.join("c")).expect("futhark c");
        let (dylib, _) = compile_module(&src, &base.join("rust"), &rt, "large_grid", Duration::from_secs(600)).expect("the rust-wgpu compile");
        for (entry, n, want_sum) in [("main", 1u64 << 24, Some(50_331_645i64)), ("main", 1 << 25, Some(100_663_291)), ("mapped", 1 << 24, None), ("mapped", 1 << 25, None)] {
            // Futhark's binary format for an i64 scalar: 'b', version 2, 0 dimensions, the 4-byte type " i64", the value.
            let mut input = b"b\x02\x00 i64".to_vec();
            input.extend_from_slice(&n.to_le_bytes());
            let want = crate::futhark_exec::output_with_timeout_bytes(Command::new(&c).arg("-b").arg("-e").arg(entry), Duration::from_secs(300), Some(&input)).expect("the C run").stdout;
            let raw = session.run(&dylib, &format!("o {entry}\n"), &[], 0, 0, Some(&input)).expect("the rust path's run");
            let recs = payloads(&raw);
            let o = recs.iter().find(|r| r.0 == "o").unwrap_or_else(|| panic!("{entry} {n}: no output record: {recs:?}"));
            assert!(o.1, "{entry} {n}: the rust path failed: {}", String::from_utf8_lossy(&o.3));
            assert_eq!(o.3.as_slice(), want.trim_ascii_end(), "{entry} {n}: the rust path differs from futhark c");
            if let Some(sum) = want_sum {
                // Past the 7-byte scalar header: the value.
                assert_eq!(i64::from_le_bytes(o.3[7..15].try_into().expect("8 bytes")), sum, "{entry} {n}: the value");
            }
            eprintln!("{entry} on {n} elements: the rust path equals futhark c ({} bytes)", o.3.len());
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// THE DEFAULT COMPILER IS THE COMMITTED SERIES: the compiler a run uses with no `FUTHARK_RUST_WGPU` or
    /// `FUTHARK_BRANCH` exists, and `build.sh` built it from exactly `futhark-rust-backend/patches` (the copy beside
    /// it, name for name and byte for byte). A patch added to the series without a rebuild fails here.
    #[test]
    fn the_default_compiler_is_built_from_the_committed_patch_series() {
        let c = default_compiler();
        assert!(c.is_file(), "the default compiler {} is not built: run futhark-rust-backend/build.sh", c.display());
        let installed = installed_series(&c).unwrap_or_else(|e| panic!("{e}: rebuild with futhark-rust-backend/build.sh"));
        let committed = committed_series().unwrap();
        let names = |s: &[(String, Vec<u8>)]| s.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(&installed), names(&committed), "the default compiler's series is not the committed one: rebuild with futhark-rust-backend/build.sh");
        for ((n, a), (_, b)) in installed.iter().zip(&committed) {
            assert!(a == b, "patch {n} of the default compiler differs from the committed one: rebuild with futhark-rust-backend/build.sh");
        }
        assert_eq!(crate::futhark_wgpu_host::branch_compiler(), compiler(), "the webgpu path and the rust path use one compiler");
    }

    #[test]
    fn the_binding_index_is_read_from_the_generated_module() {
        let m = "    rt::KernelInfo {\n        scalars_binding: 12,\n        binding_indices: &[13, 1011, 14],\n    },\n        binding_indices: &[],\n";
        assert_eq!(max_binding_index(m), 1011);
    }
}
