//! What the shim saw: dispatches, submits, the wait paths taken, and — with
//! `FUTHARK_WGPU_SHIM_TIMESTAMPS=1` — every compute pass timed on the GPU.
//!
//! The GPU timestamps are the evidence that a kernel really ran (the plan's
//! acceptance criterion 2): a pipeline can be built without a dispatch, but a
//! pass's begin and end timestamps are written by the GPU when the pass
//! executes.
//!
//! The report goes to the file named by `FUTHARK_WGPU_SHIM_REPORT`, written
//! at process exit (Futhark's executables exit without freeing their
//! context, so exit is the one moment that always comes). Without the
//! variable nothing is written.
//!
//! **The stage clocks** (wall time, UNIX nanoseconds): the instance's creation
//! (the runtime's first call into the shim), the device's readiness, the time
//! spent creating shader modules and compute pipelines (Metal compiles them
//! there), and the report's own writing at exit. With the caller's spawn time
//! they split a process into its start, its device setup, its pipelines and
//! its runs (`fuller::futhark_wgpu_host::RunStages`).
//!
//! **The pass query** ([`futhark_wgpu_shim_timed_passes`],
//! [`futhark_wgpu_shim_pass_ns`], [`futhark_wgpu_shim_dispatches`]): the
//! shim's OWN functions, not WebGPU's, for a host that runs many entry points
//! in one process (fuller's batch driver, `fuller::futhark_batch`) and must
//! know which passes were one call's. Timed passes are recorded inside
//! `wgpuQueueSubmit`, which waits for them, so after a `futhark_context_sync`
//! every pass of the calls before it is here. They are left out of the
//! generated `webgpu.h` (`cbindgen.toml`), which stays the WebGPU surface and
//! nothing else; the driver declares them itself.

use std::fmt::Write as _;
use std::sync::{Mutex, Once};

/// A compute pass recorded with GPU timestamps, waiting to be read back.
pub struct TimedPass {
    pub queries: wgpu::QuerySet,
    pub resolve: wgpu::Buffer,
    pub read: wgpu::Buffer,
    pub dispatches: u64,
}

struct Stats {
    adapter: String,
    backend: String,
    dispatches: u64,
    workgroups: u64,
    submits: u64,
    spontaneous_futures: u64,
    wait_any_only_futures: u64,
    wait_any_calls: u64,
    emscripten_sleep_calls: u64,
    /// The stage clocks (UNIX ns; 0 until reached) and the pipeline time.
    instance_unix_ns: u128,
    device_unix_ns: u128,
    pipeline_ns: u128,
    pipelines: u64,
    /// Compute pipelines answered from the device's cache (`handles::WGPUDeviceImpl::pipelines`).
    pipelines_cached: u64,
    /// Each timed pass: its GPU begin and end ticks, its GPU nanoseconds, and
    /// the dispatches in it.
    timed: Vec<(u64, u64, f64, u64)>,
}

static STATS: Mutex<Stats> = Mutex::new(Stats {
    adapter: String::new(),
    backend: String::new(),
    dispatches: 0,
    workgroups: 0,
    submits: 0,
    spontaneous_futures: 0,
    wait_any_only_futures: 0,
    wait_any_calls: 0,
    emscripten_sleep_calls: 0,
    instance_unix_ns: 0,
    device_unix_ns: 0,
    pipeline_ns: 0,
    pipelines: 0,
    pipelines_cached: 0,
    timed: Vec::new(),
});

/// Now, in UNIX nanoseconds (0 for a clock before 1970).
fn unix_ns() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
}

/// The instance was created: the runtime's first call into the shim.
pub fn instance_created() {
    with(|s| {
        if s.instance_unix_ns == 0 {
            s.instance_unix_ns = unix_ns();
        }
    });
}

/// The device is ready.
pub fn device_ready() {
    with(|s| s.device_unix_ns = unix_ns());
}

/// A shader module or a compute pipeline took `ns` to create.
pub fn count_pipeline(ns: u128) {
    with(|s| {
        s.pipeline_ns += ns;
        s.pipelines += 1;
    });
}

/// A compute pipeline was answered from the device's cache.
pub fn count_cached_pipeline() {
    with(|s| s.pipelines_cached += 1);
}

/// The nanoseconds spent creating shader modules and compute pipelines so far (a batch host reads it around each
/// call, so a call's pipeline creation is not counted as its run).
#[no_mangle]
pub extern "C" fn futhark_wgpu_shim_pipeline_ns() -> u64 {
    with(|s| u64::try_from(s.pipeline_ns).unwrap_or(u64::MAX))
}

/// The timed passes recorded so far.
#[no_mangle]
pub extern "C" fn futhark_wgpu_shim_timed_passes() -> u64 {
    with(|s| s.timed.len() as u64)
}

/// The GPU nanoseconds of timed pass `index`, or -1 when there is no such pass or the GPU did not really time it
/// (a begin of zero, or an end not after the begin: no evidence the pass ran).
#[no_mangle]
pub extern "C" fn futhark_wgpu_shim_pass_ns(index: u64) -> f64 {
    with(|s| match usize::try_from(index).ok().and_then(|i| s.timed.get(i)) {
        Some(&(begin, end, ns, _)) if begin > 0 && end > begin => ns,
        _ => -1.0,
    })
}

/// The dispatches so far.
#[no_mangle]
pub extern "C" fn futhark_wgpu_shim_dispatches() -> u64 {
    with(|s| s.dispatches)
}

fn with<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    let mut g = STATS.lock().unwrap_or_else(|p| p.into_inner());
    f(&mut g)
}

pub fn set_adapter(info: &wgpu::AdapterInfo) {
    with(|s| {
        s.adapter = info.name.clone();
        s.backend = format!("{:?}", info.backend);
    });
}
pub fn count_dispatch(x: u32, y: u32, z: u32) {
    with(|s| {
        s.dispatches += 1;
        s.workgroups += u64::from(x) * u64::from(y) * u64::from(z);
    });
}
pub fn count_submit() {
    with(|s| s.submits += 1);
}
pub fn count_future(spontaneous: bool) {
    with(|s| {
        if spontaneous {
            s.spontaneous_futures += 1
        } else {
            s.wait_any_only_futures += 1
        }
    });
}
pub fn count_wait_any() {
    with(|s| s.wait_any_calls += 1);
}
pub fn count_sleep() {
    with(|s| s.emscripten_sleep_calls += 1);
}
pub fn record_pass(begin: u64, end: u64, ns: f64, dispatches: u64) {
    with(|s| s.timed.push((begin, end, ns, dispatches)));
}

/// Whether passes are timed on the GPU (`FUTHARK_WGPU_SHIM_TIMESTAMPS=1`).
pub fn timestamps_wanted() -> bool {
    std::env::var("FUTHARK_WGPU_SHIM_TIMESTAMPS").is_ok_and(|v| v == "1")
}

/// The report as one JSON object.
fn json() -> String {
    with(|s| {
        let mut out = String::new();
        let esc = |t: &str| t.replace('\\', "\\\\").replace('"', "\\\"");
        let total: f64 = s.timed.iter().map(|t| t.2).sum();
        let timed_dispatches: u64 = s.timed.iter().map(|t| t.3).sum();
        let _ = write!(
            out,
            "{{\"adapter\":\"{}\",\"backend\":\"{}\",\"dispatches\":{},\"workgroups\":{},\"submits\":{},\
             \"spontaneous_futures\":{},\"wait_any_only_futures\":{},\"wait_any_calls\":{},\
             \"emscripten_sleep_calls\":{},\"instance_unix_ns\":{},\"device_unix_ns\":{},\"pipeline_ns\":{},\"pipelines\":{},\"pipelines_cached\":{},\
             \"exit_unix_ns\":{},\"timed_passes\":{},\"timed_dispatches\":{},\"gpu_ns_total\":{:.1},\"passes\":[",
            esc(&s.adapter),
            esc(&s.backend),
            s.dispatches,
            s.workgroups,
            s.submits,
            s.spontaneous_futures,
            s.wait_any_only_futures,
            s.wait_any_calls,
            s.emscripten_sleep_calls,
            s.instance_unix_ns,
            s.device_unix_ns,
            s.pipeline_ns,
            s.pipelines,
            s.pipelines_cached,
            unix_ns(),
            s.timed.len(),
            timed_dispatches,
            total
        );
        for (i, (begin, end, ns, n)) in s.timed.iter().enumerate() {
            let _ = write!(
                out,
                "{}{{\"begin_tick\":{begin},\"end_tick\":{end},\"gpu_ns\":{ns:.1},\"dispatches\":{n}}}",
                if i == 0 { "" } else { "," }
            );
        }
        out.push_str("]}\n");
        out
    })
}

extern "C" fn write_report() {
    if let Ok(path) = std::env::var("FUTHARK_WGPU_SHIM_REPORT") {
        if let Err(e) = std::fs::write(&path, json()) {
            eprintln!("futhark-wgpu-shim: could not write the report to {path}: {e}");
        }
    }
}

extern "C" {
    fn atexit(f: extern "C" fn()) -> std::ffi::c_int;
}

/// Write the report at process exit (registered once).
pub fn at_exit() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: `atexit` is the C library's; `write_report` is an
        // `extern "C" fn()` that lives for the whole program.
        if unsafe { atexit(write_report) } != 0 {
            crate::fatal("atexit could not register the report writer");
        }
    });
}
