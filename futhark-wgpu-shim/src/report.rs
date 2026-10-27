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
    timed: Vec::new(),
});

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
             \"emscripten_sleep_calls\":{},\"timed_passes\":{},\"timed_dispatches\":{},\"gpu_ns_total\":{:.1},\"passes\":[",
            esc(&s.adapter),
            esc(&s.backend),
            s.dispatches,
            s.workgroups,
            s.submits,
            s.spontaneous_futures,
            s.wait_any_only_futures,
            s.wait_any_calls,
            s.emscripten_sleep_calls,
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
