//! ONE GPU USER AT A TIME IN THE TEST BINARY. `cargo test` runs tests on
//! parallel threads; several tests driving the one GPU at once each wait
//! longer than the harness's 5 s runaway guard (`wgsl::harness::wait_gpu`),
//! and nine GPU tests failed so on 2026-10-15 while all 721 passed one at a
//! time. Every device a test makes holds a token for as long as it lives,
//! so GPU tests take turns and CPU tests still run in parallel. The token is
//! re-entrant on its thread (a test that holds two devices at once does not
//! wait for itself) and may be dropped on another thread. Test builds only.
//!
//! EVERY TEST THAT MAKES A DEVICE IS NAMED FOR IT. The documented CPU run
//! (`--skip device --skip gpu --skip metal --skip on_the_device`) skips a
//! test by a substring of its path, so a GPU test without one of those words
//! ran on the GPU during a CPU run and contaminated a fit's timing
//! (2026-10-17, run 1). Every device site takes its turn here before it
//! makes a wgpu instance, so `acquire` checks the test thread's name (the
//! libtest thread is named for the test's path) and panics, before any
//! device exists, naming a test whose path carries no marker.

use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

static STATE: Mutex<(Option<ThreadId>, usize)> = Mutex::new((None, 0));
static FREE: Condvar = Condvar::new();

/// A held turn on the GPU; the turn ends when every token of its thread is dropped.
#[derive(Debug)]
pub struct Token(());

/// The substrings the CPU run skips (`--skip <marker>`): a test whose path
/// holds one is a GPU test.
pub const MARKERS: [&str; 4] = ["device", "gpu", "metal", "on_the_device"];

/// Whether a test path carries a GPU marker.
pub fn marked(test_path: &str) -> bool {
    MARKERS.iter().any(|m| test_path.contains(m))
}

/// THE MARKER RULE for the current thread: a libtest thread (named for its
/// test's path) must carry a marker to make a device; an unnamed thread (a
/// test's own helper) or the main thread cannot be judged and passes.
pub fn check_marked() -> Result<(), String> {
    match std::thread::current().name() {
        Some(n) if n != "main" && !marked(n) => Err(format!(
            "our software has a bug: test {n} makes a GPU device but its path carries none of {MARKERS:?}, so the CPU filter (--skip device --skip gpu --skip metal --skip on_the_device) runs it on the GPU; name it for the GPU"
        )),
        _ => Ok(()),
    }
}

/// Wait for the GPU's turn (or join this thread's). Panics, before any device
/// is made, when the test making one is not named for the GPU (`check_marked`).
pub fn acquire() -> Token {
    if let Err(e) = check_marked() {
        panic!("{e}");
    }
    let me = std::thread::current().id();
    // A test that panicked while holding the state leaves it consistent:
    // the counts are only changed under the lock, in one step.
    let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    while s.1 > 0 && s.0 != Some(me) {
        s = FREE.wait(s).unwrap_or_else(|e| e.into_inner());
    }
    s.0 = Some(me);
    s.1 += 1;
    Token(())
}

impl Drop for Token {
    fn drop(&mut self) {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        s.1 -= 1;
        if s.1 == 0 {
            s.0 = None;
            FREE.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Eight threads, each holding a turn twice over (re-entrant), never
    /// overlap; a token moved to another thread still ends its turn.
    #[test]
    fn gpu_turns_never_overlap_and_nest_on_their_thread() {
        let inside = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (inside, most) = (inside.clone(), most.clone());
                std::thread::spawn(move || {
                    let a = acquire();
                    let b = acquire();
                    let n = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(n, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    inside.fetch_sub(1, Ordering::SeqCst);
                    drop(b);
                    drop(a);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("a thread");
        }
        assert_eq!(most.load(Ordering::SeqCst), 1);
        let t = acquire();
        std::thread::spawn(move || drop(t)).join().expect("dropped elsewhere");
        drop(acquire());
    }

    /// The marker rule: a test thread whose path holds no marker cannot take
    /// a GPU turn (the panic names it, before any device); a marked one, an
    /// unnamed helper thread and the main thread can.
    #[test]
    fn a_test_not_named_for_the_gpu_cannot_take_a_turn() {
        let run = |name: Option<&str>| {
            let b = match name {
                Some(n) => std::thread::Builder::new().name(n.to_string()),
                None => std::thread::Builder::new(),
            };
            b.spawn(|| std::panic::catch_unwind(|| drop(acquire())).map_err(|e| e.downcast_ref::<String>().cloned().unwrap_or_default())).expect("spawn").join().expect("join")
        };
        let e = run(Some("wgsl::gram::tests::a_deliberately_slow_kernel_is_stopped")).unwrap_err();
        assert!(e.contains("our software has a bug: test wgsl::gram::tests::a_deliberately_slow_kernel_is_stopped makes a GPU device"), "{e}");
        for ok in [Some("wgsl::gram::tests::a_slow_kernel_on_the_device"), Some("gpu_eval::tests::x"), Some("lint::tests::metal_x"), None, Some("main")] {
            assert!(run(ok).is_ok(), "{ok:?}");
        }
        assert!(marked("wgsl::harness::tests::the_v24_reference_runs_on_the_device") && !marked("wgsl::harness::tests::the_v24_reference_runs_end_to_end_on_the_cpu_executor"));
    }
}
