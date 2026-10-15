//! ONE GPU USER AT A TIME IN THE TEST BINARY. `cargo test` runs tests on
//! parallel threads; several tests driving the one GPU at once each wait
//! longer than the harness's 5 s runaway guard (`wgsl::harness::wait_gpu`),
//! and nine GPU tests failed so on 2026-10-15 while all 721 passed one at a
//! time. Every device a test makes holds a token for as long as it lives,
//! so GPU tests take turns and CPU tests still run in parallel. The token is
//! re-entrant on its thread (a test that holds two devices at once does not
//! wait for itself) and may be dropped on another thread. Test builds only.

use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

static STATE: Mutex<(Option<ThreadId>, usize)> = Mutex::new((None, 0));
static FREE: Condvar = Condvar::new();

/// A held turn on the GPU; the turn ends when every token of its thread is dropped.
#[derive(Debug)]
pub struct Token(());

/// Wait for the GPU's turn (or join this thread's).
pub fn acquire() -> Token {
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
    fn turns_never_overlap_and_nest_on_their_thread() {
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
}
