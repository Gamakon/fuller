//! Running a Futhark program the kingdom emitted: the real Futhark compiler,
//! the real C compiler, and the real program.
//!
//! This is the check that the kingdom means what it says. The renderer turns
//! an expressed tree into Futhark source; [`crate::futhark_interp`] turns the
//! same tree into a value; this module compiles the source and runs it, so
//! the two can be compared element for element.

use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// How long a Futhark compile may take before it is killed.
///
/// Futhark's optimiser does not terminate on every legal program. Measured on
/// Futhark 0.28.1: a program that reduces over a `map2` whose second operand
/// is a *reversed* scan-derived array, and whose first operand contains a
/// reduce over a scan result, compiles in 0.1 s without those features and
/// **never finishes** with them — killed at 300 s with no output. The program
/// is legal and well-defined; the *compiler* does not converge. So the
/// compile step needs a wall-clock bound, and a program that exceeds it is a
/// failed compile, not a failed gene.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(120);

/// Run a command to completion, killing it if it outlives `limit`.
///
/// `Command::output` has no timeout, and a compiler that does not terminate
/// would otherwise hang the caller.
///
/// The pipes are drained on their own threads: a child that fills the pipe
/// buffer blocks, and the wait would then never end — the timeout would
/// never fire either. `stdin_data` is written to the child's stdin **on its
/// own thread, after the drains have started**: a child that blocks writing
/// its output while we block writing its input would deadlock, and the
/// timeout would never fire. If the write fails, the child is killed and
/// waited for, so no orphan is left behind, and the error carries the
/// child's own output — a program that died early is more informative in its
/// stderr than in a broken pipe.
pub fn output_with_timeout(cmd: &mut Command, limit: Duration, stdin_data: Option<&str>) -> Result<Output, String> {
    output_with_timeout_bytes(cmd, limit, stdin_data.map(str::as_bytes))
}

/// [`output_with_timeout`] with the input as bytes (Futhark's binary value
/// format is not text).
pub fn output_with_timeout_bytes(cmd: &mut Command, limit: Duration, stdin_data: Option<&[u8]>) -> Result<Output, String> {
    use std::io::{Read, Write};

    let program = cmd.get_program().to_string_lossy().to_string();
    let mut child = cmd
        .stdin(if stdin_data.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run {program}: {e}"))?;

    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out_reader = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let err_reader = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));

    // The input goes in on its own thread, now that the drains are running.
    let writer = stdin_data.map(|data| {
        let program = program.clone();
        let data = data.to_vec();
        let mut si = child.stdin.take();
        std::thread::spawn(move || {
            let Some(mut si) = si.take() else {
                return Ok::<(), String>(());
            };
            si.write_all(&data)
                .map_err(|e| format!("write to {program}: {e}"))?;
            drop(si);
            Ok(())
        })
    });

    let deadline = Instant::now() + limit;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{program}: no answer within {}s", limit.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(format!("wait: {e}")),
        }
    };

    let stdout = out_reader.join().unwrap_or_default();
    let stderr = err_reader.join().unwrap_or_default();
    if let Some(writer) = writer {
        match writer.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(format!(
                    "{e}\nthe program's own output was:\n{}",
                    String::from_utf8_lossy(&stderr)
                ))
            }
            Err(_) => return Err(format!("the thread writing to {program} panicked")),
        }
    }
    Ok(Output { status, stdout, stderr })
}

/// The C driver: it reads `n` and `n` floats, runs the compiled Futhark
/// program, and writes `m` and `m` floats. It is the same for every program
/// except for the entry point's result type.
const DRIVER_ARRAY: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "prog.h"
int main(void) {
  int64_t n; float x[65536];
  if (fread(&n, sizeof n, 1, stdin) != 1) return 2;
  if (n < 0 || n > 65536) return 5;
  if (fread(x, sizeof(float), (size_t)n, stdin) != (size_t)n) return 3;
  struct futhark_context_config *cfg = futhark_context_config_new();
  struct futhark_context *ctx = futhark_context_new(cfg);
  struct futhark_f32_1d *in = futhark_new_f32_1d(ctx, x, n);
  struct futhark_f32_1d *out = NULL;
  int rc = futhark_entry_main(ctx, &out, in);
  if (rc != 0) { fprintf(stderr, "the entry point failed: %d\n", rc); return 4; }
  float y[65536];
  futhark_values_f32_1d(ctx, out, y);
  int64_t m = futhark_shape_f32_1d(ctx, out)[0];
  fwrite(&m, sizeof m, 1, stdout);
  fwrite(y, sizeof(float), (size_t)m, stdout);
  return 0;
}
"#;

const DRIVER_SCALAR: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "prog.h"
int main(void) {
  int64_t n; float x[65536];
  if (fread(&n, sizeof n, 1, stdin) != 1) return 2;
  if (n < 0 || n > 65536) return 5;
  if (fread(x, sizeof(float), (size_t)n, stdin) != (size_t)n) return 3;
  struct futhark_context_config *cfg = futhark_context_config_new();
  struct futhark_context *ctx = futhark_context_new(cfg);
  struct futhark_f32_1d *in = futhark_new_f32_1d(ctx, x, n);
  float out = 0.0f;
  int rc = futhark_entry_main(ctx, &out, in);
  if (rc != 0) { fprintf(stderr, "the entry point failed: %d\n", rc); return 4; }
  int64_t m = 1;
  fwrite(&m, sizeof m, 1, stdout);
  fwrite(&out, sizeof(float), 1, stdout);
  return 0;
}
"#;

/// The compiler's verdict on a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The program compiled.
    Compiled,
    /// The compiler refused it, with this message.
    Refused(String),
    /// The compiler did not finish within the limit, and the bounded-fusion
    /// path did not finish either.
    TimedOut,
    /// The program did **not** compile the normal way, but it **did** compile
    /// with the fusion loop bounded — see [`compile_with`].
    ///
    /// This is a **compiler workaround**, not an e-graph rescue: the engine
    /// did not find an equivalent program the compiler could handle. It
    /// recompiled *the same program* with the optimisation capped. Fusion is
    /// an optimisation, so stopping its loop early leaves fewer fusions, not
    /// wrong ones — but the binary may be slower, so a caller that scores
    /// programs should know which path produced it.
    ///
    /// The name says "bounded fusion" rather than "rescued" on purpose: the
    /// e-graph's contribution to this outcome is zero, and a verdict that
    /// implied otherwise would misattribute the result.
    BoundedFusion,
}

/// The compiler's backend.
///
/// The two that matter here: the **C** backend is the sequential one, and
/// the **OpenCL** backend runs the program **on the GPU** — on this
/// machine, Apple's OpenCL over Metal, which Futhark's GPU pipeline drives
/// all the way to the block+register tiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The sequential C backend: `futhark c`.
    C,
    /// The OpenCL backend: `futhark opencl`, on the GPU.
    OpenCl,
}

impl Backend {
    /// The `futhark` subcommand that compiles for this backend.
    fn command(self) -> &'static str {
        match self {
            Backend::C => "c",
            Backend::OpenCl => "opencl",
        }
    }
}

/// The fusion bound the bounded-fusion path uses.
///
/// Futhark's own pipeline runs the SOAC fusion pass to a fixpoint with **no
/// bound** (`fuseSOACs Nothing` in `src/Futhark/Passes.hs`), and on some
/// legal programs the fusion graph keeps changing, so the pass never
/// terminates. The pass takes a bound, and `futhark dev --fuse=N` exposes it.
///
/// **This number is a measured work budget, not a convergence bound.** The
/// bound does not say how many iterations fusion *needs*; it says how many
/// iterations of fusion work the compiler will do before the pass is cut
/// off. Measured on the pathological program of
/// [`docs/RESCUE_futhark.md`](../docs/RESCUE_futhark.md) §7, with the pass
/// list below:
///
/// | bound | result |
/// |---|---|
/// | 13 or less | compiles, and computes the right value |
/// | 14 or more | the compiler does not finish (killed at 45 s) |
///
/// So 10 sits below the cliff with margin, and the value is right at every
/// bound that compiles — the bound changes how *much* fusion happens, never
/// what the program computes. A bound that is too small costs optimisation;
/// a bound that is too large costs the compiler's convergence.
const FUSION_BOUND: u32 = 10;

/// The passes of Futhark's sequential pipeline, in the order its own pipeline
/// runs them, so that the bounded-fusion path can run the same pipeline with
/// the fusion loop capped.
///
/// This mirrors `standardPipeline`, `seqPipeline` and `seqmemPipeline` in
/// Futhark 0.28.1's `src/Futhark/Passes.hs`. It is spelled out here because
/// `futhark dev` runs the passes it is given, in the order they are given,
/// and the whole-pipeline options (`--seq-mem`) would run the *unbounded*
/// fusion again. A Futhark release that reorders its pipeline would make this
/// list stale — which is why the bounded-fusion path is a *fallback*, used only
/// for programs the normal path cannot compile at all.
const BOUNDED_FUSION_PASSES: &[&str] = &[
    // ---- standardPipeline (SOACS -> SOACS) ---------------------------------
    "--simplify",
    "--inline-conservatively",
    "--simplify",
    "--inline-aggressively",
    "--simplify",
    "--cse",
    "--simplify",
    // The one pass that takes an argument. It must appear *here*, in the
    // pipeline's order — and it must carry its bound, because `--fuse`
    // without one means "iterate to a fixpoint", which is the thing that
    // does not terminate.
    "--fuse={FUSION_BOUND}",
    "--cse",
    "--simplify",
    "--remove-dead-functions",
    // ---- seqPipeline (SOACS -> Seq) ----------------------------------------
    "--first-order-transform",
    "--simplify",
    // ---- seqmemPipeline (Seq -> SeqMem) ------------------------------------
    "--explicit-allocations",
    "--cse",
    "--simplify",
    "--simplify",
    "--lift-allocations",
    "--simplify",
    "--short-circuit",
    "--simplify",
    "--cse",
    "--simplify",
    "--lower-allocations",
    "--simplify",
];

/// Compile a Futhark program with the real compiler, and say how long it took.
///
/// The verdict is separate from the error string because a compiler that does
/// not converge is an *outcome*, not a defect: the caller (the e-graph's
/// experiment) needs to tell "refused" from "did not finish" from
/// "compiled", and to know how long each took.
pub fn compile_program(source: &str, dir: &Path, limit: Duration) -> (Verdict, Duration) {
    compile_with(source, dir, limit, true, Backend::C)
}

/// Compile a Futhark program that uses **Futhark's textual input and output
/// format** instead of the C library API.
///
/// A 2-D array is `[[1.0,2.0],[3.0,4.0]]` in that format, so a program over
/// matrices needs no C driver at all: the inputs go in on stdin and the
/// result comes out on stdout. The trade is that the format is Futhark's, not
/// ours — the 1-D kingdom's driver stays the faster path for flat arrays.
pub fn compile_text_program(source: &str, dir: &Path, limit: Duration) -> (Verdict, Duration) {
    compile_with(source, dir, limit, false, Backend::C)
}

/// Compile a program that uses **Futhark's textual input and output
/// format** for a chosen backend — and the OpenCL backend puts the program
/// **on the GPU**.
pub fn compile_text_program_on(
    backend: Backend,
    source: &str,
    dir: &Path,
    limit: Duration,
) -> (Verdict, Duration) {
    compile_with(source, dir, limit, false, backend)
}

/// Compile a program, and **retry it with the fusion loop bounded if the
/// compiler does not converge**.
///
/// Futhark 0.28.1's SOAC fusion pass iterates to a fixpoint with no bound, and
/// on some legal programs it never gets there: the fusion graph keeps
/// changing, so the compiler runs forever. Measured on the program the
/// experiment found (`docs/EGRAPH_futhark.md` §3.1), `futhark c` never
/// finishes — 180 s, and the nightly 0.29.0-prerelease neither — while the
/// *same* program compiled with the fusion loop bounded takes **0.12 s**
/// (measured, three runs) and agrees with the independent interpreter, value
/// for value.
///
/// So the compiler is not wrong about the program: it is the fusion pass that
/// will not settle. The pass takes a bound, `futhark dev --fuse=N` exposes
/// it, and the bounded program computes the same thing — fusion is an
/// optimisation, so stopping the loop early leaves *fewer* fusions, not wrong
/// ones.
///
/// The verdict says which path produced the binary: [`Verdict::Compiled`] for
/// the normal path, [`Verdict::BoundedFusion`] for the bounded one. The
/// duration is the successful compile's own time; the failed attempt's cost is
/// the limit, and the caller can add it.
fn compile_with(
    source: &str,
    dir: &Path,
    limit: Duration,
    library: bool,
    backend: Backend,
) -> (Verdict, Duration) {
    let (verdict, elapsed) = compile_normal(source, dir, limit, library, backend);
    match verdict {
        Verdict::Compiled | Verdict::Refused(_) => (verdict, elapsed),
        _ if backend != Backend::C => {
            // The bounded-fusion path is C-only, and that is a property of
            // the compiler rather than of the program: the GPU pipeline
            // cannot be spelled out by hand, because `futhark dev` does not
            // expose `mergeGPUBodies` at all. So a GPU program that does
            // not converge is a failed compile, and the caller is told so.
            (verdict, elapsed)
        }
        _ => {
            let (bounded, bounded_elapsed) =
                compile_bounded_fusion(source, dir, limit, library);
            match bounded {
                Verdict::Compiled => (Verdict::BoundedFusion, bounded_elapsed),
                other => (other, elapsed + bounded_elapsed),
            }
        }
    }
}

/// The normal compile path: `futhark c`, with the compiler's own pipeline.
fn compile_normal(
    source: &str,
    dir: &Path,
    limit: Duration,
    library: bool,
    backend: Backend,
) -> (Verdict, Duration) {
    let started = Instant::now();
    let outcome = (|| -> Result<Verdict, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        let fut = dir.join("prog.fut");
        std::fs::write(&fut, source).map_err(|e| format!("write: {e}"))?;
        let binary = dir.join(if library { "prog" } else { "prog_text" });
        let mut cmd = Command::new("futhark");
        cmd.arg(backend.command());
        if library {
            cmd.arg("--library");
        }
        let out = output_with_timeout(
            cmd.args([fut.to_str().unwrap(), "-o", binary.to_str().unwrap()]),
            limit,
            None,
        );
        Ok(verdict_of(out))
    })()
    .unwrap_or_else(Verdict::Refused);
    (outcome, started.elapsed())
}

/// The bounded-fusion path: the same pipeline, with the fusion loop bounded.
///
/// `futhark dev` writes its output into the **current directory**, named after
/// the source file, so the source is written under the name the caller
/// expects the artefact to have: `prog.fut` for the library path (which
/// wants `prog.c` and `prog.h`) and `prog_text.fut` for the textual path
/// (which wants the executable `prog_text`).
fn compile_bounded_fusion(source: &str, dir: &Path, limit: Duration, library: bool) -> (Verdict, Duration) {
    let started = Instant::now();
    let outcome = (|| -> Result<Verdict, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        let stem = if library { "prog" } else { "prog_text" };
        let fut = dir.join(format!("{stem}.fut"));
        std::fs::write(&fut, source).map_err(|e| format!("write: {e}"))?;
        let mut cmd = Command::new("futhark");
        cmd.arg("dev");
        for pass in BOUNDED_FUSION_PASSES {
            cmd.arg(pass.replace("{FUSION_BOUND}", &FUSION_BOUND.to_string()));
        }
        cmd.arg("--backend=c");
        if library {
            cmd.arg("--library");
        } else {
            cmd.arg("--executable");
        }
        cmd.arg(format!("{stem}.fut"));
        cmd.current_dir(dir);
        let out = output_with_timeout(&mut cmd, limit, None);
        Ok(verdict_of(out))
    })()
    .unwrap_or_else(Verdict::Refused);
    (outcome, started.elapsed())
}

/// The compiler's outcome, as a verdict.
fn verdict_of(out: Result<Output, String>) -> Verdict {
    match out {
        Err(e) if e.contains("no answer within") => Verdict::TimedOut,
        Err(e) => Verdict::Refused(e),
        Ok(out) if !out.status.success() => Verdict::Refused(format!(
            "futhark refused the program:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )),
        Ok(_) => Verdict::Compiled,
    }
}

/// Run a program compiled by [`compile_text_program`] on a textual input, and
/// return its textual output.
pub fn run_text_program(source: &str, input: &str, dir: &Path, limit: Duration) -> Result<String, String> {
    run_text_program_on(Backend::C, source, input, dir, limit)
}

/// Run a program for a chosen backend — and the OpenCL backend runs it
/// **on the GPU**.
pub fn run_text_program_on(
    backend: Backend,
    source: &str,
    input: &str,
    dir: &Path,
    limit: Duration,
) -> Result<String, String> {
    let verdict = compile_text_program_on(backend, source, dir, limit).0;
    match verdict {
        Verdict::Compiled | Verdict::BoundedFusion => {}
        Verdict::Refused(e) => return Err(e),
        Verdict::TimedOut => return Err(format!("futhark: no answer within {}s", limit.as_secs())),
    }
    let binary = dir.join("prog_text");
    let out = output_with_timeout(
        Command::new(binary.to_str().unwrap()).stdin(Stdio::piped()),
        limit,
        Some(input),
    )
    .map_err(|e| format!("the program did not run: {e}"))?;
    if !out.status.success() {
        return Err(format!("the program failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Compile a Futhark program with the real compiler, run it on `input`, and
/// return what it produced.
pub fn compile_and_run(source: &str, scalar_root: bool, input: &[f32], dir: &Path) -> Result<Vec<f32>, String> {
    let verdict = compile_program(source, dir, COMPILE_TIMEOUT).0;
    match verdict {
        Verdict::Compiled | Verdict::BoundedFusion => {}
        Verdict::Refused(e) => return Err(e),
        Verdict::TimedOut => {
            return Err(format!("futhark: no answer within {}s", COMPILE_TIMEOUT.as_secs()))
        }
    }
    run_compiled(scalar_root, input, dir)
}

/// Run a program that has already been compiled into `dir`, and return what it
/// produced.
///
/// The C driver and the executable are built from the `prog.c` and `prog.h`
/// that [`compile_program`] left there, so a caller that has already compiled
/// does not compile twice.
pub fn run_compiled(scalar_root: bool, input: &[f32], dir: &Path) -> Result<Vec<f32>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let driver = dir.join("driver.c");
    std::fs::write(&driver, if scalar_root { DRIVER_SCALAR } else { DRIVER_ARRAY })
        .map_err(|e| format!("write: {e}"))?;
    let exe = dir.join("prog_exec");
    let build = Command::new("cc")
        .args(["-O2", "-o", exe.to_str().unwrap(), driver.to_str().unwrap(), dir.join("prog.c").to_str().unwrap(), "-lm"])
        .output()
        .map_err(|e| format!("could not run cc: {e}"))?;
    if !build.status.success() {
        return Err(format!("cc failed:\n{}", String::from_utf8_lossy(&build.stderr)));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(input.len() as i64).to_le_bytes());
    for x in input {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    use std::io::Write;
    let mut child = Command::new(exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run the program: {e}"))?;
    child.stdin.as_mut().unwrap().write_all(&bytes).map_err(|e| format!("write: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        return Err(format!("the program exited with {:?}", out.status));
    }
    let raw = out.stdout;
    if raw.len() < 8 {
        return Err("the program wrote too little".to_string());
    }
    let m = i64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    if raw.len() < 8 + 4 * m {
        return Err("the program wrote too little".to_string());
    }
    Ok((0..m)
        .map(|i| f32::from_le_bytes(raw[8 + 4 * i..12 + 4 * i].try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The program Futhark 0.28.1's optimiser never finishes.
    ///
    /// It is the program the experiment found
    /// (`docs/EGRAPH_futhark.md` §3.1), written out here so that this test
    /// does not depend on the kingdom's gene machinery. The trigger is the
    /// conjunction Futhark's SOAC fusion pass cannot settle: an inner
    /// reduction over an accumulator, a reversed accumulator-derived operand,
    /// and an outer reduction.
    const PATHOLOGICAL: &str = "entry main (xs: []f32) : f32 =\n  reduce (*) 1.0\n    (map2 f32.max\n      (map2 f32.min\n        (reverse (xs))\n        (map (\\x -> x - (reduce (+) 0.0 (map (\\x -> x) (scan (+) 0.0 xs)))) (xs)))\n      (reverse (map (\\x -> f32.round x)\n        (map2 (*) (map (\\x -> x) (scan (+) 0.0 xs)) (scan (*) 1.0 (xs))))))\n";

    /// The compiler cannot compile this program; the bounded-fusion path can.
    ///
    /// Futhark's SOAC fusion pass iterates to a fixpoint with no bound, and on
    /// this program it never gets there: `futhark c` runs for ever (measured:
    /// killed at 180 s, and the 0.29.0-prerelease nightly neither). The pass
    /// takes a bound, and the bounded-fusion path runs the same pipeline with
    /// the fusion loop capped — the program then compiles in **0.12 s**
    /// (measured, three runs).
    ///
    /// And it computes the right thing: on `[1, 2, 3]` the program is
    /// `reduce (*) 1.0 [36, 6, 1] = 216`, which is what Futhark's own
    /// interpreter gives for the same source.
    ///
    /// The scope is **one** program and **one** input: the point is that the
    /// workaround exists and computes the right value, not that the compiler's
    /// hang is characterised for every program.
    #[test]
    fn a_program_the_compiler_cannot_compile_is_bounded_fusion() {
        let have = |p: &str| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("futhark") || !have("cc") {
            eprintln!("skipped: futhark or cc is not installed");
            return;
        }
        let dir = std::env::temp_dir().join("futhark_rescue_test").join(format!("t{}", std::process::id()));

        // The normal path does not finish.
        let verdict =
            compile_normal(PATHOLOGICAL, &dir, Duration::from_secs(10), true, Backend::C).0;
        assert_eq!(
            verdict,
            Verdict::TimedOut,
            "futhark now compiles this program, so the bounded-fusion path is no longer needed: \
             the compiler's fusion loop has been fixed upstream"
        );

        // The bounded-fusion path compiles it.
        let (verdict, elapsed) = compile_bounded_fusion(PATHOLOGICAL, &dir, Duration::from_secs(60), true);
        assert_eq!(verdict, Verdict::Compiled);
        assert!(elapsed < Duration::from_secs(30), "the bounded-fusion compile took {elapsed:?}");

        // And the program computes the right value.
        let got = run_compiled(true, &[1.0, 2.0, 3.0], &dir).expect("the compiled program runs");
        assert_eq!(got, vec![216.0], "the compiled program computed the wrong value");
    }
}
