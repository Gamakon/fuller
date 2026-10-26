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
/// never fire either.
fn output_with_timeout(cmd: &mut Command, limit: Duration) -> Result<Output, String> {
    use std::io::Read;

    let program = cmd.get_program().to_string_lossy().to_string();
    let mut child = cmd
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

/// Compile a Futhark program with the real compiler, run it on `input`, and
/// return what it produced.
pub fn compile_and_run(source: &str, scalar_root: bool, input: &[f32], dir: &Path) -> Result<Vec<f32>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let fut = dir.join("prog.fut");
    std::fs::write(&fut, source).map_err(|e| format!("write: {e}"))?;
    let out = output_with_timeout(
        Command::new("futhark")
            .args(["c", "--library", fut.to_str().unwrap(), "-o", dir.join("prog").to_str().unwrap()]),
        COMPILE_TIMEOUT,
    )?;
    if !out.status.success() {
        return Err(format!(
            "futhark refused the program:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
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

