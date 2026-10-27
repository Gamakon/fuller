//! Running a Futhark program on OUR GPU stack: the branch compiler's
//! `webgpu` backend, the WebGPU C API implemented over `wgpu-rs`
//! (`futhark-wgpu-shim/`), and the same `wgpu`/Metal path the rest of fuller
//! uses — `docs/PLAN_futhark_wgpu_host.md`, step S6.
//!
//! The sibling of [`crate::futhark_exec`], which runs the same programs
//! through Futhark's sequential C backend. A gene goes
//!
//! **express → render → branch compiler → link → run → read back**:
//!
//! 1. the caller expresses and renders it (the kingdom's renderer, as for the
//!    C path), giving a Futhark program;
//! 2. the branch compiler (`futhark webgpu`, a CI artefact pinned in
//!    `futhark-wgpu-shim/README.md`) writes the host C with the WGSL embedded;
//!    it then fails at its Emscripten step, by design — the C is what is
//!    wanted, and its absence is the failure;
//! 3. `futhark-wgpu-shim/build_futhark.sh` ports the WGSL, compiles the C with
//!    plain `cc`, checks the surface, and links the shim;
//! 4. the executable reads the input in Futhark's value format, runs its
//!    kernels on the GPU through the shim, and writes the result in Futhark's
//!    binary format, which is read back here.
//!
//! The shim's report ([`ShimReport`]) comes back with the values: the
//! dispatches, and — with timestamps on — every compute pass timed by the GPU
//! itself, which is the evidence that the kernel ran on the device.
//!
//! **Where the values equal the interpreter's, and where they cannot.** The
//! shim is exact (`docs/PLAN_futhark_wgpu_host.md` §12): on exact arithmetic
//! every bit agrees with Futhark's C backend. Metal compiles with fast math,
//! though, so a multiply followed by an add is fused into one `fma` (one
//! rounding where the interpreter and C round twice), and a parallel `scan` or
//! `reduce` of reals adds in another order than the interpreter's loop. Those
//! differences are the device's and the algorithm's; this module reports the
//! values as the GPU computed them and never adjusts them.

use crate::futhark_exec::{output_with_timeout, output_with_timeout_bytes};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// How long the branch compiler may take (the same bound as the C path's:
/// [`crate::futhark_exec`] explains why a compile needs one).
const COMPILE_TIMEOUT: Duration = Duration::from_secs(120);

/// How long the link step may take. Its first run builds the shim crate
/// (`wgpu` included); later runs only check it is up to date.
const LINK_TIMEOUT: Duration = Duration::from_secs(900);

/// How long a program may run on the GPU before it is killed.
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

/// The shim crate, next to this one.
pub fn shim_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("futhark-wgpu-shim")
}

/// The branch compiler: `$FUTHARK_BRANCH`, or the pinned artefact's path
/// (`futhark-wgpu-shim/fetch_futhark_branch.sh` puts it there).
pub fn branch_compiler() -> PathBuf {
    std::env::var_os("FUTHARK_BRANCH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/branch_artifact/futhark-nightly-macos-arm64/bin/futhark"))
}

/// What the shim saw while the program ran (its `FUTHARK_WGPU_SHIM_REPORT`).
#[derive(Debug, Clone, PartialEq)]
pub struct ShimReport {
    /// The adapter the device was made on (`Apple M3 Max`).
    pub adapter: String,
    /// The `wgpu` backend (`Metal`).
    pub backend: String,
    /// `wgpuComputePassEncoderDispatchWorkgroups` calls.
    pub dispatches: u64,
    /// `wgpuQueueSubmit` calls.
    pub submits: u64,
    /// `emscripten_sleep` calls: 0 when the runtime's spin was never entered.
    pub emscripten_sleep_calls: u64,
    /// The compute passes the GPU timed: (begin tick, end tick, nanoseconds).
    /// Empty unless timestamps were asked for.
    pub passes: Vec<(u64, u64, f64)>,
}

impl ShimReport {
    fn parse(text: &str) -> Result<ShimReport, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("the shim's report is not JSON: {e}"))?;
        let num = |k: &str| v[k].as_u64().ok_or_else(|| format!("the shim's report has no {k}"));
        let txt = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| format!("the shim's report has no {k}"));
        let passes = v["passes"]
            .as_array()
            .ok_or("the shim's report has no passes")?
            .iter()
            .map(|p| {
                Ok((
                    p["begin_tick"].as_u64().ok_or("a pass with no begin_tick")?,
                    p["end_tick"].as_u64().ok_or("a pass with no end_tick")?,
                    p["gpu_ns"].as_f64().ok_or("a pass with no gpu_ns")?,
                ))
            })
            .collect::<Result<Vec<_>, &str>>()?;
        Ok(ShimReport {
            adapter: txt("adapter")?,
            backend: txt("backend")?,
            dispatches: num("dispatches")?,
            submits: num("submits")?,
            emscripten_sleep_calls: num("emscripten_sleep_calls")?,
            passes,
        })
    }

    /// The passes whose timestamps the GPU really wrote (a later end than a
    /// non-zero begin).
    pub fn timed_passes(&self) -> usize {
        self.passes.iter().filter(|(b, e, _)| *b > 0 && e > b).count()
    }
}

/// A program's result on the GPU, with the shim's report.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuRun {
    /// The output values (one for a scalar root).
    pub values: Vec<f32>,
    pub report: ShimReport,
}

/// Build the shim once and copy its static library and generated header into
/// `dir` (`futhark-wgpu-shim/prebuild_shim.sh`, under the cargo lock), for
/// [`compile_program_prebuilt`]. Returns `dir`.
pub fn prebuild_shim(dir: &Path) -> Result<PathBuf, String> {
    let out = output_with_timeout(Command::new(shim_dir().join("prebuild_shim.sh")).arg(dir).env_remove("CARGO_TARGET_DIR"), LINK_TIMEOUT, None)?;
    if !out.status.success() {
        return Err(format!("prebuilding the shim failed:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(dir.to_path_buf())
}

/// Compile a Futhark program for the GPU: the branch compiler's C, then the
/// link step. Returns the executable.
pub fn compile_program(source: &str, dir: &Path) -> Result<PathBuf, String> {
    compile_with(source, dir, None)
}

/// [`compile_program`] linking against a shim [`prebuild_shim`] wrote: the link
/// step is plain `cc`, with no cargo and no lock, so an evolution's candidates
/// never wait on a build.
pub fn compile_program_prebuilt(source: &str, dir: &Path, shim: &Path) -> Result<PathBuf, String> {
    compile_with(source, dir, Some(shim))
}

fn compile_with(source: &str, dir: &Path, prebuilt: Option<&Path>) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fut = dir.join("prog.fut");
    let c = dir.join("prog.c");
    std::fs::write(&fut, source).map_err(|e| format!("write {}: {e}", fut.display()))?;
    if c.exists() {
        std::fs::remove_file(&c).map_err(|e| format!("remove the stale {}: {e}", c.display()))?;
    }
    let compiler = branch_compiler();
    let out = output_with_timeout(
        Command::new(&compiler).arg("webgpu").arg(&fut).arg("-o").arg(dir.join("prog")),
        COMPILE_TIMEOUT,
        None,
    )?;
    // The compiler writes prog.c and then fails at its emcc step: there is no
    // Emscripten here, by design. The C is the product; no C is the failure.
    if !c.exists() {
        return Err(format!(
            "{} wrote no C for the program:\n{}",
            compiler.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let exe = dir.join("prog_wgpu");
    // The shim builds in its own target directory: a `cargo test` of fuller
    // exports CARGO_TARGET_DIR, and the shim is not part of fuller's build.
    let mut cmd = Command::new(shim_dir().join("build_futhark.sh"));
    cmd.arg(&c).arg(&exe).env_remove("CARGO_TARGET_DIR");
    if let Some(p) = prebuilt {
        cmd.env("FUTHARK_SHIM_PREBUILT", p);
    }
    let link = output_with_timeout(&mut cmd, LINK_TIMEOUT, None)?;
    if !link.status.success() {
        return Err(format!(
            "the link step failed:\n{}{}",
            String::from_utf8_lossy(&link.stdout),
            String::from_utf8_lossy(&link.stderr)
        ));
    }
    Ok(exe)
}

/// The input array in Futhark's textual value format. `{:?}` prints the
/// shortest decimal that reads back as the same `f32`, so the program sees
/// exactly these values.
pub fn input_text(input: &[f32]) -> Result<String, String> {
    if input.is_empty() {
        return Ok("empty([0]f32)\n".to_string());
    }
    let mut parts = Vec::with_capacity(input.len());
    for x in input {
        if !x.is_finite() {
            return Err(format!("an input value that is not finite: {x}"));
        }
        parts.push(format!("{x:?}f32"));
    }
    Ok(format!("[{}]\n", parts.join(", ")))
}

/// One value of Futhark's binary format holding `f32`s, as written by `-b`.
pub fn read_binary_f32(raw: &[u8]) -> Result<Vec<f32>, String> {
    let mut i = 0;
    while i < raw.len() && raw[i].is_ascii_whitespace() {
        i += 1;
    }
    let head = raw.get(i..i + 7).ok_or("the program wrote too little")?;
    if head[0] != b'b' || head[1] != 2 {
        return Err(format!("the program's output is not Futhark's binary format: {:?}", &head[..2]));
    }
    let rank = head[2] as usize;
    if &head[3..7] != b" f32" {
        return Err(format!("the program's output is not f32: {:?}", String::from_utf8_lossy(&head[3..7])));
    }
    i += 7;
    let mut n = 1usize;
    for _ in 0..rank {
        let d = raw.get(i..i + 8).ok_or("the program's output ends inside its shape")?;
        n *= i64::from_le_bytes(d.try_into().map_err(|_| "a shape that is not 8 bytes")?) as usize;
        i += 8;
    }
    let data = raw.get(i..i + 4 * n).ok_or("the program's output ends inside its values")?;
    Ok(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Run a compiled program on `input` (an `[]f32`) and read its result back.
///
/// `timestamps` asks the shim to time every compute pass on the GPU; it waits
/// for each submission, so it is a diagnostic mode, not the mode a fitness
/// should be taken in.
pub fn run_program(exe: &Path, input: &[f32], timestamps: bool) -> Result<GpuRun, String> {
    run_program_times(exe, input, timestamps, 1)
}

/// [`run_program`] with the entry point run `runs` times in the one process
/// (Futhark's `-r`): the values are the last run's, and the report holds the
/// passes of every run, in order.
pub fn run_program_times(exe: &Path, input: &[f32], timestamps: bool, runs: usize) -> Result<GpuRun, String> {
    let (stdout, report) = run_program_bytes(exe, input_text(input)?.as_bytes(), timestamps, runs)?;
    Ok(GpuRun { values: read_binary_f32(&stdout)?, report })
}

/// [`run_program_times`] for a program of any signature: `stdin` is its input
/// in either of Futhark's value formats, and the result is its output in the
/// binary format, as bytes, with the shim's report.
pub fn run_program_bytes(exe: &Path, stdin: &[u8], timestamps: bool, runs: usize) -> Result<(Vec<u8>, ShimReport), String> {
    if runs == 0 {
        return Err("a program run zero times has no result".into());
    }
    let dir = exe.parent().ok_or("an executable with no directory")?;
    let report_path = dir.join("shim_report.json");
    if report_path.exists() {
        std::fs::remove_file(&report_path).map_err(|e| format!("remove the stale report: {e}"))?;
    }
    let mut cmd = Command::new(exe);
    cmd.arg("-b").env("FUTHARK_WGPU_SHIM_REPORT", &report_path);
    if runs > 1 {
        cmd.arg("-r").arg(runs.to_string());
    }
    if timestamps {
        cmd.env("FUTHARK_WGPU_SHIM_TIMESTAMPS", "1");
    }
    let out = output_with_timeout_bytes(&mut cmd, RUN_TIMEOUT, Some(stdin))?;
    if !out.status.success() {
        return Err(format!("the program failed on the GPU:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    let report = std::fs::read_to_string(&report_path).map_err(|e| format!("the shim wrote no report: {e}"))?;
    Ok((out.stdout, ShimReport::parse(&report)?))
}

/// Compile a program for the GPU, run it on `input`, and read the result back.
pub fn compile_and_run(source: &str, input: &[f32], dir: &Path, timestamps: bool) -> Result<GpuRun, String> {
    let exe = compile_program(source, dir)?;
    run_program(&exe, input, timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::futhark_interp::Value;
    use crate::futhark_kingdom::FutharkKingdom;
    use crate::typed_gep::{express, GeneLayout, Tree};

    /// A candidate links against the prebuilt shim while the cargo lock is held by someone else: the per-candidate
    /// step runs no cargo and takes no lock. (Needs the branch compiler and `cc`; no GPU.)
    #[test]
    fn a_candidate_links_while_the_cargo_lock_is_held() {
        let have = |p: &Path| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have(&branch_compiler()) || !have(Path::new("cc")) {
            eprintln!("skipped: the branch compiler or cc is not installed");
            return;
        }
        let base = std::env::temp_dir().join(format!("futhark_wgpu_prebuilt_{}", std::process::id()));
        let shim = prebuild_shim(&base.join("shim")).unwrap();
        // Hold the lock (or find it held by someone else already: then it is theirs, and left alone).
        let lock = Path::new("/private/tmp/claude-501/locks/cargo");
        std::fs::create_dir_all(lock.parent().unwrap()).unwrap();
        let ours = std::fs::create_dir(lock).is_ok();
        assert!(lock.is_dir());
        let t = std::time::Instant::now();
        let got = compile_program_prebuilt("entry main (xs: []f32) : []f32 = map (+1f32) xs\n", &base.join("prog"), &shim);
        let secs = t.elapsed().as_secs_f64();
        if ours {
            std::fs::remove_dir(lock).unwrap();
        }
        let exe = got.unwrap();
        assert!(exe.is_file(), "no executable");
        assert!(secs < 60.0, "the link took {secs:.1} s with the lock held: it waited on the lock");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_value_formats_round_trip() {
        let xs = [0.25f32, -1.5, 1e-7, 3.4028235e38, -0.0, 7.0];
        let text = input_text(&xs).unwrap();
        assert_eq!(text, "[0.25f32, -1.5f32, 1e-7f32, 3.4028235e38f32, -0.0f32, 7.0f32]\n");
        assert_eq!(input_text(&[]).unwrap(), "empty([0]f32)\n");
        assert!(input_text(&[f32::NAN]).is_err(), "a NaN input has no textual form here");

        let mut raw = b"b\x02\x01 f32".to_vec();
        raw.extend_from_slice(&2i64.to_le_bytes());
        raw.extend_from_slice(&1.5f32.to_le_bytes());
        raw.extend_from_slice(&(-0.0f32).to_le_bytes());
        let got = read_binary_f32(&raw).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].to_bits(), 1.5f32.to_bits());
        assert_eq!(got[1].to_bits(), (-0.0f32).to_bits());
        let mut scalar = b"b\x02\x00 f32".to_vec();
        scalar.extend_from_slice(&4.0f32.to_le_bytes());
        assert_eq!(read_binary_f32(&scalar).unwrap(), vec![4.0]);
        assert!(read_binary_f32(b"b\x02\x01 i32").is_err());
    }

    /// The rows whose arithmetic is exact on any IEEE device: data movement,
    /// one addition or subtraction per element, exact operations. No
    /// multiplication (a multiply then an add is fused into one `fma` by
    /// Metal's fast math), no scan or reduction of reals (the parallel order
    /// is not the loop's), no transcendental (Metal's are not libm's).
    const EXACT_ROWS: [&str; 14] = [
        "map_a",
        "map2_add_a",
        "map2_max_a",
        "map2_min_a",
        "rev_a",
        "fn.f32_f32.neg",
        "fn.f32_f32.abs",
        "fn.f32_f32.floor",
        "fn.f32_f32.round",
        "fn.f32_f32.add_c",
        "fn.f32_f32.sub_c",
        "fn.f32_f32.const",
        "?_f32",
        "arr.f32.input",
    ];

    fn uses_only(k: &FutharkKingdom, t: &Tree, allowed: &[&str]) -> bool {
        allowed.contains(&k.rows[t.symbol as usize].name.as_str()) && t.kids.iter().all(|c| uses_only(k, c, allowed))
    }

    fn uses(k: &FutharkKingdom, t: &Tree, name: &str) -> bool {
        k.rows[t.symbol as usize].name == name || t.kids.iter().any(|c| uses(k, c, name))
    }

    /// S6's gate: a GENE goes express → render → the branch compiler → link →
    /// run on the GPU → read back, and its values equal the interpreter's —
    /// bit for bit — and Futhark's C backend's.
    ///
    /// The genes are drawn at random from the 1-D kingdom with a fixed seed
    /// and kept when they use only [`EXACT_ROWS`] and at least one map (so a
    /// kernel must run): on those rows every IEEE device computes the same
    /// bits, so "equal" means equal, with no tolerance. The GPU's own
    /// timestamps show each program's kernels ran on the device.
    ///
    /// Needs the branch compiler (`futhark-wgpu-shim/fetch_futhark_branch.sh`),
    /// `futhark` 0.28.1 for the C path, `cc`, and the GPU.
    #[test]
    fn a_gene_runs_on_the_gpu_and_equals_the_interpreter() {
        let have = |p: &Path| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have(&branch_compiler()) || !have(Path::new("futhark")) || !have(Path::new("cc")) {
            eprintln!("skipped: the branch compiler, futhark or cc is not installed");
            return;
        }
        let _gpu = crate::gpu_test_lock::acquire();
        let k = FutharkKingdom::load().unwrap();
        let contract = k.relations().check().unwrap();
        let const_len = k.const_len();
        let root = k.ty("arr.f32").unwrap();
        let layout = GeneLayout::from_program_size(&contract, root, 4).unwrap();
        let input: Vec<f32> = (0..1000).map(|i| ((i as f32) - 500.0) * 0.375).collect();
        let base = std::env::temp_dir().join("futhark_wgpu_host_test").join(format!("t{}", std::process::id()));

        let mut rng: u64 = 0x5EED_0F57_6A70_0001;
        let mut checked = 0;
        let mut draws = 0;
        while checked < 4 {
            draws += 1;
            assert!(draws < 200_000, "the fixed seed found only {checked} exact genes");
            let mut gene = vec![0u32; layout.width];
            for (pos, cell) in gene.iter_mut().enumerate() {
                let alphabet = layout.alphabet(&contract, pos, &const_len);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *cell = alphabet[((rng >> 33) as usize) % alphabet.len()];
            }
            let tree = express(&contract, &layout, &gene, &const_len).expect("a random gene is legal").tree();
            if !uses_only(&k, &tree, &EXACT_ROWS) || !(uses(&k, &tree, "map_a") || uses(&k, &tree, "map2_add_a")) {
                continue;
            }
            let source = format!("entry main (xs: []f32) : []f32 = {}\n", k.render(&tree).unwrap());
            let want = match k.eval(&tree, &input).unwrap() {
                Value::Arr(a) => a,
                other => panic!("an arr.f32 root evaluated to {other:?}"),
            };
            let dir = base.join(format!("gene{checked}"));
            let gpu = compile_and_run(&source, &input, &dir, true).unwrap_or_else(|e| panic!("{e}\n{source}"));
            let c = crate::futhark_exec::compile_and_run(&source, false, &input, &dir.join("c")).unwrap();

            assert_eq!(gpu.report.backend, "Metal", "the program did not run on Metal");
            assert!(gpu.report.dispatches >= 1, "no kernel was dispatched\n{source}");
            assert_eq!(
                gpu.report.timed_passes(),
                gpu.report.passes.len(),
                "a pass the GPU did not time: {:?}\n{source}",
                gpu.report.passes
            );
            assert_eq!(gpu.report.emscripten_sleep_calls, 0, "the runtime's spin was entered");
            assert_eq!(gpu.values.len(), want.len(), "the GPU and the interpreter disagree on the length\n{source}");
            for (i, ((g, w), cv)) in gpu.values.iter().zip(&want).zip(&c).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "value {i}: the GPU says {g}, the interpreter says {w}\n{source}");
                assert_eq!(g.to_bits(), cv.to_bits(), "value {i}: the GPU says {g}, Futhark's C backend says {cv}\n{source}");
            }
            eprintln!(
                "gene {checked}: {} values bit-identical; {} dispatches, {} timed on the GPU ({:.1} us)\n  {}",
                want.len(),
                gpu.report.dispatches,
                gpu.report.timed_passes(),
                gpu.report.passes.iter().map(|p| p.2).sum::<f64>() / 1e3,
                source.trim()
            );
            checked += 1;
        }
    }
}
