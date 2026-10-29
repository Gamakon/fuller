//! The batch host's tests. On the c backend (host only, `futhark` 0.28.1 and `cc`): a batch of random programs gives
//! every program's outputs exactly as its own per-candidate quality program does; a program the compiler refuses is
//! found by halving, named with the compiler's message, and the others are scored from their batches. On the GPU
//! (the branch compiler, the GPU lock): the same equality through the shim, and timed pairs inside one process.

use super::*;
use crate::futhark_kingdom::GeneRng;
use crate::futhark_sig::{compile_c, load, load_cases, problem, program_cases, random_gene, signature, Case};
use crate::typed_gep::{express, GeneLayout};

fn have(p: &str) -> bool {
    Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("futhark_batch_{name}_{}", std::process::id()))
}

fn stdin_of(cases: &[Case]) -> Vec<u8> {
    cases.iter().flat_map(|c| c.stdin.clone()).collect()
}

/// `n` random trees of a problem's kingdom (program size 8, the study's), distinct by text, from a fixed seed.
fn trees(k: &FutharkKingdom, n: usize, seed: u64) -> Vec<Tree> {
    let contract = k.relations().check().unwrap();
    let layout = GeneLayout::from_program_size(&contract, k.roots()[0], 8).unwrap();
    let const_len = k.const_len();
    let mut rng = GeneRng::new(seed);
    let mut out: Vec<Tree> = Vec::new();
    while out.len() < n {
        let g = random_gene(&contract, &layout, &const_len, &mut rng).unwrap();
        let t = express(&contract, &layout, &g, &const_len).unwrap().tree();
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// The per-candidate path's quality outputs of a tree on the c backend: its own quality program, its own process.
fn per_candidate(k: &FutharkKingdom, sig: &Signature, t: &Tree, cases: &[Case], dir: &Path) -> Vec<Array> {
    let exe = compile_c(&program_cases(k, sig, t, cases.len()).unwrap(), dir).unwrap();
    let out = output_with_timeout_bytes(Command::new(&exe).arg("-b"), Duration::from_secs(120), Some(&stdin_of(cases))).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    read_binary(&out.stdout).unwrap()
}

/// A batch of 6 random programs of each micro problem (a size input and two inputs in the histogram) gives every
/// program's outputs on every case exactly as its own quality program in its own process does, and its timed entry's
/// output on the timing input exactly as the plain program's. One compile, one process.
#[test]
fn a_batch_gives_each_programs_outputs_exactly_as_its_own_program() {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test");
    for name in ["reduce", "histogram", "pathfinder"] {
        let p = problem(name).unwrap();
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let cases = load_cases(&p).unwrap();
        let base = scratch(&format!("eq_{name}"));
        // The timing input: the last case's (a small one, so the c backend is quick).
        let timing = cases.last().unwrap().stdin.clone();
        let host = Host::new(&k, &sig, cases.len(), Backend::C, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&timing), yardstick: Some(p.yardstick) }).unwrap();
        let ts = trees(&k, 6, 0xBA7C_0001);
        let members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality, Step::Output] }).collect();
        let got = host.run(&members).unwrap();
        assert_eq!(got.cost.compiles, 1, "{name}: one compile");
        assert_eq!(got.cost.processes, 1, "{name}: one process");
        for (i, t) in ts.iter().enumerate() {
            let want = per_candidate(&k, &sig, t, &cases, &base.join(format!("single{i}")));
            assert_eq!(got.outcomes[i][0], Outcome::Quality(want), "{name}: program {i}'s quality outputs");
            let plain = compile_c(&crate::futhark_sig::program(&k, &sig, t).unwrap(), &base.join(format!("plain{i}"))).unwrap();
            let want_out = crate::futhark_sig::run_exe(&plain, &timing).unwrap();
            assert_eq!(got.outcomes[i][1], Outcome::Output(want_out), "{name}: program {i}'s output on the timing input");
        }
        eprintln!("{name}: 6 programs, 1 compile ({:.2} s futhark, {:.2} s cc), 1 process ({:.2} s)", got.cost.compile.futhark, got.cost.compile.cc, got.cost.run.total);
        let _ = std::fs::remove_dir_all(&base);
    }
}

/// A program the compiler refuses (here a text naming no definition: legal genes cannot be made to fail on the c
/// backend) is found by halving the batch: its steps are failures carrying the compiler's own message, and every
/// other program is scored from its half, exactly.
#[test]
fn one_program_the_compiler_refuses_is_found_by_halving_and_the_rest_are_scored() {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test");
    let p = problem("reduce").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let base = scratch("halving");
    let timing = cases.last().unwrap().stdin.clone();
    let host = Host::new(&k, &sig, cases.len(), Backend::C, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&timing), yardstick: Some(p.yardstick) }).unwrap();
    let ts = trees(&k, 5, 0xBA7C_0002);
    let bad = "entry main [n] (xs: [n]i32) : i32 = no_such_definition xs\n";
    let mut members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality] }).collect();
    members.insert(3, Member { program: Program::Text(bad), steps: vec![Step::Output] });
    let got = host.run(&members).unwrap();
    match &got.outcomes[3][0] {
        Outcome::Failed(e) => assert!(e.contains("no_such_definition"), "the failure does not carry the compiler's message: {e}"),
        o => panic!("the refused program was scored: {o:?}"),
    }
    for (j, t) in ts.iter().enumerate() {
        let i = if j < 3 { j } else { j + 1 };
        let want = per_candidate(&k, &sig, t, &cases, &base.join(format!("single{j}")));
        assert_eq!(got.outcomes[i][0], Outcome::Quality(want), "program {i} was not scored from its half");
    }
    // 6 members: the whole (fails), [0..3) (compiles), [3..6) (fails), [3] (fails), [4..6) (compiles).
    assert_eq!((got.cost.compiles, got.cost.failed_compiles), (2, 3), "the halving's compiles");
    let _ = std::fs::remove_dir_all(&base);
}

/// A program that fails while it runs (an index out of bounds, Futhark's own check) fails only its own step: the
/// driver records the entry's error and stops, and a new process runs the commands after it, so the programs before
/// and after it are scored exactly.
#[test]
fn a_program_that_fails_while_it_runs_fails_only_its_own_step() {
    assert!(have("futhark") && have("cc"), "futhark 0.28.1 and cc are needed for this test");
    let p = problem("reduce").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let base = scratch("runtime_failure");
    let timing = cases.last().unwrap().stdin.clone();
    let host = Host::new(&k, &sig, cases.len(), Backend::C, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&timing), yardstick: Some(p.yardstick) }).unwrap();
    let ts = trees(&k, 2, 0xBA7C_0005);
    let bad = "entry main [n] (xs: [n]i32) : i32 = xs[n + 5]\n";
    let members = vec![
        Member { program: Program::Tree(&ts[0]), steps: vec![Step::Quality] },
        Member { program: Program::Text(bad), steps: vec![Step::Output] },
        Member { program: Program::Tree(&ts[1]), steps: vec![Step::Quality] },
    ];
    let got = host.run(&members).unwrap();
    match &got.outcomes[1][0] {
        Outcome::Failed(e) => assert!(e.contains("out of bounds"), "the failure does not carry Futhark's message: {e}"),
        o => panic!("the failing program was scored: {o:?}"),
    }
    for (i, t) in [(0, &ts[0]), (2, &ts[1])] {
        assert_eq!(got.outcomes[i][0], Outcome::Quality(per_candidate(&k, &sig, t, &cases, &base.join(format!("single{i}")))), "program {i}");
    }
    assert_eq!((got.cost.compiles, got.cost.processes), (1, 2), "one compile; a second process after the failure");
    let _ = std::fs::remove_dir_all(&base);
}

/// The largest binding index of the generated C, which wgpu bounds at 1000 (the halving's trigger before a run).
#[test]
fn the_largest_binding_index_is_read_from_the_generated_c() {
    let c = "static int k_binding_indices[] = {7, 8, 9};\n{.name =\"a\", .scalars_binding =1007, .x =3}\nstatic int j_binding_indices[] = {1011};\n";
    assert_eq!(max_binding_index(c), 1011);
    assert_eq!(max_binding_index("no kernels"), 0);
}

/// The time steps' arithmetic: the fastest run after the warm-up; no timed pass in any run is no GPU work (infinite,
/// no speed credit); a mix is an error; a failure carries its message.
#[test]
fn a_timing_is_the_fastest_run_after_the_warm_up() {
    assert_eq!(Outcome::Time(vec![Some(90.0), Some(7.0), Some(5.0), Some(6.0)]).fastest_ns(), Ok(5.0));
    assert_eq!(Outcome::Time(vec![None, None, None]).fastest_ns(), Ok(f64::INFINITY));
    assert!(Outcome::Time(vec![Some(9.0), Some(5.0), None]).fastest_ns().is_err());
    assert_eq!(Outcome::Failed("x".into()).fastest_ns(), Err("x".into()));
}

/// The results file's records, payloads of any bytes included.
#[test]
fn the_results_records_parse() {
    let mut raw = b"start -1 ok 17 0 0\nctx -1 ok 18 9 0\nq 0 ok 5 2 3\n".to_vec();
    raw.extend_from_slice(b"a\nb");
    raw.extend_from_slice(b"t 1 err 0 0 4\noops");
    let r = parse_records(&raw).unwrap();
    assert_eq!(r.len(), 4);
    assert_eq!((r[2].tag.as_str(), r[2].index, r[2].ok, r[2].ns, r[2].pipeline_ns, r[2].payload.as_slice()), ("q", 0, true, 5, 2, &b"a\nb"[..]));
    assert_eq!(r[1].pipeline_ns, 9);
    assert_eq!((r[3].tag.as_str(), r[3].ok, r[3].payload.as_slice()), ("t", false, &b"oops"[..]));
    assert!(parse_records(b"q 0 ok 5 0 9\nab").is_err(), "a record cut short");
}

/// The GPU backend's build, without the GPU: a batch of programs, timed pairs and the published program as a member
/// compiles with the branch compiler as a library, its driver written from the manifest, and links against the shim.
/// (The branch compiler and `cc`; no device.)
#[test]
fn the_branch_compilers_batch_compiles_and_links() {
    if !have(crate::futhark_wgpu_host::branch_compiler().to_str().unwrap()) || !have("cc") {
        eprintln!("skipped: the branch compiler or cc is not installed");
        return;
    }
    let p = problem("histogram").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let base = scratch("gpu_build");
    let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).unwrap();
    let host = Host::new(&k, &sig, cases.len(), Backend::Gpu { shim }, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&cases[0].stdin), yardstick: Some(p.yardstick) }).unwrap();
    let ts = trees(&k, 3, 0xBA7C_0004);
    let mut members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality, Step::Time { yardstick: false, runs: 2 }, Step::Time { yardstick: true, runs: 2 }] }).collect();
    members.push(Member { program: Program::Text(p.yardstick), steps: vec![Step::Time { yardstick: false, runs: 2 }, Step::Output] });
    let idx: Vec<usize> = (0..members.len()).collect();
    let (exe, cmds, stages) = host.compile(&members, &idx, &mut Cost::default()).unwrap();
    assert!(exe.is_file(), "no executable");
    assert_eq!(cmds.len(), 11);
    assert!(stages.futhark > 0.0 && stages.cc > 0.0 && stages.link > 0.0, "{stages:?}");
    let driver = std::fs::read_to_string(exe.parent().unwrap().join("driver.c")).unwrap();
    assert!(driver.contains("futhark_wgpu_shim_timed_passes") && driver.contains("futhark_project_opaque_tup5"), "the GPU driver reads the shim's passes and projects the cases' tuple");
    eprintln!("GPU batch build of 4 members: {stages:?}");
    let _ = std::fs::remove_dir_all(&base);
}

/// On the GPU: a batch of random programs gives each program's quality outputs exactly as its own per-candidate GPU
/// program does, and timed pairs inside the one process give real GPU times of the published program against itself.
/// (The branch compiler, `cc`, the GPU; run under the GPU lock.)
#[test]
fn on_the_gpu_a_batch_equals_the_per_candidate_programs_and_times_in_process() {
    if !have(crate::futhark_wgpu_host::branch_compiler().to_str().unwrap()) || !have("cc") {
        eprintln!("skipped: the branch compiler or cc is not installed");
        return;
    }
    let _gpu = crate::gpu_test_lock::acquire();
    let p = problem("histogram").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let base = scratch("gpu");
    let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).unwrap();
    let timing = cases.last().unwrap().stdin.clone();
    let host = Host::new(&k, &sig, cases.len(), Backend::Gpu { shim: shim.clone() }, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&timing), yardstick: Some(p.yardstick) }).unwrap();
    let ts = trees(&k, 4, 0xBA7C_0003);
    let mut members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality] }).collect();
    members.push(Member { program: Program::Text(p.yardstick), steps: vec![Step::Time { yardstick: false, runs: 3 }, Step::Time { yardstick: true, runs: 3 }, Step::Output] });
    let got = host.run(&members).unwrap();
    assert_eq!((got.cost.compiles, got.cost.processes), (1, 1));
    for (i, t) in ts.iter().enumerate() {
        let exe = crate::futhark_wgpu_host::compile_program_prebuilt(&program_cases(&k, &sig, t, cases.len()).unwrap(), &base.join(format!("single{i}")), &shim).unwrap();
        let (out, _) = crate::futhark_wgpu_host::run_program_bytes(&exe, &stdin_of(&cases), false, 1).unwrap();
        assert_eq!(got.outcomes[i][0], Outcome::Quality(read_binary(&out).unwrap()), "program {i}: the batch and its own GPU program differ");
    }
    let a = got.outcomes[4][0].fastest_ns().unwrap();
    let y = got.outcomes[4][1].fastest_ns().unwrap();
    assert!(a.is_finite() && y.is_finite() && a > 0.0 && y > 0.0, "the published program was not timed: {:?}", got.outcomes[4]);
    let want = crate::futhark_sig::run_exe(&compile_c(p.yardstick, &base.join("c")).unwrap(), &timing).unwrap();
    assert_eq!(got.outcomes[4][2], Outcome::Output(want), "the published program in the batch differs from futhark c");
    eprintln!("GPU batch: 5 members, compile {:.2} s, process {:.2} s (device {:.2}, pipelines {:.2}); A/A {:.4}", got.cost.compile.total(), got.cost.run.total, got.cost.run.device, got.cost.run.pipelines, a / y);
    let _ = std::fs::remove_dir_all(&base);
}

/// The rust-wgpu backend's build, without the GPU: a batch of programs, timed pairs and the published program as a
/// member compiles with the fork compiler into one Rust module, and rustc builds it into a dylib against the prebuilt
/// runtime with no warning (`-D warnings`); the commands name the entries. (The fork compiler and rustc; no device.)
#[test]
fn the_rust_backends_batch_compiles() {
    if !crate::futhark_rust::compiler().is_file() {
        eprintln!("skipped: the fork compiler is not installed");
        return;
    }
    let p = problem("histogram").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let base = scratch("rust_build");
    let rt = crate::futhark_rust::prebuild_rt(&base.join("rt")).unwrap();
    let host = Host::new(&k, &sig, cases.len(), Backend::Rust { rt }, &base.join("batch"), Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&cases[0].stdin), yardstick: Some(p.yardstick) }).unwrap();
    let ts = trees(&k, 3, 0xBA7C_0004);
    let mut members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality, Step::Time { yardstick: false, runs: 2 }, Step::Time { yardstick: true, runs: 2 }] }).collect();
    members.push(Member { program: Program::Text(p.yardstick), steps: vec![Step::Time { yardstick: false, runs: 2 }, Step::Output] });
    let idx: Vec<usize> = (0..members.len()).collect();
    let (dylib, cmds, stages) = host.compile(&members, &idx, &mut Cost::default()).unwrap();
    assert!(dylib.is_file(), "no dylib");
    let lines: Vec<&str> = cmds.iter().map(|c| c.line.as_str()).collect();
    assert_eq!(lines[..3], ["q q0", "t t0 2", "t yard 2"]);
    assert_eq!(lines[9..], ["t t3 2", "o t3"]);
    assert!(stages.futhark > 0.0 && stages.cc > 0.0, "{stages:?}");
    eprintln!("rust-wgpu batch build of 4 members: futhark {:.2} s, rustc {:.2} s", stages.futhark, stages.cc);
    let _ = std::fs::remove_dir_all(&base);
}

/// On the GPU: for each of the study's four problems, a batch of random programs and the published program gives,
/// through the rust-wgpu backend in this process, exactly the outcomes of the batch backend (the branch compiler's C
/// over the shim, its own process): every quality output, the published program's output, and real GPU times of the
/// published program against itself. (The fork compiler, rustc, the branch compiler, cc, the GPU; run under the GPU
/// lock.)
#[test]
fn on_the_gpu_the_rust_backend_equals_the_batch_backend() {
    if !have(crate::futhark_wgpu_host::branch_compiler().to_str().unwrap()) || !have("cc") || !crate::futhark_rust::compiler().is_file() {
        eprintln!("skipped: the branch compiler, cc or the fork compiler is not installed");
        return;
    }
    let _gpu = crate::gpu_test_lock::acquire();
    let base = scratch("rust_gpu");
    let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).unwrap();
    let rt = crate::futhark_rust::prebuild_rt(&base.join("rt")).unwrap();
    for (pi, name) in ["reduce", "scan", "histogram", "pathfinder"].into_iter().enumerate() {
        let p = problem(name).unwrap();
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let cases = load_cases(&p).unwrap();
        let timing = cases.last().unwrap().stdin.clone();
        let inputs = Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&timing), yardstick: Some(p.yardstick) };
        let batch = Host::new(&k, &sig, cases.len(), Backend::Gpu { shim: shim.clone() }, &base.join(format!("{name}_batch")), inputs).unwrap();
        let rust = Host::new(&k, &sig, cases.len(), Backend::Rust { rt: rt.clone() }, &base.join(format!("{name}_rust")), inputs).unwrap();
        let mut ts = trees(&k, 8, 0xBA7C_0010 + pi as u64);
        ts.push(crate::futhark_sig::build(&k, p.reference).unwrap());
        let mut members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality] }).collect();
        members.push(Member { program: Program::Text(p.yardstick), steps: vec![Step::Time { yardstick: false, runs: 3 }, Step::Time { yardstick: true, runs: 3 }, Step::Output] });
        let want = batch.run(&members).unwrap();
        let got = rust.run(&members).unwrap();
        let n = ts.len();
        for i in 0..n {
            assert_eq!(got.outcomes[i], want.outcomes[i], "{name} program {i}: the rust backend and the batch backend differ");
        }
        assert_eq!(got.outcomes[n][2], want.outcomes[n][2], "{name}: the published program's output differs");
        let a = got.outcomes[n][0].fastest_ns().unwrap();
        let y = got.outcomes[n][1].fastest_ns().unwrap();
        assert!(a.is_finite() && y.is_finite() && a > 0.0 && y > 0.0, "{name}: the published program was not timed: {:?}", got.outcomes[n]);
        let failed = (0..n).filter(|&i| matches!(got.outcomes[i][0], Outcome::Failed(_))).count();
        eprintln!(
            "{name}: {n} programs + the published one, {failed} failed on both paths; batch: compile {:.2} s, run {:.2} s; rust: compile {:.2} s (futhark {:.2}, rustc {:.2}), run {:.2} s (pipelines {:.2}); rust A/A {:.4}",
            want.cost.compile.total(),
            want.cost.run.total,
            got.cost.compile.total(),
            got.cost.compile.futhark,
            got.cost.compile.cc,
            got.cost.run.total,
            got.cost.run.pipelines,
            a / y
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// THE KINGDOM COMPILES TO RUST, without the GPU: for each of the study's four problems, a batch of 24 random programs
/// of the signature kingdom and the published reference (each a quality and a timed entry) compiles through the fork's
/// rust-wgpu backend and rustc (`-D warnings`) into one module. The branch compiler's webgpu build of the same batch
/// (C, the shim's link) is timed beside it, and its failure, if any, printed: it is the path the rust backend replaces.
/// (The fork compiler, rustc, the branch compiler, cc; no device.)
#[test]
fn every_problems_random_programs_compile_to_rust() {
    if !crate::futhark_rust::compiler().is_file() || !have(crate::futhark_wgpu_host::branch_compiler().to_str().unwrap()) || !have("cc") {
        eprintln!("skipped: the fork compiler, the branch compiler or cc is not installed");
        return;
    }
    let base = scratch("kingdom_rust");
    let rt = crate::futhark_rust::prebuild_rt(&base.join("rt")).unwrap();
    let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).unwrap();
    for (pi, name) in ["reduce", "scan", "histogram", "pathfinder"].into_iter().enumerate() {
        let p = problem(name).unwrap();
        let k = load(&p.instance).unwrap();
        let sig = signature(&p.instance).unwrap();
        let cases = load_cases(&p).unwrap();
        let inputs = Inputs { cases_stdin: &stdin_of(&cases), timing_stdin: Some(&cases[0].stdin), yardstick: Some(p.yardstick) };
        let mut ts = trees(&k, 24, 0xC0DE_0100 + pi as u64);
        ts.push(crate::futhark_sig::build(&k, p.reference).unwrap());
        let members: Vec<Member> = ts.iter().map(|t| Member { program: Program::Tree(t), steps: vec![Step::Quality, Step::Time { yardstick: false, runs: 1 }] }).collect();
        let idx: Vec<usize> = (0..members.len()).collect();
        let rust = Host::new(&k, &sig, cases.len(), Backend::Rust { rt: rt.clone() }, &base.join(format!("{name}_rust")), inputs).unwrap();
        let (_, _, rs) = rust.compile(&members, &idx, &mut Cost::default()).unwrap_or_else(|e| panic!("{name}: the rust-wgpu batch of {} programs did not compile:\n{e}", members.len()));
        let gpu = Host::new(&k, &sig, cases.len(), Backend::Gpu { shim: shim.clone() }, &base.join(format!("{name}_gpu")), inputs).unwrap();
        let t = std::time::Instant::now();
        let c = gpu.compile(&members, &idx, &mut Cost::default());
        let cs = t.elapsed().as_secs_f64();
        eprintln!(
            "{name}: {} programs; rust-wgpu: futhark {:.2} s + rustc {:.2} s; webgpu + cc: {:.2} s{}",
            members.len(),
            rs.futhark,
            rs.cc,
            cs,
            match c {
                Ok(_) => String::new(),
                Err(e) => format!(", FAILED: {}", e.lines().find(|l| l.contains("limitation") || l.contains("rror")).unwrap_or("")),
            }
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// The cross-path set: the published histogram program, its planted slowdown (the work twice), Fable's exact program
/// (`(tab (sum.n (tab (sel ...))))`, about 70 times the published one), a sum of the values in every bin, and a
/// constant (`tab 10`): five speeds.
const CROSS_PATH: [&str; 5] = [
    crate::futhark_sig::HISTOGRAM_REFERENCE,
    crate::futhark_sig::HISTOGRAM_PLANTED,
    "(tab (sum.n (tab (sel (sub (abs (sub (mod.k (get.n is j2)) (ival.k j1))) 1) (get.n vs j2) 0))))",
    "(tab (sum.n (tab (get.n vs j2))))",
    "(tab 10)",
];

/// The cross-path set builds in the histogram kingdom, and every program of it is timed (its work at the timing
/// input within `futhark_sig::TIMING_WORK_CAP`), so the GPU test below times each one on every path. CPU only.
#[test]
fn the_cross_path_programs_build_and_are_within_the_timing_work_bound() {
    let p = problem("histogram").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let interp = crate::futhark_sig::Interp::new(&k, &sig).unwrap();
    let timing = crate::futhark_sig::timing_input(&p).unwrap().expect("histogram's timing input");
    let sizes = sig.sizes(&read_binary(&timing).unwrap()).unwrap();
    for text in CROSS_PATH {
        let t = crate::futhark_sig::build(&k, text).unwrap_or_else(|e| panic!("{text}: {e}"));
        let w = interp.work(&t, &sizes).unwrap();
        assert!(w <= crate::futhark_sig::TIMING_WORK_CAP, "{text}: work {w} at the timing input, over the timing bound");
    }
}

/// The steps of `pairs` timed pairs after one discarded process-equivalent, the order alternating (the evolution's
/// `pair_steps` for a fresh program).
fn fresh_pairs(pairs: usize, runs: usize) -> Vec<Step> {
    let own = Step::Time { yardstick: false, runs };
    let yard = Step::Time { yardstick: true, runs };
    let mut v = vec![own];
    for i in 0..pairs {
        v.extend(if i % 2 == 0 { [own, yard] } else { [yard, own] });
    }
    v
}

/// The ratios candidate / published of [`fresh_pairs`]' outcomes, and the published program's fastest time.
fn pair_ratios(o: &[Outcome], pairs: usize) -> (Vec<f64>, f64) {
    o[0].fastest_ns().unwrap();
    let mut r = Vec::new();
    let mut yard = f64::INFINITY;
    for i in 0..pairs {
        let (a, b) = (o[1 + 2 * i].fastest_ns().unwrap(), o[2 + 2 * i].fastest_ns().unwrap());
        let (c, y) = if i % 2 == 0 { (a, b) } else { (b, a) };
        assert!(c.is_finite() && y.is_finite() && c > 0.0 && y > 0.0, "a timing that is not a finite GPU time: {o:?}");
        yard = yard.min(y);
        r.push(c / y);
    }
    (r, yard)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// THE THREE EVALUATION PATHS AGREE, on the GPU (`futhark_evolve --eval single | batch | rust`): on the cross-path set
/// and histogram's own timing input, (1) every path gives bit-identical quality outputs and so bit-identical quality
/// inputs; (2) every timing is a finite GPU time; (3) each path's A/A (the published program against itself, 11 pairs,
/// 20 runs each, as the study times) is within the study's gate of 5 %; (4) every program's median ratio against the
/// published program agrees between any two paths within the sum of their A/A noise floors (the largest |ratio - 1|
/// of each). The single path is a process a timing (`futhark_wgpu_host`), the batch path one process for all
/// (`Backend::Gpu`), the rust path this process (`Backend::Rust`). (The fork compiler, rustc, cc, the GPU; run
/// under the GPU lock.)
#[test]
fn on_the_gpu_the_three_evaluation_paths_agree() {
    if !crate::futhark_rust::compiler().is_file() || !have("cc") {
        eprintln!("skipped: the fork compiler or cc is not installed");
        return;
    }
    let _gpu = crate::gpu_test_lock::acquire();
    let (pairs, runs) = (11, 20);
    let p = problem("histogram").unwrap();
    let k = load(&p.instance).unwrap();
    let sig = signature(&p.instance).unwrap();
    let cases = load_cases(&p).unwrap();
    let timing = crate::futhark_sig::timing_input(&p).unwrap().expect("histogram's timing input");
    let base = scratch("three_paths");
    let shim = crate::futhark_wgpu_host::prebuild_shim(&base.join("shim")).unwrap();
    let rt = crate::futhark_rust::prebuild_rt(&base.join("rt")).unwrap();
    let trees: Vec<Tree> = CROSS_PATH.iter().map(|t| crate::futhark_sig::build(&k, t).unwrap()).collect();
    let all = stdin_of(&cases);
    // THE SINGLE PATH: a quality program and a timed program a candidate, a process a timing.
    let time_once = |exe: &Path| -> f64 {
        let (_, r) = crate::futhark_wgpu_host::run_program_bytes(exe, &timing, true, runs).unwrap();
        r.fastest_run_ns(runs).unwrap_or_else(|e| panic!("{}: {e}", exe.display()))
    };
    let yard_exe = crate::futhark_wgpu_host::compile_program_prebuilt(p.yardstick, &base.join("single_yard"), &shim).unwrap();
    let single_pairs = |exe: &Path| -> (Vec<f64>, f64) {
        time_once(exe);
        let mut r = Vec::new();
        let mut yard = f64::INFINITY;
        for i in 0..pairs {
            let (c, y) = if i % 2 == 0 {
                let c = time_once(exe);
                (c, time_once(&yard_exe))
            } else {
                let y = time_once(&yard_exe);
                (time_once(exe), y)
            };
            yard = yard.min(y);
            r.push(c / y);
        }
        (r, yard)
    };
    let mut quality: Vec<Vec<Vec<Array>>> = vec![Vec::new(); 3];
    let mut ratios: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 3];
    let mut aa: Vec<(Vec<f64>, f64)> = Vec::new();
    for (i, t) in trees.iter().enumerate() {
        let q = crate::futhark_wgpu_host::compile_program_prebuilt(&program_cases(&k, &sig, t, cases.len()).unwrap(), &base.join(format!("single_q{i}")), &shim).unwrap();
        let (out, _) = crate::futhark_wgpu_host::run_program_bytes(&q, &all, false, 1).unwrap();
        quality[0].push(read_binary(&out).unwrap());
        let e = crate::futhark_wgpu_host::compile_program_prebuilt(&crate::futhark_sig::program(&k, &sig, t).unwrap(), &base.join(format!("single_t{i}")), &shim).unwrap();
        ratios[0].push(single_pairs(&e).0);
    }
    aa.push(single_pairs(&yard_exe));
    // THE BATCH AND RUST PATHS: every program in one batch, the published program as a member for the A/A.
    let inputs = Inputs { cases_stdin: &all, timing_stdin: Some(&timing), yardstick: Some(p.yardstick) };
    for (path, backend) in [(1, Backend::Gpu { shim: shim.clone() }), (2, Backend::Rust { rt: rt.clone() })] {
        let host = Host::new(&k, &sig, cases.len(), backend, &base.join(format!("path{path}")), inputs).unwrap();
        let mut members: Vec<Member> = trees.iter().map(|t| Member { program: Program::Tree(t), steps: [vec![Step::Quality], fresh_pairs(pairs, runs)].concat() }).collect();
        members.push(Member { program: Program::Text(p.yardstick), steps: fresh_pairs(pairs, runs) });
        let got = host.run(&members).unwrap();
        for (o, text) in got.outcomes.iter().zip(CROSS_PATH) {
            match &o[0] {
                Outcome::Quality(v) => quality[path].push(v.clone()),
                other => panic!("path {path}, {text}: no quality outputs: {other:?}"),
            }
            ratios[path].push(pair_ratios(&o[1..], pairs).0);
        }
        aa.push(pair_ratios(&got.outcomes[trees.len()], pairs));
    }
    let names = ["single", "batch", "rust"];
    let floor: Vec<f64> = aa.iter().map(|(r, _)| r.iter().map(|x| (x - 1.0).abs()).fold(0.0, f64::max)).collect();
    let mut table = String::from("path\tpublished GPU us\tA/A min\tA/A max\tfloor");
    for c in CROSS_PATH {
        table.push_str(&format!("\t{c}"));
    }
    table.push('\n');
    for path in 0..3 {
        let (r, y) = &aa[path];
        table.push_str(&format!("{}\t{:.1}\t{:.4}\t{:.4}\t{:.4}", names[path], y / 1e3, r.iter().cloned().fold(f64::INFINITY, f64::min), r.iter().cloned().fold(0.0, f64::max), floor[path]));
        for v in &ratios[path] {
            table.push_str(&format!("\t{:.4}", median(v)));
        }
        table.push('\n');
    }
    eprintln!("{table}");
    for i in 0..trees.len() {
        for path in 1..3 {
            assert_eq!(quality[path][i], quality[0][i], "{}: the {} path's quality outputs differ from the single path's", CROSS_PATH[i], names[path]);
            let fit = cases.iter().zip(&quality[path][i]).filter(|(c, _)| c.split.in_fitness());
            let qa: Vec<u64> = fit.flat_map(|(c, g)| crate::futhark_sig::quality_inputs(g, &c.output)).map(f64::to_bits).collect();
            let fit0 = cases.iter().zip(&quality[0][i]).filter(|(c, _)| c.split.in_fitness());
            let q0: Vec<u64> = fit0.flat_map(|(c, g)| crate::futhark_sig::quality_inputs(g, &c.output)).map(f64::to_bits).collect();
            assert_eq!(qa, q0, "{}: the {} path's quality inputs differ", CROSS_PATH[i], names[path]);
        }
    }
    for path in 0..3 {
        let within = aa[path].0.iter().filter(|r| (*r - 1.0).abs() <= 0.05).count();
        assert_eq!(within, pairs, "the {} path's A/A gate: {within} of {pairs} pairs within 5 %\n{table}", names[path]);
    }
    for i in 0..trees.len() {
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let (ma, mb) = (median(&ratios[a][i]), median(&ratios[b][i]));
            assert!((ma / mb - 1.0).abs() <= floor[a] + floor[b], "{}: the {} path's median ratio {ma:.4} and the {} path's {mb:.4} differ by more than their A/A floors {:.4} + {:.4}\n{table}", CROSS_PATH[i], names[a], names[b], floor[a], floor[b]);
        }
    }
    let _ = std::fs::remove_dir_all(&base);
}
