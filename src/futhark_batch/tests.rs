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
