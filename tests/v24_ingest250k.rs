//! THE ANTI-SCAN TEST LOADS AND PLANS (fuller docs/PLAN_true_gep_wgsl.md task UI2): `problems/v24_ingest250k`,
//! v24_ingest5b's contract at 250,000 stream records in five microbatches of 50,000 and the same 2,500 oracles,
//! read from the exporter's case (`/Users/andrewmorgan/data/arm/v24_case_250k`). The plan test runs on the host;
//! the device test (named for the GPU, so the CPU run skips it) stages stage 1 and runs the reference once.
#![cfg(feature = "wgsl")]

use std::path::Path;

use fuller::wgsl::harness::{kernel_info, BufferDecl, Cmd, Init, LimitRule, Problem};

fn problem() -> Problem {
    Problem::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/v24_ingest250k")).expect("problems/v24_ingest250k")
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
}

/// At a 512 MiB binding limit (the adapter's is larger) the stream case plans: the setup projects per seed its
/// queries (rows 250,000..252,500) and five microbatches of 50,000 rows (409.6 MB each); every ingest dispatch binds
/// its microbatch with 50,000 workgroups, every search the 64 zero floats; the arenas are 8,388,608 words (32 MiB,
/// v24_ingest5b's: 268 bytes a record at stage 5 for both); one submission a stage and seed (20 syncs). At the
/// 128 MiB effective limit the microbatch's require names the binding limit.
#[test]
fn v24_ingest250k_plans_by_the_adapter_limit_with_32_mib_arenas() {
    let p = problem();
    assert_eq!(p.spec.binding_limit, LimitRule::Adapter);
    let c = &p.suites[0].cases[0];
    assert_eq!((c.params["n_stream"], c.params["per_section"], c.params["n_oracle"]), (250_000.0, 50_000.0, 2_500.0));
    let info = kernel_info(&p.spec, &p.reference).expect("the reference reads");
    let limit = 512u64 << 20;
    let plan = p.plan(c, limit, Some(&info)).expect("plans at 512 MiB");
    let buf = |id: usize| -> &BufferDecl { &plan.buffers[id] };
    assert_eq!(plan.setup_dispatch.len(), 4 * 6);
    for (n, d) in plan.setup_dispatch.iter().enumerate() {
        let (s, k) = (n / 6, n % 6);
        let at = |b: u32| buf(d.bind.iter().find(|x| x.0 == b).expect("bound").1);
        let Init::Words(w) = &at(4).init else { panic!("pcfg") };
        let (name, first, rows) = if k == 0 { (format!("queries{s}"), 250_000u32, 2_500u32) } else { (format!("mb{s}_{}", k - 1), 50_000 * (k as u32 - 1), 50_000) };
        assert_eq!((at(3).name.clone(), at(3).bytes, d.workgroups), (name, u64::from(rows) * 2048 * 4, [rows, 1, 1]));
        assert_eq!(u32s(w)[2..], [first, rows]);
    }
    for s in 0..4 {
        for arena in [format!("iu{s}"), format!("if{s}")] {
            let b = plan.buffers.iter().find(|b| b.name == arena).unwrap_or_else(|| panic!("{arena}"));
            assert_eq!(b.bytes, 8_388_608 * 4, "{arena}");
        }
    }
    let ds: Vec<&Cmd> = plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).collect();
    assert_eq!(ds.len(), 5 * 4 * 2);
    for (n, d) in ds.iter().enumerate() {
        let Cmd::Dispatch { bind, entry, workgroups, .. } = d else { unreachable!() };
        let (stage, seed) = (n / 8, (n % 8) / 2);
        let keys = buf(bind.iter().find(|x| x.0 == 0).expect("keys").1);
        let Init::Words(w) = &buf(bind.iter().find(|x| x.0 == 5).expect("cfg").1).init else { panic!("cfg") };
        assert_eq!(u32s(w)[1..4], [stage as u32 + 1, 50_000 * stage as u32, 50_000 * (stage as u32 + 1)]);
        if *entry == 0 {
            assert_eq!((keys.name.clone(), keys.bytes, *workgroups), (format!("mb{seed}_{stage}"), 50_000 * 2048 * 4, [50_000, 1, 1]));
        } else {
            assert_eq!((keys.name.as_str(), keys.bytes, *workgroups), ("nokeys", 256, [2_500, 1, 1]));
        }
    }
    assert_eq!(plan.run.iter().filter(|x| matches!(x, Cmd::Sync)).count(), 5 * 4);
    assert_eq!(plan.outputs, vec![5 * 4 * 2500 * 64 * 4]);
    let e = p.plan(c, 128 << 20, Some(&info)).unwrap_err();
    assert!(e.contains("per_section * d * 4 <= binding_limit"), "{e}");
}

/// STAGE 1 ON THE DEVICE (GPU; run by the session that holds the GPU lock): the stream case at stage 1 only
/// (50,000 records), staged at the adapter's limit with every seed's microbatch keys and queries projected on the
/// device, then the reference run once and judged on the device: two windows (stage 1 x two oracle halves), every
/// oracle judged, no broken entry. The reference's search sees no keys here, so it returns pads (P 0, printed).
#[cfg(feature = "gpu")]
#[test]
fn v24_ingest250k_reference_runs_stage_1_on_the_device() {
    use fuller::wgsl::harness::{DeviceExecutor, Executor};
    use fuller::wgsl::oracle::device::{Device, Fuel};
    let t = std::time::Instant::now();
    let p = problem();
    let dev = Device::new().expect("device");
    let exec = DeviceExecutor::for_problem(&dev, &p, Fuel::Proven("the problem's hand-written reference kernel"));
    let limit = exec.binding_limit();
    assert!(limit >= 50_000 * 2048 * 4, "the adapter's limit {limit} is under a microbatch's 409.6 MB of keys");
    let mut c = p.suites[0].cases[0].clone();
    c.params.insert("stages".into(), 1.0);
    let (k, info) = exec.compile(&p, &p.reference).expect("the reference compiles");
    let data = p.case_data(&c, limit).expect("the case's files");
    let env = p.env(&c, limit).expect("env");
    let plan = p.plan(&c, limit, Some(&info)).expect("plans at stage 1");
    let (staged, stage_ms) = exec.stage(&plan, &data).expect("staged");
    let r = exec.run(&k, &plan, &data, &staged, true).expect("the reference runs");
    let out = p.pooled_output().expect("pooled").clone();
    let w = exec.judge_pooled(&p, &out, &env, &data, &r.outputs[0]).expect("judged on the device");
    println!("v24_ingest250k stage 1 at {limit} bytes a binding: staged in {stage_ms:.0} ms, the reference {:.0} ms GPU, {:.1} s in all; {w:?}", r.gpu_ms, t.elapsed().as_secs_f64());
    assert_eq!(w.len(), 2);
    assert_eq!(w.iter().map(|x| x.queries).sum::<u64>(), 2_500);
    assert!(w.iter().all(|x| x.stage == 1 && x.broken == 0), "{w:?}");
}
