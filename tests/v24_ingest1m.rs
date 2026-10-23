//! THE 1,000,000-RECORD ANTI-SCAN PROBLEM PLANS (fuller docs/PLAN_true_gep_wgsl.md task UI2): `problems/v24_ingest1m`,
//! five microbatches of 200,000 arriving in sub-batches of 50,000 (a dispatch holds at most 65,535 workgroups a
//! dimension). The 1M set is not built yet, so the case lacks the exporter's sizes (nnz, accept_n1..5,
//! today_settle_ms_1..5); this host test gives them placeholder values of 1 to plan the program's layout only (no data
//! is read, nothing runs).
#![cfg(feature = "wgsl")]

use std::path::Path;

use fuller::wgsl::harness::{kernel_info, Cmd, Init, LimitRule, Problem};

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
}

/// At a 512 MiB binding limit the stream case plans: per seed the queries (rows 1,000,000..1,002,500) and 20
/// sub-batch key buffers of 50,000 rows (409.6 MB each, rows j * 50,000 onward in order); every ingest dispatch binds
/// its sub-batch with 50,000 workgroups and a cfg naming the sub-batch's rows, every search the 64 zero floats and a
/// cfg naming the whole section; 32 MiB arenas; one submission a stage and seed (20 syncs). At 128 MiB the sub-batch's
/// require names the binding limit.
#[test]
fn v24_ingest1m_plans_sub_batches_of_50000_by_the_adapter_limit() {
    let p = Problem::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("problems/v24_ingest1m")).expect("problems/v24_ingest1m");
    assert_eq!(p.spec.binding_limit, LimitRule::Adapter);
    let mut c = p.suites[0].cases[0].clone();
    for k in ["nnz", "accept_n1", "accept_n2", "accept_n3", "accept_n4", "accept_n5", "today_settle_ms_1", "today_settle_ms_2", "today_settle_ms_3", "today_settle_ms_4", "today_settle_ms_5"] {
        assert!(!c.params.contains_key(k), "{k} is filled: replace these placeholders with the case");
        c.params.insert(k.into(), 1.0);
    }
    let info = kernel_info(&p.spec, &p.reference).expect("the reference reads");
    let plan = p.plan(&c, 512 << 20, Some(&info)).expect("plans at 512 MiB");
    assert_eq!(plan.setup_dispatch.len(), 4 * (1 + 5 * 4));
    for (n, d) in plan.setup_dispatch.iter().enumerate() {
        let (s, k) = (n / 21, n % 21);
        let at = |b: u32| &plan.buffers[d.bind.iter().find(|x| x.0 == b).expect("bound").1];
        let Init::Words(w) = &at(4).init else { panic!("pcfg") };
        let (name, first, rows) = if k == 0 { (format!("queries{s}"), 1_000_000u32, 2_500u32) } else { (format!("mb{s}_{}_{}", (k - 1) / 4, (k - 1) % 4), 50_000 * (k as u32 - 1), 50_000) };
        assert_eq!((at(3).name.clone(), at(3).bytes, d.workgroups), (name, u64::from(rows) * 2048 * 4, [rows, 1, 1]));
        assert_eq!(u32s(w)[2..], [first, rows]);
    }
    let ds: Vec<&Cmd> = plan.run.iter().filter(|x| matches!(x, Cmd::Dispatch { .. })).collect();
    assert_eq!(ds.len(), 5 * 4 * (4 + 1));
    for (n, d) in ds.iter().enumerate() {
        let Cmd::Dispatch { bind, entry, workgroups, .. } = d else { unreachable!() };
        let (stage, seed, slot) = (n / 20, (n % 20) / 5, n % 5);
        let keys = &plan.buffers[bind.iter().find(|x| x.0 == 0).expect("keys").1];
        let Init::Words(w) = &plan.buffers[bind.iter().find(|x| x.0 == 5).expect("cfg").1].init else { panic!("cfg") };
        let cfg = u32s(w);
        assert_eq!(cfg[1], stage as u32 + 1);
        if slot < 4 {
            let begin = 200_000 * stage as u32 + 50_000 * slot as u32;
            assert_eq!((*entry, keys.name.clone(), keys.bytes, *workgroups), (0, format!("mb{seed}_{stage}_{slot}"), 50_000 * 2048 * 4, [50_000, 1, 1]));
            assert_eq!(cfg[2..4], [begin, begin + 50_000]);
        } else {
            assert_eq!((*entry, keys.name.as_str(), *workgroups), (1, "nokeys", [2_500, 1, 1]));
            assert_eq!(cfg[2..4], [200_000 * stage as u32, 200_000 * (stage as u32 + 1)]);
        }
    }
    for s in 0..4 {
        for arena in [format!("iu{s}"), format!("if{s}")] {
            assert_eq!(plan.buffers.iter().find(|b| b.name == arena).unwrap_or_else(|| panic!("{arena}")).bytes, 8_388_608 * 4);
        }
    }
    assert_eq!(plan.run.iter().filter(|x| matches!(x, Cmd::Sync)).count(), 5 * 4);
    let e = p.plan(&c, 128 << 20, Some(&info)).unwrap_err();
    assert!(e.contains("chunk * d * 4 <= binding_limit"), "{e}");
}
