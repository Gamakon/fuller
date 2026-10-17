// passes 1
// THE v24 MEASUREMENT REFERENCE. NOT AN INDEX DESIGN: designs are evolved,
// never hand-written (fuller docs/PLAN_v24_ingest_evolution.md). This kernel
// exists for the harness's compile and timing paths and as the trivially
// correct baseline of the contract (problems/v24_ingest_small/contract.md):
// `ingest` does nothing and keeps no index; `search` is exact brute force
// over this seed's arrived keys, each query's C_MAX best rows by the cosine
// of the f32 keys. Sixteen threads a query scan the arrived rows strided,
// each keeping its own best C_MAX, sorted; every thread merges the
// sixteen lists and writes its own four of the C_MAX slots (t, t + 16, ..).
// Slots it cannot fill (fewer than C_MAX arrived rows with a nonzero key)
// stay pads.

struct Cfg {
    seed_slot: u32,
    stage: u32,
    arrived_begin: u32,
    arrived_end: u32,
    n_queries: u32,
    pass_index: u32,
    n_passes: u32,
    c_max: u32,
};

@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;

const C_MAX: u32 = 64u;
// 2,048 dimensions as vec4.
const D4: u32 = 512u;
const THREADS: u32 = 16u;
const PAD: u32 = 0xFFFFFFFFu;
const NONE: f32 = -3.0e38;

// THREADS * C_MAX entries: each thread's list, thread-major.
var<workgroup> wscore: array<f32, 1024>;
var<workgroup> wrow: array<u32, 1024>;

@compute @workgroup_size(1)
fn ingest() {
}

@compute @workgroup_size(16)
fn search(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let q = wid.x;
    let t = lid.x;
    var best: array<f32, 64>;
    var rows: array<u32, 64>;
    for (var j = 0u; j < C_MAX; j = j + 1u) {
        best[j] = NONE;
        rows[j] = PAD;
    }
    if (q < cfg.n_queries) {
        var qn = 0.0;
        for (var d = 0u; d < D4; d = d + 1u) {
            let v = queries[q * D4 + d];
            qn = qn + dot(v, v);
        }
        for (var r = t; r < cfg.arrived_end; r = r + 16u) {
            var s = 0.0;
            var kn = 0.0;
            for (var d = 0u; d < D4; d = d + 1u) {
                // The row masked under 2^19 (the full set has 25,000): the
                // index provably fits 32 bits.
                let k = keys[(r & 0x7FFFFu) * D4 + d];
                s = s + dot(k, queries[q * D4 + d]);
                kn = kn + dot(k, k);
            }
            let denom = sqrt(qn * kn);
            var score = NONE;
            if (denom > 0.0) {
                score = s / denom;
            }
            if (score > best[C_MAX - 1u]) {
                // A sorted insert: carry the new entry down the list,
                // swapping it past every entry it beats (ties keep the
                // earlier entry first).
                var cs = score;
                var cr = r;
                for (var j = 0u; j < C_MAX; j = j + 1u) {
                    if (cs > best[j]) {
                        let ts = best[j];
                        let tr = rows[j];
                        best[j] = cs;
                        rows[j] = cr;
                        cs = ts;
                        cr = tr;
                    }
                }
            }
        }
    }
    for (var j = 0u; j < C_MAX; j = j + 1u) {
        wscore[t * C_MAX + j] = best[j];
        wrow[t * C_MAX + j] = rows[j];
    }
    workgroupBarrier();
    // Every thread merges the sixteen lists (the same result in each) and
    // writes its own four slots, t + 16 i: one writer a slot.
    if (q < cfg.n_queries) {
        for (var j = 0u; j < C_MAX; j = j + 1u) {
            best[j] = NONE;
            rows[j] = PAD;
        }
        for (var i = 0u; i < THREADS * C_MAX; i = i + 1u) {
            let score = wscore[i];
            let r = wrow[i];
            if (r != PAD && score > best[C_MAX - 1u]) {
                // A sorted insert: carry the new entry down the list,
                // swapping it past every entry it beats (ties keep the
                // earlier entry first).
                var cs = score;
                var cr = r;
                for (var j = 0u; j < C_MAX; j = j + 1u) {
                    if (cs > best[j]) {
                        let ts = best[j];
                        let tr = rows[j];
                        best[j] = cs;
                        rows[j] = cr;
                        cs = ts;
                        cr = tr;
                    }
                }
            }
        }
        cands[q * C_MAX + t] = rows[t];
        cands[q * C_MAX + 16u + t] = rows[16u + t];
        cands[q * C_MAX + 32u + t] = rows[32u + t];
        cands[q * C_MAX + 48u + t] = rows[48u + t];
    }
}
