// grid 16 16 4 4
// A tiled Gram: each workgroup of 16 x 16 threads owns a 64 x 64 block of H
// (each thread a 4 x 4 register block); 32 items at a time are staged in
// workgroup memory (the block's 64 row values and 64 column values per item),
// then every thread accumulates its 16 cells from shared memory. Rows are
// band-local as in the shipped kernel: the global row is r_off + lr, the
// accumulator h is r_rows x d, read-add-write. The positive control of
// docs/PLAN_hgp_gram.md step 0, written by hand.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
const TILE: u32 = 32u;
var<workgroup> tr: array<f32, 2048>;
var<workgroup> tc: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + TILE) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let e = li + k * 256u;
            let tt = e / 64u;
            let j = e % 64u;
            let t = t0 + tt;
            var vr = 0.0;
            var vc = 0.0;
            if (t < cfg.m) {
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) { vr = phi[t * cfg.d + gr]; }
                let c = col0 + j;
                if (c < cfg.d) { vc = phi[t * cfg.d + c]; }
            }
            tr[e] = vr;
            tc[e] = vc;
        }
        workgroupBarrier();
        let tn = min(TILE, cfg.m - t0);
        for (var tt: u32 = 0u; tt < tn; tt = tt + 1u) {
            let rb = tt * 64u + lid.y * 4u;
            let cb = tt * 64u + lid.x * 4u;
            let cv = vec4<f32>(tc[cb], tc[cb + 1u], tc[cb + 2u], tc[cb + 3u]);
            acc0 = acc0 + tr[rb] * cv;
            acc1 = acc1 + tr[rb + 1u] * cv;
            acc2 = acc2 + tr[rb + 2u] * cv;
            acc3 = acc3 + tr[rb + 3u] * cv;
        }
        workgroupBarrier();
    }
    let c0 = col0 + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr >= cfg.r_rows) { continue; }
        var a = acc0;
        if (i == 1u) { a = acc1; }
        if (i == 2u) { a = acc2; }
        if (i == 3u) { a = acc3; }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            if (c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + a[j];
            }
        }
    }
}
