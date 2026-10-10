// grid 16 16 4 4
// A tiled Gram: each workgroup of 16 x 16 threads owns a 64 x 64 block of H
// (each thread a 4 x 4 register block); 32 items at a time are staged in
// workgroup memory as vec4 groups (16 groups of 4 rows and 16 of 4 columns
// per item), so a thread reads its 4 row values and 4 column values as two
// 16-byte loads per item and does 16 multiply-adds. Rows are band-local as
// in the shipped kernel: the global row is r_off + lr, the accumulator h is
// r_rows x d, read-add-write. The positive control of docs/PLAN_hgp_gram.md
// step 0, written by hand.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
const TILE: u32 = 32u;
var<workgroup> tr: array<vec4<f32>, 512>;
var<workgroup> tc: array<vec4<f32>, 512>;
fn load4(t: u32, base: u32, bound: u32) -> vec4<f32> {
    // Four consecutive values of item t starting at column base, zero past bound.
    let p = t * cfg.d + base;
    var v = vec4<f32>(0.0);
    if (base + 3u < bound) {
        v = vec4<f32>(phi[p], phi[p + 1u], phi[p + 2u], phi[p + 3u]);
    } else {
        if (base < bound) { v.x = phi[p]; }
        if (base + 1u < bound) { v.y = phi[p + 1u]; }
        if (base + 2u < bound) { v.z = phi[p + 2u]; }
    }
    return v;
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    // The band's rows end at r_rows and the matrix at d: a row group past
    // either reads zero, so it contributes nothing and stores nothing.
    let row_bound = min(cfg.r_rows, cfg.d - min(cfg.r_off, cfg.d));
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + TILE) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = li + k * 256u;      // 0..511: item tt, group g of 4
            let tt = e / 16u;
            let g = e % 16u;
            let t = t0 + tt;
            var vr = vec4<f32>(0.0);
            var vc = vec4<f32>(0.0);
            if (t < cfg.m) {
                let lr = row0 + g * 4u;
                if (lr < row_bound) { vr = load4(t, cfg.r_off + lr, cfg.r_off + row_bound); }
                let c = col0 + g * 4u;
                if (c < cfg.d) { vc = load4(t, c, cfg.d); }
            }
            tr[e] = vr;
            tc[e] = vc;
        }
        workgroupBarrier();
        let tn = min(TILE, cfg.m - t0);
        for (var tt: u32 = 0u; tt < tn; tt = tt + 1u) {
            let rv = tr[tt * 16u + lid.y];
            let cv = tc[tt * 16u + lid.x];
            acc0 = acc0 + rv.x * cv;
            acc1 = acc1 + rv.y * cv;
            acc2 = acc2 + rv.z * cv;
            acc3 = acc3 + rv.w * cv;
        }
        workgroupBarrier();
    }
    let c0 = col0 + lid.x * 4u;
    let lr0 = row0 + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = lr0 + i;
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
