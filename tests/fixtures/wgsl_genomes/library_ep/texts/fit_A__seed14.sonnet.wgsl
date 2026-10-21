// grid 16 8 4 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ctile: array<f32, 512>;
var<workgroup> rtile: array<f32, 128>;

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let colbase = wg.x * 64u;
    let rowbase = wg.y * 16u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }

        // column tile: 8 items x 64 columns
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = tid + k * 128u;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tt;
            let c = colbase + cc;
            var v: f32 = 0.0;
            if (t < m && c < d) {
                v = phi[t * d + c];
            }
            ctile[idx] = v;
        }
        // row tile: 8 items x 16 rows
        {
            let tt = tid >> 4u;
            let rr = tid & 15u;
            let t = t0 + tt;
            let lrr = rowbase + rr;
            let gr = cfg.r_off + lrr;
            var v: f32 = 0.0;
            if (t < m && lrr < cfg.r_rows && gr < d) {
                v = phi[t * d + gr];
            }
            rtile[tid] = v;
        }
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 8u; tt = tt + 1u) {
            let cb = tt * 64u + lx * 4u;
            let a = vec4<f32>(ctile[cb], ctile[cb + 1u], ctile[cb + 2u], ctile[cb + 3u]);
            let rb = tt * 16u + ly * 2u;
            let r0 = rtile[rb];
            let r1 = rtile[rb + 1u];
            acc0 = acc0 + a * r0;
            acc1 = acc1 + a * r1;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let c0 = colbase + lx * 4u;
    let lrA = rowbase + ly * 2u;
    let lrB = lrA + 1u;

    if (lrA < cfg.r_rows && cfg.r_off + lrA < d) {
        let b = lrA * d + c0;
        if (c0 < d) { h[b] = h[b] + acc0.x; }
        if (c0 + 1u < d) { h[b + 1u] = h[b + 1u] + acc0.y; }
        if (c0 + 2u < d) { h[b + 2u] = h[b + 2u] + acc0.z; }
        if (c0 + 3u < d) { h[b + 3u] = h[b + 3u] + acc0.w; }
    }
    if (lrB < cfg.r_rows && cfg.r_off + lrB < d) {
        let b = lrB * d + c0;
        if (c0 < d) { h[b] = h[b] + acc1.x; }
        if (c0 + 1u < d) { h[b + 1u] = h[b + 1u] + acc1.y; }
        if (c0 + 2u < d) { h[b + 2u] = h[b + 2u] + acc1.z; }
        if (c0 + 3u < d) { h[b + 3u] = h[b + 3u] + acc1.w; }
    }
}
