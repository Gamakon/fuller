// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> tc: array<f32, 512>;
var<workgroup> tr: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let cbase = wid.x * 32u;
    let rbase = wid.y * 32u;
    let tx = lid.x;
    let ty = lid.y;
    let li = ty * 8u + tx;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = li + k * 64u;
            let tt = idx >> 5u;
            let j = idx & 31u;
            let t = t0 + tt;
            let col = cbase + j;
            let lrow = rbase + j;
            let grow = cfg.r_off + lrow;
            var cv: f32 = 0.0;
            var rv: f32 = 0.0;
            if (t < m) {
                if (col < d) { cv = phi[t * d + col]; }
                if (lrow < cfg.r_rows && grow < d) { rv = phi[t * d + grow]; }
            }
            tc[idx] = cv;
            tr[idx] = rv;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let cb = tt * 32u + tx * 4u;
            let rb = tt * 32u + ty * 4u;
            let c = vec4<f32>(tc[cb], tc[cb + 1u], tc[cb + 2u], tc[cb + 3u]);
            let r = vec4<f32>(tr[rb], tr[rb + 1u], tr[rb + 2u], tr[rb + 3u]);
            a0 = a0 + r.x * c;
            a1 = a1 + r.y * c;
            a2 = a2 + r.z * c;
            a3 = a3 + r.w * c;
        }
        workgroupBarrier();
    }

    let col0 = cbase + tx * 4u;
    let row0 = rbase + ty * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        if (lr < cfg.r_rows) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let base = lr * d + col0;
            if (col0 < d) { h[base] = h[base] + v.x; }
            if (col0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (col0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (col0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
