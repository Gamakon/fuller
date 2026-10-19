// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> colT: array<f32, 512>;
var<workgroup> rowT: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 8u + lid.x;
    let colBase = wg.x * 32u;
    let rowBase = wg.y * 32u;

    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let idx = tid + i * 64u;
            let k = idx >> 5u;
            let j = idx & 31u;
            let t = t0 + k;
            let col = colBase + j;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            var cv: f32 = 0.0;
            var rv: f32 = 0.0;
            if (t < m) {
                if (col < d) {
                    cv = phi[t * d + col];
                }
                if (lr < cfg.r_rows && gr < d) {
                    rv = phi[t * d + gr];
                }
            }
            colT[idx] = cv;
            rowT[idx] = rv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let cb = k * 32u + lid.x * 4u;
            let rb = k * 32u + lid.y * 4u;
            let b = vec4<f32>(colT[cb], colT[cb + 1u], colT[cb + 2u], colT[cb + 3u]);
            acc[0] = acc[0] + rowT[rb] * b;
            acc[1] = acc[1] + rowT[rb + 1u] * b;
            acc[2] = acc[2] + rowT[rb + 2u] * b;
            acc[3] = acc[3] + rowT[rb + 3u] * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + lid.x * 4u;
    let r0 = rowBase + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows) {
            let a = acc[i];
            let base = lr * d;
            if (c0 < d) { h[base + c0] = h[base + c0] + a.x; }
            if (c0 + 1u < d) { h[base + c0 + 1u] = h[base + c0 + 1u] + a.y; }
            if (c0 + 2u < d) { h[base + c0 + 2u] = h[base + c0 + 2u] + a.z; }
            if (c0 + 3u < d) { h[base + c0 + 3u] = h[base + c0 + 3u] + a.w; }
        }
    }
}