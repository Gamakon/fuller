// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 512>;
var<workgroup> sB: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let rowBase = wg.y * 64u;
    let colBase = wg.x * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lr = rowBase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sA[e] = va;
            sB[e] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let pa = k * 64u + lid.y * 4u;
            let pb = k * 64u + lid.x * 4u;
            let a = vec4<f32>(sA[pa], sA[pa + 1u], sA[pa + 2u], sA[pa + 3u]);
            let b = vec4<f32>(sB[pb], sB[pb + 1u], sB[pb + 2u], sB[pb + 3u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + v.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
