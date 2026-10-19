// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var q = 0u; q < 2u; q = q + 1u) {
            let e = lid + q * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var av = 0.0;
            var bv = 0.0;
            if (t < m) {
                let lr = rowBase + j;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    av = phi[t * d + cfg.r_off + lr];
                }
                let c = colBase + j;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            sa[e & 511u] = av;
            sb[e & 511u] = bv;
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k = k + 1u) {
            let bo = k * 64u + tx * 4u;
            let b = vec4<f32>(sb[bo & 511u], sb[(bo + 1u) & 511u], sb[(bo + 2u) & 511u], sb[(bo + 3u) & 511u]);
            let ao = k * 64u + ty * 4u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = sa[(ao + i) & 511u];
                acc[i & 3u] = acc[i & 3u] + a * b;
            }
        }
        workgroupBarrier();
    }

    let c0 = colBase + tx * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
