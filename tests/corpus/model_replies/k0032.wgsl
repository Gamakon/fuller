// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 8u + lid.x;
    let rowBase = wg.y * 32u;
    let colBase = wg.x * 32u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            let idx = tid + i * 64u;
            let t = idx >> 5u;
            let j = idx & 31u;
            let item = t0 + t;
            var va = 0.0;
            var vb = 0.0;
            if (item < m) {
                let lrr = rowBase + j;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[item * d + gr];
                }
                let c = colBase + j;
                if (c < d) {
                    vb = phi[item * d + c];
                }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let ao = k * 32u + lid.y * 4u;
            let bo = k * 32u + lid.x * 4u;
            let a = vec4<f32>(sa[ao & 511u], sa[(ao + 1u) & 511u], sa[(ao + 2u) & 511u], sa[(ao + 3u) & 511u]);
            let b = vec4<f32>(sb[bo & 511u], sb[(bo + 1u) & 511u], sb[(bo + 2u) & 511u], sb[(bo + 3u) & 511u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }
    let c0 = colBase + lid.x * 4u;
    let r0 = rowBase + lid.y * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[i & 3u];
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
