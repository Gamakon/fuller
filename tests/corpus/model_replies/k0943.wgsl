// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let idx = lid + q * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            let col = colbase + j;
            let lr = rowbase + j;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (col < d) { vb = phi[t * d + col]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[idx & 511u] = vb;
            sa[idx & 511u] = va;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let o = k * 64u;
            let bo = o + tx * 4u;
            let b = vec4<f32>(sb[bo & 511u], sb[(bo + 1u) & 511u], sb[(bo + 2u) & 511u], sb[(bo + 3u) & 511u]);
            let ao = o + ty * 4u;
            acc0 = acc0 + sa[ao & 511u] * b;
            acc1 = acc1 + sa[(ao + 1u) & 511u] * b;
            acc2 = acc2 + sa[(ao + 2u) & 511u] * b;
            acc3 = acc3 + sa[(ao + 3u) & 511u] * b;
        }
        workgroupBarrier();
    }

    let c0 = colbase + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            var a = acc0;
            if (i == 1u) { a = acc1; }
            if (i == 2u) { a = acc2; }
            if (i == 3u) { a = acc3; }
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + a.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + a.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + a.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + a.w; }
        }
    }
}
