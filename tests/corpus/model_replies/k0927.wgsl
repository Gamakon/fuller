// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // stage 8 items x 64 rows and 64 cols
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let idx = l + e * 256u;
            let k = idx >> 6u;
            let i = idx & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = row0 + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = col0 + i;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let o = k * 64u;
            let ra = o + ty * 4u;
            let rb = o + tx * 4u;
            let a = vec4<f32>(sa[ra & 511u], sa[(ra + 1u) & 511u], sa[(ra + 2u) & 511u], sa[(ra + 3u) & 511u]);
            let b = vec4<f32>(sb[rb & 511u], sb[(rb + 1u) & 511u], sb[(rb + 2u) & 511u], sb[(rb + 3u) & 511u]);
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let cb = col0 + tx * 4u;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = row0 + ty * 4u + r;
        if (lr < cfg.r_rows) {
            var v = a0;
            if (r == 1u) { v = a1; }
            if (r == 2u) { v = a2; }
            if (r == 3u) { v = a3; }
            let base = lr * d + cb;
            if (cb < d) { h[base] = h[base] + v.x; }
            if (cb + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (cb + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (cb + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
