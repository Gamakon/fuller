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
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = tid + k * 256u;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tt;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                let c = col0 + cc;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            sa[idx & 511u] = av;
            sb[idx & 511u] = bv;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 8u; tt = tt + 1u) {
            let bi = tt * 64u + tx * 4u;
            let b = vec4<f32>(sb[bi & 511u], sb[(bi + 1u) & 511u], sb[(bi + 2u) & 511u], sb[(bi + 3u) & 511u]);
            let ai = tt * 64u + ty * 4u;
            acc[0u] = acc[0u] + sa[ai & 511u] * b;
            acc[1u] = acc[1u] + sa[(ai + 1u) & 511u] * b;
            acc[2u] = acc[2u] + sa[(ai + 2u) & 511u] * b;
            acc[3u] = acc[3u] + sa[(ai + 3u) & 511u] * b;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let cbase = col0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let o = lr * d + cbase;
            if (cbase < d) { h[o] = h[o] + v.x; }
            if (cbase + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (cbase + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (cbase + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
