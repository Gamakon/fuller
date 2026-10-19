// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        acc[j & 3u] = vec4<f32>(0.0);
    }
    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        workgroupBarrier();
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let base = t * d;
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[base + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[base + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let bi = k * 64u + lid.x * 4u;
            let b = vec4<f32>(sb[bi & 1023u], sb[(bi + 1u) & 1023u],
                              sb[(bi + 2u) & 1023u], sb[(bi + 3u) & 1023u]);
            let ai = k * 64u + lid.y * 4u;
            let a = vec4<f32>(sa[ai & 1023u], sa[(ai + 1u) & 1023u],
                              sa[(ai + 2u) & 1023u], sa[(ai + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
    }
    let cb = col0 + lid.x * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y * 4u + j;
        if (lr < cfg.r_rows) {
            let v = acc[j & 3u];
            let o = lr * d + cb;
            if (cb < d) { h[o] = h[o] + v.x; }
            if (cb + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (cb + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (cb + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
