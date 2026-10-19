// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        acc[j & 3u] = vec4<f32>(0.0);
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = tid + k * 256u;
            let it = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + it;
            var vb = 0.0;
            var va = 0.0;
            if (t < m) {
                let c = col0 + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sb[idx & 1023u] = vb;
            sa[idx & 1023u] = va;
        }
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            let ba = i * 64u + lid.y * 4u;
            let bb = i * 64u + lid.x * 4u;
            let a = vec4<f32>(sa[ba & 1023u], sa[(ba + 1u) & 1023u], sa[(ba + 2u) & 1023u], sa[(ba + 3u) & 1023u]);
            let b = vec4<f32>(sb[bb & 1023u], sb[(bb + 1u) & 1023u], sb[(bb + 2u) & 1023u], sb[(bb + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }
    let cb = col0 + lid.x * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[j & 3u];
            let o = lr * d + cb;
            if (cb < d) { h[o] = h[o] + v.x; }
            if (cb + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (cb + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (cb + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
