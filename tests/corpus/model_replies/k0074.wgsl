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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
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
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let ai = o + lid.y * 4u;
            let bi = o + lid.x * 4u;
            let a = vec4<f32>(sa[ai & 1023u], sa[(ai + 1u) & 1023u], sa[(ai + 2u) & 1023u], sa[(ai + 3u) & 1023u]);
            let b = vec4<f32>(sb[bi & 1023u], sb[(bi + 1u) & 1023u], sb[(bi + 2u) & 1023u], sb[(bi + 3u) & 1023u]);
            acc[0u] = acc[0u] + a.x * b;
            acc[1u] = acc[1u] + a.y * b;
            acc[2u] = acc[2u] + a.z * b;
            acc[3u] = acc[3u] + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    let c0 = colBase + lid.x * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let base = lr * d + c0;
            if (c0 + 3u < d) {
                h[base] = h[base] + v.x;
                h[base + 1u] = h[base + 1u] + v.y;
                h[base + 2u] = h[base + 2u] + v.z;
                h[base + 3u] = h[base + 3u] + v.w;
            } else {
                if (c0 < d) { h[base] = h[base] + v.x; }
                if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
                if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            }
        }
    }
}
