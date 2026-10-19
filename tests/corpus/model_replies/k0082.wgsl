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
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    let tid = lid.y * 16u + lid.x;

    var acc: array<vec4<f32>, 4>;
    for (var q = 0u; q < 4u; q = q + 1u) {
        acc[q & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = tid + k * 256u;
            let tt = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + tt;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let c = cbase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = rbase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();

        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            let ao = o + lid.y * 4u;
            let bo = o + lid.x * 4u;
            let b = vec4<f32>(sb[bo & 1023u], sb[(bo + 1u) & 1023u],
                              sb[(bo + 2u) & 1023u], sb[(bo + 3u) & 1023u]);
            let a = vec4<f32>(sa[ao & 1023u], sa[(ao + 1u) & 1023u],
                              sa[(ao + 2u) & 1023u], sa[(ao + 3u) & 1023u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = cbase + lid.x * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
