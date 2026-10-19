// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let kk = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    va = phi[t * d + gc];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[t * d + gr];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let base = kk * 64u;
            let a = vec4<f32>(
                sa[(base + lid.x) & 1023u],
                sa[(base + 16u + lid.x) & 1023u],
                sa[(base + 32u + lid.x) & 1023u],
                sa[(base + 48u + lid.x) & 1023u]);
            let b0 = sb[(base + lid.y) & 1023u];
            let b1 = sb[(base + 16u + lid.y) & 1023u];
            let b2 = sb[(base + 32u + lid.y) & 1023u];
            let b3 = sb[(base + 48u + lid.y) & 1023u];
            acc0 = acc0 + vec4<f32>(b0) * a;
            acc1 = acc1 + vec4<f32>(b1) * a;
            acc2 = acc2 + vec4<f32>(b2) * a;
            acc3 = acc3 + vec4<f32>(b3) * a;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i * 16u + lid.y;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var av = acc0;
            if (i == 1u) { av = acc1; }
            if (i == 2u) { av = acc2; }
            if (i == 3u) { av = acc3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + j * 16u + lid.x;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + av[j];
                }
            }
        }
    }
}
