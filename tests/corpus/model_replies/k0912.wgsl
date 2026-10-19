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
    var a00 = vec4<f32>(0.0);
    var a01 = vec4<f32>(0.0);
    var a02 = vec4<f32>(0.0);
    var a03 = vec4<f32>(0.0);
    let lx4 = lid.x * 4u;
    let ly4 = lid.y * 4u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let kk = idx >> 6u;
            let j = idx & 63u;
            let item = t + kk;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (item < m) {
                let base = item * d;
                if (col0 + j < d) {
                    vb = phi[base + col0 + j];
                }
                if (row0 + j < cfg.r_rows && cfg.r_off + row0 + j < d) {
                    va = phi[base + cfg.r_off + row0 + j];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let ba = kk * 64u + ly4;
            let bb = kk * 64u + lx4;
            let b = vec4<f32>(sb[bb & 1023u], sb[(bb + 1u) & 1023u], sb[(bb + 2u) & 1023u], sb[(bb + 3u) & 1023u]);
            let r0 = sa[ba & 1023u];
            let r1 = sa[(ba + 1u) & 1023u];
            let r2 = sa[(ba + 2u) & 1023u];
            let r3 = sa[(ba + 3u) & 1023u];
            a00 = fma(vec4<f32>(r0), b, a00);
            a01 = fma(vec4<f32>(r1), b, a01);
            a02 = fma(vec4<f32>(r2), b, a02);
            a03 = fma(vec4<f32>(r3), b, a03);
        }
        workgroupBarrier();
    }
    let cb = col0 + lx4;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = row0 + ly4 + r;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a00;
            if (r == 1u) { v = a01; }
            if (r == 2u) { v = a02; }
            if (r == 3u) { v = a03; }
            let ro = lr * d;
            if (cb < d) { h[ro + cb] = h[ro + cb] + v.x; }
            if (cb + 1u < d) { h[ro + cb + 1u] = h[ro + cb + 1u] + v.y; }
            if (cb + 2u < d) { h[ro + cb + 2u] = h[ro + cb + 2u] + v.z; }
            if (cb + 3u < d) { h[ro + cb + 3u] = h[ro + cb + 3u] + v.w; }
        }
    }
}
