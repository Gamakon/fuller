// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    let d = cfg.d;
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let tr = t + (idx >> 6u);
            let c = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (tr < cfg.m) {
                if (rowBase + c < cfg.r_rows && cfg.r_off + rowBase + c < d) {
                    f0 = phi[tr * d + cfg.r_off + rowBase + c];
                }
                if (colBase + c < d) {
                    f1 = phi[tr * d + colBase + c];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a0 = wg_a[o + lid.y];
            let a1 = wg_a[o + lid.y + 16u];
            let a2 = wg_a[o + lid.y + 32u];
            let a3 = wg_a[o + lid.y + 48u];
            let b = vec4<f32>(wg_b[o + lid.x], wg_b[o + lid.x + 16u], wg_b[o + lid.x + 32u], wg_b[o + lid.x + 48u]);
            q0 = q0 + a0 * b;
            q1 = q1 + a1 * b;
            q2 = q2 + a2 * b;
            q3 = q3 + a3 * b;
        }
        s0 = s0 + q0; s1 = s1 + q1; s2 = s2 + q2; s3 = s3 + q3;
        workgroupBarrier();
    }
    var pa: array<vec4<f32>, 4>;
    pa[0] = s0; pa[1] = s1; pa[2] = s2; pa[3] = s3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = rowBase + lid.y + k * 16u;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = pa[k];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = colBase + lid.x + k2 * 16u;
                if (c < d) {
                    h[lr * d + c] = h[lr * d + c] + v[k2];
                }
            }
        }
    }
}
