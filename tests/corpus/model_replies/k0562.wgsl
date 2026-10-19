// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var tot0 = vec4<f32>(0.0);
    var tot1 = vec4<f32>(0.0);
    var tot2 = vec4<f32>(0.0);
    var tot3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var e: u32 = 0u; e < 8u; e = e + 1u) {
            let idx = lidx + 256u * e;
            let kk = idx >> 6u;
            let cc = idx & 63u;
            let item = t + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                let base = item * cfg.d;
                let col = wid.x * 64u + cc;
                if (col < cfg.d) {
                    va = phi[base + col];
                }
                let lr = wid.y * 64u + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    vb = phi[base + cfg.r_off + lr];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(wg_a[o + lid.x], wg_a[o + lid.x + 16u], wg_a[o + lid.x + 32u], wg_a[o + lid.x + 48u]);
            let b0 = wg_b[o + lid.y];
            let b1 = wg_b[o + lid.y + 16u];
            let b2 = wg_b[o + lid.y + 32u];
            let b3 = wg_b[o + lid.y + 48u];
            v0 = v0 + vec4<f32>(b0) * a;
            v1 = v1 + vec4<f32>(b1) * a;
            v2 = v2 + vec4<f32>(b2) * a;
            v3 = v3 + vec4<f32>(b3) * a;
        }
        tot0 = tot0 + v0;
        tot1 = tot1 + v1;
        tot2 = tot2 + v2;
        tot3 = tot3 + v3;
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0] = tot0;
    pv[1] = tot1;
    pv[2] = tot2;
    pv[3] = tot3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 16u * j;
                if (c < cfg.d) {
                    let ix = lr * cfg.d + c;
                    h[ix] = h[ix] + pv[i][j];
                }
            }
        }
    }
}
