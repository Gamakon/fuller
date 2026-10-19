// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let r0 = gid.y * 8u;
    let c0 = gid.x * 4u;
    let dm = d - 1u;
    let cj0 = min(c0, dm);
    let cj1 = min(c0 + 1u, dm);
    let cj2 = min(c0 + 2u, dm);
    let cj3 = min(c0 + 3u, dm);
    var ri: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        ri[k] = min(cfg.r_off + r0 + k, dm);
    }
    var pa: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pa[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        var pb: array<vec4<f32>, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pb[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            if (t + k < m) {
                let base = (t + k) * d;
                let cv = vec4<f32>(phi[base + cj0], phi[base + cj1], phi[base + cj2], phi[base + cj3]);
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    pb[i] = pb[i] + phi[base + ri[i]] * cv;
                }
            }
        }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + pa[i][j];
                }
            }
        }
    }
}
