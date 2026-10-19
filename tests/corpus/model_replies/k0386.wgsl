// grid 16 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let r0 = gid.y * 4u;
    let c0 = gid.x * 4u;
    let dm1 = cfg.d - 1u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        pi[i] = min(cfg.r_off + r0 + i, dm1);
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        pj[j] = min(c0 + j, dm1);
    }
    for (var s: u32 = 0u; s < cfg.m; s = s + 1u) {
        let b0 = s * cfg.d;
        let cv0 = vec4<f32>(phi[b0 + pj[0]], phi[b0 + pj[1]], phi[b0 + pj[2]], phi[b0 + pj[3]]);
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let a0 = phi[b0 + pi[i]];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + a0 * cv0[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
