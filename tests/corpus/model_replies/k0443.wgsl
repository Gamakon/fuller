// grid 8 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    let d = cfg.d;
    for (k = 0u; k < 64u; k = k + 1u) {
        pc[k] = 0.0;
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let lr = wid.y * 64u + k * 8u + lid.y;
        pi[k] = lr;
        pm[k] = (lr < cfg.r_rows) && ((cfg.r_off + lr) < d);
        pj[k] = cfg.r_off + lr;
        let c = wid.x * 64u + k * 8u + lid.x;
        pn[k] = c < d;
        pi[k + 8u] = c;
    }
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * d;
        for (k = 0u; k < 8u; k = k + 1u) {
            var a: f32 = 0.0;
            if (pm[k]) {
                a = phi[base + pj[k]];
            }
            pa[k] = a;
            var b: f32 = 0.0;
            if (pn[k]) {
                b = phi[base + pi[k + 8u]];
            }
            pb[k] = b;
        }
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                pc[k * 8u + k2] = pc[k * 8u + k2] + pa[k] * pb[k2];
            }
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
            if (pm[k] && pn[k2]) {
                let idx = pi[k] * d + pi[k2 + 8u];
                h[idx] = h[idx] + pc[k * 8u + k2];
            }
        }
    }
}
