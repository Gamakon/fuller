// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 8u;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        pm[k] = ((r0 + k) < cfg.r_rows) && ((cfg.r_off + r0 + k) < cfg.d);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        pn[k] = (c0 + k) < cfg.d;
    }
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (k = 0u; k < 8u; k = k + 1u) {
            var rv: f32 = 0.0;
            if (pm[k]) {
                rv = phi[base + cfg.r_off + r0 + k];
            }
            pb[k] = rv;
        }
        for (k = 0u; k < 4u; k = k + 1u) {
            var cv: f32 = 0.0;
            if (pn[k]) {
                cv = phi[base + c0 + k];
            }
            pc[k] = cv;
        }
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                pa[k * 4u + k2] = pa[k * 4u + k2] + pb[k] * pc[k2];
            }
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if (pm[k] && pn[k2]) {
                let idx = (r0 + k) * cfg.d + c0 + k2;
                h[idx] = h[idx] + pa[k * 4u + k2];
            }
        }
    }
}
