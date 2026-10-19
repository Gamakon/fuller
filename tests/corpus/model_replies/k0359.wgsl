// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c0 = gid.x * 4u;
    let r0 = gid.y * 8u;
    let dm1 = cfg.d - 1u;
    var acc: array<f32, 64>;
    var part: array<f32, 64>;
    var ra: array<f32, 64>;
    var cb: array<f32, 64>;
    var ri: array<u32, 16>;
    var ci: array<u32, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        ri[k] = min(cfg.r_off + r0 + k, dm1);
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        ci[j] = min(c0 + j, dm1);
    }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        acc[k] = 0.0;
    }
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 32u) {
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            part[k] = 0.0;
        }
        for (var tt: u32 = 0u; tt < 32u; tt = tt + 1u) {
            let t = t0 + tt;
            if (t < cfg.m) {
                let base = t * cfg.d;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    ra[i] = phi[base + ri[i]];
                }
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    cb[j] = phi[base + ci[j]];
                }
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        part[i * 4u + j] = part[i * 4u + j] + ra[i] * cb[j];
                    }
                }
            }
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc[k] = acc[k] + part[k];
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = r0 + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[i * 4u + j];
            }
        }
    }
}
