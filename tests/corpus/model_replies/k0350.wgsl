// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var ra: array<f32, 64>;
    var ca: array<f32, 64>;
    var rb: array<f32, 64>;
    var cb: array<f32, 64>;
    var ri: array<u32, 16>;
    var ci: array<u32, 16>;
    let c0 = gid.x * 4u;
    let lr0 = gid.y * 8u;
    let dm1 = cfg.d - 1u;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        ci[j] = min(c0 + j, dm1);
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        ri[i] = min(cfg.r_off + lr0 + i, dm1);
    }
    let half = cfg.m / 2u;
    for (var p: u32 = 0u; p < half; p = p + 1u) {
        let base0 = (p * 2u) * cfg.d;
        let base1 = base0 + cfg.d;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            ca[j] = phi[base0 + ci[j]];
            cb[j] = phi[base1 + ci[j]];
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            ra[i] = phi[base0 + ri[i]];
            rb[i] = phi[base1 + ri[i]];
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + ra[i] * ca[j];
                pb[i * 4u + j] = pb[i * 4u + j] + rb[i] * cb[j];
            }
        }
    }
    let tail = cfg.m - half * 2u;
    for (var q: u32 = 0u; q < tail; q = q + 1u) {
        let base0 = (half * 2u + q) * cfg.d;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            ca[j] = phi[base0 + ci[j]];
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            ra[i] = phi[base0 + ri[i]];
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + ra[i] * ca[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = lr0 + i;
            let c = c0 + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + (pa[i * 4u + j] + pb[i * 4u + j]);
            }
        }
    }
}
