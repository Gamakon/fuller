// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lr = gid.y;
    let c = gid.x;
    if (lr >= cfg.r_rows || c >= cfg.d || (cfg.r_off + lr) >= cfg.d) { return; }
    let gr = cfg.r_off + lr;
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    let m4 = cfg.m / 4u;
    for (var q: u32 = 0u; q < m4; q = q + 1u) {
        let t = q * 4u;
        let b0 = t * cfg.d;
        let b1 = b0 + cfg.d;
        let b2 = b1 + cfg.d;
        let b3 = b2 + cfg.d;
        f0 = f0 + phi[b0 + gr] * phi[b0 + c];
        f1 = f1 + phi[b1 + gr] * phi[b1 + c];
        f2 = f2 + phi[b2 + gr] * phi[b2 + c];
        f3 = f3 + phi[b3 + gr] * phi[b3 + c];
    }
    let rem = cfg.m - m4 * 4u;
    for (var k: u32 = 0u; k < rem; k = k + 1u) {
        let base = (m4 * 4u + k) * cfg.d;
        f0 = f0 + phi[base + gr] * phi[base + c];
    }
    let acc = (f0 + f1) + (f2 + f3);
    let idx = lr * cfg.d + c;
    h[idx] = h[idx] + acc;
}
