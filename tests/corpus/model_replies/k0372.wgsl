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
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    var a3: f32 = 0.0;
    let q4 = cfg.m / 4u;
    for (var q: u32 = 0u; q < q4; q = q + 1u) {
        let b0 = (q * 4u) * cfg.d;
        let b1 = b0 + cfg.d;
        let b2 = b1 + cfg.d;
        let b3 = b2 + cfg.d;
        a0 = a0 + phi[b0 + gr] * phi[b0 + c];
        a1 = a1 + phi[b1 + gr] * phi[b1 + c];
        a2 = a2 + phi[b2 + gr] * phi[b2 + c];
        a3 = a3 + phi[b3 + gr] * phi[b3 + c];
    }
    let rem = cfg.m - q4 * 4u;
    for (var j: u32 = 0u; j < rem; j = j + 1u) {
        let b = (q4 * 4u + j) * cfg.d;
        a0 = a0 + phi[b + gr] * phi[b + c];
    }
    let idx = lr * cfg.d + c;
    h[idx] = h[idx] + ((a0 + a1) + (a2 + a3));
}
