struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;         // m × d (item-major band)
@group(0) @binding(1) var<storage, read_write> h: array<f32>;     // r_rows × d output-row band
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lr = gid.y;              // local row within this output-row band
    let c = gid.x;              // global column
    if (lr >= cfg.r_rows || c >= cfg.d) { return; }
    let gr = cfg.r_off + lr;    // global feature row r
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        acc = acc + phi[base + gr] * phi[base + c];
    }
    let idx = lr * cfg.d + c;   // accumulator is band-local (r_rows × d)
    h[idx] = h[idx] + acc;
}
