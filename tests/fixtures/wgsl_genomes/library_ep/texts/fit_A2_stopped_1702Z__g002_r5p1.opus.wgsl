// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let lr = wid.y * 16u + ly;
    let c = wid.x * 16u + lx;
    let load_lr = wid.y * 16u + lx;
    let load_gr = cfg.r_off + load_lr;
    let row_ok = load_lr < cfg.r_rows && load_gr < cfg.d;
    let col_ok = c < cfg.d;
    var acc: f32 = 0.0;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        let t = tt * 16u + ly;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < cfg.m) {
            let base = t * cfg.d;
            if (row_ok) { va = phi[base + load_gr]; }
            if (col_ok) { vb = phi[base + c]; }
        }
        wg_a[ly * 16u + lx] = va;
        wg_b[ly * 16u + lx] = vb;
        workgroupBarrier();
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        for (var k: u32 = 0u; k < 16u; k = k + 4u) {
            s0 = s0 + wg_a[k * 16u + ly] * wg_b[k * 16u + lx];
            s1 = s1 + wg_a[(k + 1u) * 16u + ly] * wg_b[(k + 1u) * 16u + lx];
            s2 = s2 + wg_a[(k + 2u) * 16u + ly] * wg_b[(k + 2u) * 16u + lx];
            s3 = s3 + wg_a[(k + 3u) * 16u + ly] * wg_b[(k + 3u) * 16u + lx];
        }
        acc = acc + ((s0 + s1) + (s2 + s3));
        workgroupBarrier();
    }
    if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
        let idx = lr * cfg.d + c;
        h[idx] = h[idx] + acc;
    }
}
