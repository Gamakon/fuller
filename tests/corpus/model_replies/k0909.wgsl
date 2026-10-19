// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let col0 = wid.x * 16u;
    let row0 = wid.y * 16u;
    let c = col0 + lid.x;
    let lrl = row0 + lid.x;
    let grl = cfg.r_off + lrl;
    let slot = lid.y * 16u + lid.x;
    var acc: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        let t = t0 + lid.y;
        var a: f32 = 0.0;
        var b: f32 = 0.0;
        if (t < cfg.m) {
            if (c < d) { b = phi[t * d + c]; }
            if (lrl < cfg.r_rows && grl < d) { a = phi[t * d + grl]; }
        }
        wg_a[slot & 255u] = a;
        wg_b[slot & 255u] = b;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + wg_a[(k * 16u + lid.y) & 255u] * wg_b[(k * 16u + lid.x) & 255u];
        }
        workgroupBarrier();
    }
    let lr = row0 + lid.y;
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
