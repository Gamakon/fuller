// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let c = wid.x * 16u + lid.x;
    let lr = wid.y * 16u + lid.y;
    let ldc = wid.x * 16u + lid.x;
    let ldr = wid.y * 16u + lid.x;
    let ldgr = cfg.r_off + ldr;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        let item = tt * 16u + lid.y;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (item < cfg.m) {
            let base = item * cfg.d;
            if (ldc < cfg.d) {
                vb = phi[base + ldc];
            }
            if (ldr < cfg.r_rows && ldgr < cfg.d) {
                va = phi[base + ldgr];
            }
        }
        wg_a[lid.y * 16u + lid.x] = va;
        wg_b[lid.y * 16u + lid.x] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 4u) {
            acc0 = acc0 + wg_a[k * 16u + lid.y] * wg_b[k * 16u + lid.x];
            acc1 = acc1 + wg_a[(k + 1u) * 16u + lid.y] * wg_b[(k + 1u) * 16u + lid.x];
            acc2 = acc2 + wg_a[(k + 2u) * 16u + lid.y] * wg_b[(k + 2u) * 16u + lid.x];
            acc3 = acc3 + wg_a[(k + 3u) * 16u + lid.y] * wg_b[(k + 3u) * 16u + lid.x];
        }
        workgroupBarrier();
    }
    let gr = cfg.r_off + lr;
    if (lr < cfg.r_rows && c < cfg.d && gr < cfg.d) {
        let idx = lr * cfg.d + c;
        h[idx] = h[idx] + ((acc0 + acc1) + (acc2 + acc3));
    }
}
