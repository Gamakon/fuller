// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lx = lid.x;
    let ly = lid.y;
    let c = wid.x * 16u + lx;
    let lr = wid.y * 16u + ly;
    let gr = cfg.r_off + lr;
    var acc: f32 = 0.0;
    let ntiles = (cfg.m + 63u) / 64u;
    for (var tb: u32 = 0u; tb < ntiles; tb = tb + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + q * 256u;
            let ti = e / 16u;
            let j = e % 16u;
            let t = tb * 64u + ti;
            let lcol = wid.x * 16u + j;
            let llr = wid.y * 16u + j;
            let lgr = cfg.r_off + llr;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (llr < cfg.r_rows) {
                    if (lgr < cfg.d) {
                        va = phi[t * cfg.d + lgr];
                    }
                }
                if (lcol < cfg.d) {
                    vb = phi[t * cfg.d + lcol];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        for (var k: u32 = 0u; k < 64u; k = k + 4u) {
            s0 = s0 + wg_a[k * 16u + ly] * wg_b[k * 16u + lx];
            s1 = s1 + wg_a[(k + 1u) * 16u + ly] * wg_b[(k + 1u) * 16u + lx];
            s2 = s2 + wg_a[(k + 2u) * 16u + ly] * wg_b[(k + 2u) * 16u + lx];
            s3 = s3 + wg_a[(k + 3u) * 16u + ly] * wg_b[(k + 3u) * 16u + lx];
        }
        acc = acc + ((s0 + s1) + (s2 + s3));
        workgroupBarrier();
    }
    if (lr < cfg.r_rows) {
        if (c < cfg.d) {
            if (gr < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc;
            }
        }
    }
}
