// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { pa[k] = 0.0; }
    let tid = lid.y * 16u + lid.x;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = tid + k * 256u;
            let item = t + (idx >> 6u);
            let col = idx & 63u;
            let lrow = wid.y * 64u + col;
            let gcol = wid.x * 64u + col;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < cfg.m) {
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < cfg.d) {
                    f0 = phi[item * cfg.d + cfg.r_off + lrow];
                }
                if (gcol < cfg.d) {
                    f1 = phi[item * cfg.d + gcol];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                ra[q] = wg_a[k * 64u + lid.y + 16u * q];
                rb[q] = wg_b[k * 64u + lid.x + 16u * q];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + ra[i] * rb[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + i * 16u;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + j * 16u;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + pa[i * 4u + j];
                }
            }
        }
    }
}
