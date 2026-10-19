// grid 32 8 2 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }
    let rowbase = wid.y * 32u;
    let colbase = wid.x * 64u;
    let arow = rowbase + (lidx % 32u);
    let aok = arow < cfg.r_rows && (cfg.r_off + arow) < cfg.d;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let item = t + q * 8u + (lidx / 32u);
            var v: f32 = 0.0;
            if (item < cfg.m && aok) {
                v = phi[item * cfg.d + cfg.r_off + arow];
            }
            wg_a[q * 256u + lidx] = v;
        }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let item = t + (idx / 64u);
            let col = colbase + (idx % 64u);
            var v: f32 = 0.0;
            if (item < cfg.m && col < cfg.d) {
                v = phi[item * cfg.d + col];
            }
            wg_b[idx] = v;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let p = (k & 1u) * 8u;
            let b0 = wg_b[k * 64u + lid.x];
            let b1 = wg_b[k * 64u + lid.x + 32u];
            for (var k3: u32 = 0u; k3 < 4u; k3 = k3 + 1u) {
                let a = wg_a[k * 32u + lid.y + 8u * k3];
                acc[p + k3 * 2u] = acc[p + k3 * 2u] + a * b0;
                acc[p + k3 * 2u + 1u] = acc[p + k3 * 2u + 1u] + a * b1;
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = rowbase + lid.y + 8u * k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
                let c = colbase + lid.x + 32u * k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + (acc[k * 2u + k2] + acc[8u + k * 2u + k2]);
                }
            }
        }
    }
}
