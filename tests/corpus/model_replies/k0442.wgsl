// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
        pa[j] = 0.0;
        pb[j] = 0.0;
        pc[j] = 0.0;
    }
    for (var t: u32 = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let id = lidx + k * 128u;
            let it = t * 32u + id / 64u;
            let cc = wid.x * 64u + id % 64u;
            let rl = wid.y * 64u + id % 64u;
            var f0: f32 = 0.0;
            if (it < cfg.m && cc < cfg.d) {
                f0 = phi[it * cfg.d + cc];
            }
            wg_a[id] = f0;
            var f1: f32 = 0.0;
            if (it < cfg.m && rl < cfg.r_rows && (cfg.r_off + rl) < cfg.d) {
                f1 = phi[it * cfg.d + cfg.r_off + rl];
            }
            wg_b[id] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            for (var k2: u32 = 0u; k2 < 8u; k2 = k2 + 1u) {
                let bv0 = wg_b[k * 64u + lid.y * 8u + k2];
                let bv1 = wg_b[k * 64u + 64u + lid.y * 8u + k2];
                for (var k3: u32 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    let q = k2 * 4u + k3;
                    pa[q] = pa[q] + bv0 * wg_a[k * 64u + lid.x * 4u + k3];
                    pb[q] = pb[q] + bv1 * wg_a[k * 64u + 64u + lid.x * 4u + k3];
                }
            }
        }
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            pc[j] = pc[j] + (pa[j] + pb[j]);
            pa[j] = 0.0;
            pb[j] = 0.0;
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let row = gid.y * 8u + k;
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let col = gid.x * 4u + k2;
            if (row < cfg.r_rows && col < cfg.d && (cfg.r_off + row) < cfg.d) {
                h[row * cfg.d + col] = h[row * cfg.d + col] + pc[k * 4u + k2];
            }
        }
    }
}
