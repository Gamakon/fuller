// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pc: array<f32, 16>;
    var pb: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let nsuper = (cfg.m + 255u) / 256u;
    for (var s: u32 = 0u; s < nsuper; s = s + 1u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pc[k] = 0.0;
        }
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let t = s * 256u + j * 16u;
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                let li = lidx + k * 256u;
                let it = t + li / 64u;
                let cl = li % 64u;
                var f0: f32 = 0.0;
                var f1: f32 = 0.0;
                if (it < cfg.m) {
                    if ((wid.y * 64u + cl) < cfg.r_rows && (cfg.r_off + wid.y * 64u + cl) < cfg.d) {
                        f0 = phi[it * cfg.d + cfg.r_off + wid.y * 64u + cl];
                    }
                    if ((wid.x * 64u + cl) < cfg.d) {
                        f1 = phi[it * cfg.d + wid.x * 64u + cl];
                    }
                }
                wg_a[li] = f0;
                wg_b[li] = f1;
            }
            workgroupBarrier();
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                pb[k] = 0.0;
            }
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let ra = q * 64u + lid.y * 4u;
                let rb = q * 64u + lid.x * 4u;
                for (var k3: u32 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    let av = wg_a[ra + k3];
                    for (var k4: u32 = 0u; k4 < 4u; k4 = k4 + 1u) {
                        pb[k3 * 4u + k4] = pb[k3 * 4u + k4] + av * wg_b[rb + k4];
                    }
                }
            }
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                pc[k] = pc[k] + pb[k];
            }
            workgroupBarrier();
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pc[k];
        }
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let rr = gid.y * 4u + k;
            let cc = gid.x * 4u + k2;
            if (rr < cfg.r_rows && cc < cfg.d && (cfg.r_off + rr) < cfg.d) {
                h[rr * cfg.d + cc] = h[rr * cfg.d + cc] + pa[k * 4u + k2];
            }
        }
    }
}
