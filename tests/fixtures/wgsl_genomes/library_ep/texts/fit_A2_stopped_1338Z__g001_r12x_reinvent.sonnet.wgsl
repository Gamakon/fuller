// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pc: array<f32, 64>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        pc[z] = 0.0;
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let kk = idx / 64u;
            let cc = idx % 64u;
            var ta: f32 = 0.0;
            var tb: f32 = 0.0;
            let item = t + kk;
            if (item < cfg.m) {
                let r = row0 + cc;
                if (r < cfg.r_rows && (cfg.r_off + r) < cfg.d) {
                    ta = phi[item * cfg.d + cfg.r_off + r];
                }
                let c = col0 + cc;
                if (c < cfg.d) {
                    tb = phi[item * cfg.d + c];
                }
            }
            wg_a[idx] = ta;
            wg_b[idx] = tb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 4u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let ai = lid.y + 16u * i;
                let a0 = wg_a[k * 64u + ai];
                let a1 = wg_a[(k + 1u) * 64u + ai];
                let a2 = wg_a[(k + 2u) * 64u + ai];
                let a3 = wg_a[(k + 3u) * 64u + ai];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let bj = lid.x + 16u * j;
                    let s = (a0 * wg_b[k * 64u + bj] + a1 * wg_b[(k + 1u) * 64u + bj]) + (a2 * wg_b[(k + 2u) * 64u + bj] + a3 * wg_b[(k + 3u) * 64u + bj]);
                    pc[i * 4u + j] = pc[i * 4u + j] + s;
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let r = row0 + lid.y + 16u * i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + lid.x + 16u * j;
            if (r < cfg.r_rows && (cfg.r_off + r) < cfg.d && c < cfg.d) {
                h[r * cfg.d + c] = h[r * cfg.d + c] + pc[i * 4u + j];
            }
        }
    }
}
