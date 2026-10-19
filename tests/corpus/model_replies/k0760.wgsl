// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 4>;
    var pc: array<f32, 4>;
    for (var z: u32 = 0u; z < 64u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let ntiles = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = idx / 64u;
            let col = idx % 64u;
            let ti = t * 16u + item;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (ti < cfg.m) {
                let row = wid.y * 64u + col;
                if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
                    f0 = phi[ti * cfg.d + cfg.r_off + row];
                }
                let c = wid.x * 64u + col;
                if (c < cfg.d) {
                    f1 = phi[ti * cfg.d + c];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 4u; kk = kk + 1u) {
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let item = kk * 4u + q;
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    pb[i] = wg_a[item * 64u + lid.y * 4u + i];
                    pc[i] = wg_b[item * 64u + lid.x * 4u + i];
                }
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        let p = (i * 4u + j) * 4u + q;
                        pa[p] = pa[p] + pb[i] * pc[j];
                    }
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let row = wid.y * 64u + lid.y * 4u + i;
            let col = wid.x * 64u + lid.x * 4u + j;
            if (row < cfg.r_rows && col < cfg.d && (cfg.r_off + row) < cfg.d) {
                let b = (i * 4u + j) * 4u;
                let s = (pa[b] + pa[b + 1u]) + (pa[b + 2u] + pa[b + 3u]);
                h[row * cfg.d + col] = h[row * cfg.d + col] + s;
            }
        }
    }
}
