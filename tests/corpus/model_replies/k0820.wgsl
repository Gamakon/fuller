// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 32>;
    var pb: array<f32, 8>;
    var pc: array<f32, 4>;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let rbase = wid.y * 128u;
    let cbase = wid.x * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let e = lidx + 256u * q;
            let it = e >> 7u;
            let lr = rbase + (e & 127u);
            var v: f32 = 0.0;
            if (t + it < m && lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                v = phi[(t + it) * d + cfg.r_off + lr];
            }
            wg_a[e] = v;
        }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let it = e >> 6u;
            let c = cbase + (e & 63u);
            var v: f32 = 0.0;
            if (t + it < m && c < d) {
                v = phi[(t + it) * d + c];
            }
            wg_b[e] = v;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                pb[i] = wg_a[k * 128u + lid.y + 16u * i];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pc[j] = wg_b[k * 64u + lid.x + 16u * j];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + pb[i] * pc[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = rbase + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + pa[i * 4u + j];
                }
            }
        }
    }
}
