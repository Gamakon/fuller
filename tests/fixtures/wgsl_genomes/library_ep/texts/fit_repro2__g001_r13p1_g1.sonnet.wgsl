// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    var av: array<f32, 4>;
    var bv: array<f32, 4>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    var idx: u32 = 0u;
    var tt: u32 = 0u;
    var cc: u32 = 0u;
    var item: u32 = 0u;
    var col: u32 = 0u;
    var lrw: u32 = 0u;
    var va: f32 = 0.0;
    var vb: f32 = 0.0;
    for (k = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            idx = lidx + 256u * k2;
            tt = idx / 64u;
            cc = idx % 64u;
            item = t + tt;
            col = wid.x * 64u + cc;
            lrw = wid.y * 64u + cc;
            va = 0.0;
            vb = 0.0;
            if (item < cfg.m) {
                if (col < cfg.d) {
                    va = phi[item * cfg.d + col];
                }
                if (lrw < cfg.r_rows) {
                    if (cfg.r_off + lrw < cfg.d) {
                        vb = phi[item * cfg.d + cfg.r_off + lrw];
                    }
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            for (j = 0u; j < 4u; j = j + 1u) {
                av[j] = wg_a[k * 64u + lid.x * 4u + j];
                bv[j] = wg_b[k * 64u + lid.y * 4u + j];
            }
            for (i = 0u; i < 4u; i = i + 1u) {
                for (j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + bv[i] * av[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        lrw = wid.y * 64u + lid.y * 4u + i;
        for (j = 0u; j < 4u; j = j + 1u) {
            col = wid.x * 64u + lid.x * 4u + j;
            if (lrw < cfg.r_rows) {
                if (col < cfg.d) {
                    if (cfg.r_off + lrw < cfg.d) {
                        h[lrw * cfg.d + col] = h[lrw * cfg.d + col] + acc[i * 4u + j];
                    }
                }
            }
        }
    }
}
