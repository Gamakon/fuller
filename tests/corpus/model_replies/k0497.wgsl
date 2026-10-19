// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (i = 0u; i < 8u; i = i + 1u) {
        pv[i] = vec4<f32>(0.0);
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = idx >> 6u;
            let cc = idx & 63u;
            var va: f32 = 0.0;
            if ((t + tr) < cfg.m && (wid.x * 64u + cc) < cfg.d) {
                va = phi[(t + tr) * cfg.d + wid.x * 64u + cc];
            }
            wg_a[idx] = va;
        }
        for (k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = idx >> 7u;
            let cc = idx & 127u;
            let lrw = wid.y * 128u + cc;
            var vb: f32 = 0.0;
            if ((t + tr) < cfg.m && lrw < cfg.r_rows && (cfg.r_off + lrw) < cfg.d) {
                vb = phi[(t + tr) * cfg.d + cfg.r_off + lrw];
            }
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (i = 0u; i < 8u; i = i + 1u) {
            pw[i] = vec4<f32>(0.0);
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            let ab = k * 64u + lid.x * 4u;
            let a = vec4<f32>(wg_a[ab], wg_a[ab + 1u], wg_a[ab + 2u], wg_a[ab + 3u]);
            for (i = 0u; i < 8u; i = i + 1u) {
                let b = wg_b[k * 128u + lid.y + 16u * i];
                pw[i] = pw[i] + vec4<f32>(b) * a;
            }
        }
        for (i = 0u; i < 8u; i = i + 1u) {
            pv[i] = pv[i] + pw[i];
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 128u + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 64u + lid.x * 4u + j;
                if (col < cfg.d) {
                    h[lr * cfg.d + col] = h[lr * cfg.d + col] + pv[i][j];
                }
            }
        }
    }
}
