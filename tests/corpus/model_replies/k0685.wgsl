// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 16>;
    var tmp: array<vec4<f32>, 16>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let ry = lid.y * 8u;
    let cx = lid.x * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let e = lidx + q * 128u;
            let kk = e / 64u;
            let x = e % 64u;
            let item = t + kk;
            let lr = row0 + x;
            let gr = cfg.r_off + lr;
            var va: f32 = 0.0;
            if (item < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                va = phi[item * cfg.d + gr];
            }
            wg_a[e] = va;
            let c = col0 + x;
            var vb: f32 = 0.0;
            if (item < cfg.m && c < cfg.d) {
                vb = phi[item * cfg.d + c];
            }
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            tmp[i] = vec4<f32>(0.0);
        }
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let bb = kk * 64u + cx;
            let b = vec4<f32>(wg_b[bb], wg_b[bb + 1u], wg_b[bb + 2u], wg_b[bb + 3u]);
            let ab = kk * 64u + ry;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                tmp[i] = tmp[i] + wg_a[ab + i] * b;
            }
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            acc[i] = acc[i] + tmp[i];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + ry + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + cx + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[i][j];
                }
            }
        }
    }
}
