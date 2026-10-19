// grid 32 8 2 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    let row0 = wid.y * 32u;
    let col0 = wid.x * 64u;
    let a_item = lidx / 32u;
    let a_col = lidx % 32u;
    let a_lr = row0 + a_col;
    let a_ok = a_lr < cfg.r_rows && (cfg.r_off + a_lr) < cfg.d;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        var fa: f32 = 0.0;
        if (t + a_item < cfg.m && a_ok) {
            fa = phi[(t + a_item) * cfg.d + cfg.r_off + a_lr];
        }
        wg_a[lidx] = fa;
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let it = idx / 64u;
            let cc = col0 + (idx % 64u);
            var fb: f32 = 0.0;
            if (t + it < cfg.m && cc < cfg.d) {
                fb = phi[(t + it) * cfg.d + cc];
            }
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 2u) {
            for (var s: u32 = 0u; s < 2u; s = s + 1u) {
                let kk = k + s;
                var bv: array<f32, 2>;
                bv[0] = wg_b[kk * 64u + lid.x];
                bv[1] = wg_b[kk * 64u + lid.x + 32u];
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let av = wg_a[kk * 32u + lid.y + 8u * i];
                    acc[s * 8u + i * 2u + 0u] = acc[s * 8u + i * 2u + 0u] + av * bv[0];
                    acc[s * 8u + i * 2u + 1u] = acc[s * 8u + i * 2u + 1u] + av * bv[1];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 8u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = col0 + lid.x + 32u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + (acc[i * 2u + j] + acc[8u + i * 2u + j]);
                }
            }
        }
    }
}
