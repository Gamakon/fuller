// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }
    let ntiles = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let idx = lidx + 128u * q;
            let kk = idx / 64u;
            let cc = idx % 64u;
            let item = t * 16u + kk;
            let g = wid.y * 64u + cc;
            let c = wid.x * 64u + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                if (g < cfg.r_rows && (cfg.r_off + g) < cfg.d) {
                    va = phi[item * cfg.d + cfg.r_off + g];
                }
                if (c < cfg.d) {
                    vb = phi[item * cfg.d + c];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var p: array<vec4<f32>, 8>;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            p[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let bb = k * 64u + lid.x;
            let bv = vec4<f32>(wg_b[bb], wg_b[bb + 16u], wg_b[bb + 32u], wg_b[bb + 48u]);
            let ab = k * 64u + lid.y;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                p[i] = p[i] + vec4<f32>(wg_a[ab + 8u * i]) * bv;
            }
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            acc[i] = acc[i] + p[i];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 8u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 64u + lid.x + 16u * j;
                if (col < cfg.d) {
                    let o = lr * cfg.d + col;
                    h[o] = h[o] + acc[i][j];
                }
            }
        }
    }
}
