// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var p: array<f32, 16>;
    var q: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        p[i] = 0.0;
        q[i] = 0.0;
    }
    let nt = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < nt; t = t + 1u) {
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let e = lidx + l * 256u;
            let kk = e / 64u;
            let col = e % 64u;
            let item = t * 16u + kk;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (item < cfg.m) {
                let lrow = wid.y * 64u + col;
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < cfg.d) {
                    fa = phi[item * cfg.d + cfg.r_off + lrow];
                }
                let c = wid.x * 64u + col;
                if (c < cfg.d) {
                    fb = phi[item * cfg.d + c];
                }
            }
            wg_a[e] = fa;
            wg_b[e] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 2u) {
            let o0 = k * 64u;
            let o1 = o0 + 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a0 = wg_a[o0 + lid.y + 16u * i];
                let a1 = wg_a[o1 + lid.y + 16u * i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    p[i * 4u + j] = p[i * 4u + j] + a0 * wg_b[o0 + lid.x + 16u * j];
                    q[i * 4u + j] = q[i * 4u + j] + a1 * wg_b[o1 + lid.x + 16u * j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = wid.x * 64u + lid.x + 16u * j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + (p[i * 4u + j] + q[i * 4u + j]);
            }
        }
    }
}
