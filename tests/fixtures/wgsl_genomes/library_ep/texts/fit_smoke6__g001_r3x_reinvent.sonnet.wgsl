// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let kk = e >> 6u;
            let c = e & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + kk < cfg.m) {
                let base = (t + kk) * cfg.d;
                let lrw = rbase + c;
                if (lrw < cfg.r_rows && cfg.r_off + lrw < cfg.d) {
                    fa = phi[base + cfg.r_off + lrw];
                }
                if (cbase + c < cfg.d) {
                    fb = phi[base + cbase + c];
                }
            }
            sa[e] = fa;
            sb[e] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ao = k * 64u + lid.y * 4u;
            let bo = k * 64u + lid.x * 4u;
            let b = vec4<f32>(sb[bo], sb[bo + 1u], sb[bo + 2u], sb[bo + 3u]);
            acc0 = acc0 + sa[ao] * b;
            acc1 = acc1 + sa[ao + 1u] * b;
            acc2 = acc2 + sa[ao + 2u] * b;
            acc3 = acc3 + sa[ao + 3u] * b;
        }
        workgroupBarrier();
    }
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let row = rbase + lid.y * 4u + i;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = cbase + lid.x * 4u + j;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + accs[i][j];
                }
            }
        }
    }
}
