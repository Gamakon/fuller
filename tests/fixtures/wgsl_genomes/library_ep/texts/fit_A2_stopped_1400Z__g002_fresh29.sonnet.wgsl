// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = vec4<f32>(0.0); }
    for (var t = 0u; t < m; t = t + 16u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let it = t + (e >> 6u);
            let cc = e & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < m) {
                let lr = rbase + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    va = phi[it * d + cfg.r_off + lr];
                }
                let col = cbase + cc;
                if (col < d) {
                    vb = phi[it * d + col];
                }
            }
            sa[e] = va;
            sb[e] = vb;
        }
        workgroupBarrier();
        var tmp: array<vec4<f32>, 4>;
        for (var i = 0u; i < 4u; i = i + 1u) { tmp[i] = vec4<f32>(0.0); }
        for (var k = 0u; k < 16u; k = k + 1u) {
            let ao = k * 64u + lid.y * 4u;
            let bo = k * 64u + lid.x * 4u;
            let bv = vec4<f32>(sb[bo], sb[bo + 1u], sb[bo + 2u], sb[bo + 3u]);
            tmp[0] = tmp[0] + sa[ao] * bv;
            tmp[1] = tmp[1] + sa[ao + 1u] * bv;
            tmp[2] = tmp[2] + sa[ao + 2u] * bv;
            tmp[3] = tmp[3] + sa[ao + 3u] * bv;
        }
        for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = acc[i] + tmp[i]; }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let col = cbase + lid.x * 4u + j;
                if (col < d) {
                    let idx = lr * d + col;
                    h[idx] = h[idx] + acc[i][j];
                }
            }
        }
    }
}
