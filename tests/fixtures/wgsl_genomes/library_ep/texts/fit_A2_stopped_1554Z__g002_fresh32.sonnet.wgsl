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
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = cb + cc;
                if (col < d) {
                    va = phi[t * d + col];
                }
                let lr = rb + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[t * d + gr];
                }
            }
            sa[idx] = va;
            sb[idx] = vb;
        }
        workgroupBarrier();
        var tmp: array<vec4<f32>, 4>;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            tmp[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let cv = vec4<f32>(sa[o + lid.x], sa[o + lid.x + 16u], sa[o + lid.x + 32u], sa[o + lid.x + 48u]);
            tmp[0] = tmp[0] + vec4<f32>(sb[o + lid.y]) * cv;
            tmp[1] = tmp[1] + vec4<f32>(sb[o + lid.y + 16u]) * cv;
            tmp[2] = tmp[2] + vec4<f32>(sb[o + lid.y + 32u]) * cv;
            tmp[3] = tmp[3] + vec4<f32>(sb[o + lid.y + 48u]) * cv;
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            acc[i] = acc[i] + tmp[i];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rb + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i][j];
                }
            }
        }
    }
}
