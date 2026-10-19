// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 2048>;
var<workgroup> wb: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    let lx = lid.x;
    let ly = lid.y;
    var acc = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let it = idx >> 6u;
            let cc = idx & 63u;
            let item = t + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                if (cb + cc < cfg.d) {
                    va = phi[item * cfg.d + cb + cc];
                }
                if (rb + cc < cfg.r_rows && cfg.r_off + rb + cc < cfg.d) {
                    vb = phi[item * cfg.d + cfg.r_off + rb + cc];
                }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();
        var p = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(wa[o + lx], wa[o + lx + 16u], wa[o + lx + 32u], wa[o + lx + 48u]);
            p[0] = p[0] + a * wb[o + ly];
            p[1] = p[1] + a * wb[o + ly + 16u];
            p[2] = p[2] + a * wb[o + ly + 32u];
            p[3] = p[3] + a * wb[o + ly + 48u];
        }
        acc[0] = acc[0] + p[0];
        acc[1] = acc[1] + p[1];
        acc[2] = acc[2] + p[2];
        acc[3] = acc[3] + p[3];
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rb + ly + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + lx + 16u * j;
                if (c < cfg.d) {
                    let ix = lr * cfg.d + c;
                    h[ix] = h[ix] + acc[i][j];
                }
            }
        }
    }
}
