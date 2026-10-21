// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lx = lid.x;
    let ly = lid.y;
    let c0 = wid.x * 128u;
    let r0 = wid.y * 64u;
    var acc: array<vec4<f32>, 8>;
    var s: array<vec4<f32>, 8>;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        acc[j] = vec4<f32>(0.0);
        s[j] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let e = lidx + 256u * j;
            let item = t + (e >> 7u);
            let gc = c0 + (e & 127u);
            var val: f32 = 0.0;
            if (item < cfg.m && gc < cfg.d) {
                val = phi[item * cfg.d + gc];
            }
            wg_a[e] = val;
        }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let item = t + (e >> 6u);
            let lr = r0 + (e & 63u);
            var val: f32 = 0.0;
            if (item < cfg.m && lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                val = phi[item * cfg.d + cfg.r_off + lr];
            }
            wg_b[e] = val;
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            s[j] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = vec4<f32>(wg_a[k * 128u + lx], wg_a[k * 128u + lx + 32u], wg_a[k * 128u + lx + 64u], wg_a[k * 128u + lx + 96u]);
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let b = wg_b[k * 64u + ly + 8u * j];
                s[j] = s[j] + vec4<f32>(b) * a;
            }
        }
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            acc[j] = acc[j] + s[j];
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = r0 + ly + 8u * j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = c0 + lx + 32u * i;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[j][i];
                }
            }
        }
    }
}
