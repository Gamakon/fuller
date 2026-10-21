// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let it = t + (idx >> 6u);
            let off = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (it < cfg.m) {
                let lr = row0 + off;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) {
                    fa = phi[it * cfg.d + gr];
                }
                let c = col0 + off;
                if (c < cfg.d) {
                    fb = phi[it * cfg.d + c];
                }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let ba = kk * 64u + lid.y;
            let bb = kk * 64u + lid.x;
            let b = vec4<f32>(wg_b[bb], wg_b[bb + 16u], wg_b[bb + 32u], wg_b[bb + 48u]);
            v0 = v0 + wg_a[ba] * b;
            v1 = v1 + wg_a[ba + 16u] * b;
            v2 = v2 + wg_a[ba + 32u] * b;
            v3 = v3 + wg_a[ba + 48u] * b;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            var acc: vec4<f32> = v0;
            if (i == 1u) { acc = v1; }
            if (i == 2u) { acc = v2; }
            if (i == 3u) { acc = v3; }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + acc[j];
                }
            }
        }
    }
}
