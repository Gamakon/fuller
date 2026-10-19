// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 6u;
            let cc = e & 63u;
            let t = t0 + it;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (t < cfg.m) {
                if (col0 + cc < d) {
                    a = phi[t * d + col0 + cc];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    b = phi[t * d + cfg.r_off + lr];
                }
            }
            wg_a[e & 1023u] = a;
            wg_b[e & 1023u] = b;
        }
        workgroupBarrier();
        for (var it: u32 = 0u; it < 16u; it = it + 1u) {
            let base = it * 64u;
            let ca = vec4<f32>(
                wg_a[(base + lid.x) & 1023u],
                wg_a[(base + lid.x + 16u) & 1023u],
                wg_a[(base + lid.x + 32u) & 1023u],
                wg_a[(base + lid.x + 48u) & 1023u]);
            let r0 = wg_b[(base + lid.y) & 1023u];
            let r1 = wg_b[(base + lid.y + 16u) & 1023u];
            let r2 = wg_b[(base + lid.y + 32u) & 1023u];
            let r3 = wg_b[(base + lid.y + 48u) & 1023u];
            acc0 = acc0 + r0 * ca;
            acc1 = acc1 + r1 * ca;
            acc2 = acc2 + r2 * ca;
            acc3 = acc3 + r3 * ca;
        }
        workgroupBarrier();
    }
    var accs: array<vec4<f32>, 4>;
    accs[0] = acc0;
    accs[1] = acc1;
    accs[2] = acc2;
    accs[3] = acc3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = row0 + lid.y + k * 16u;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = accs[k & 3u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + j * 16u;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j & 3u];
                }
            }
        }
    }
}
