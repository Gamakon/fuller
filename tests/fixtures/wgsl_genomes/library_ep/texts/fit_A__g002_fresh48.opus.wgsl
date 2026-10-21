// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;

fn wrow(r: u32, c0: u32, v: vec4<f32>) {
    if (r < cfg.r_rows && cfg.r_off + r < cfg.d) {
        let base = r * cfg.d;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            if (c < cfg.d) {
                h[base + c] = h[base + c] + v[j];
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let colb = wid.x * 64u;
    let rowb = wid.y * 64u;
    var t0 = vec4<f32>(0.0); var t1 = vec4<f32>(0.0); var t2 = vec4<f32>(0.0); var t3 = vec4<f32>(0.0);
    var m0 = vec4<f32>(0.0); var m1 = vec4<f32>(0.0); var m2 = vec4<f32>(0.0); var m3 = vec4<f32>(0.0);
    var n: u32 = 0u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var p: u32 = 0u; p < 2u; p = p + 1u) {
            let vi = lidx + 256u * p;
            let kk = vi >> 4u;
            let c4 = (vi & 15u) * 4u;
            let item = t + kk;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colb + c4 + j;
                    if (c < cfg.d) { va[j] = phi[base + c]; }
                    let lr = rowb + c4 + j;
                    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) { vb[j] = phi[base + cfg.r_off + lr]; }
                }
            }
            wg_a[vi] = va;
            wg_b[vi] = vb;
        }
        workgroupBarrier();
        var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_a[k * 16u + lid.x];
            let b = wg_b[k * 16u + lid.y];
            c0 = fma(vec4<f32>(b.x), a, c0);
            c1 = fma(vec4<f32>(b.y), a, c1);
            c2 = fma(vec4<f32>(b.z), a, c2);
            c3 = fma(vec4<f32>(b.w), a, c3);
        }
        m0 = m0 + c0; m1 = m1 + c1; m2 = m2 + c2; m3 = m3 + c3;
        n = n + 1u;
        if (n >= 16u) {
            t0 = t0 + m0; t1 = t1 + m1; t2 = t2 + m2; t3 = t3 + m3;
            m0 = vec4<f32>(0.0); m1 = vec4<f32>(0.0); m2 = vec4<f32>(0.0); m3 = vec4<f32>(0.0);
            n = 0u;
        }
        workgroupBarrier();
    }
    t0 = t0 + m0; t1 = t1 + m1; t2 = t2 + m2; t3 = t3 + m3;
    let r0 = rowb + lid.y * 4u;
    let cc = colb + lid.x * 4u;
    wrow(r0, cc, t0);
    wrow(r0 + 1u, cc, t1);
    wrow(r0 + 2u, cc, t2);
    wrow(r0 + 3u, cc, t3);
}
