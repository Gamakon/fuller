// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0: vec4<f32> = vec4<f32>(0.0);
    var a1: vec4<f32> = vec4<f32>(0.0);
    var a2: vec4<f32> = vec4<f32>(0.0);
    var a3: vec4<f32> = vec4<f32>(0.0);
    var a4: vec4<f32> = vec4<f32>(0.0);
    var a5: vec4<f32> = vec4<f32>(0.0);
    var a6: vec4<f32> = vec4<f32>(0.0);
    var a7: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let e = lidx + 128u * k;
            let it = t + (e >> 6u);
            let cc = e & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (it < m) {
                if (colbase + cc < d) {
                    f0 = phi[it * d + colbase + cc];
                }
                if (rowbase + cc < cfg.r_rows && cfg.r_off + rowbase + cc < d) {
                    f1 = phi[it * d + cfg.r_off + rowbase + cc];
                }
            }
            wg_a[e & 2047u] = f0;
            wg_b[e & 2047u] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let base = k * 64u;
            let ia = base + lid.x;
            let av = vec4<f32>(wg_a[ia & 2047u], wg_a[(ia + 16u) & 2047u], wg_a[(ia + 32u) & 2047u], wg_a[(ia + 48u) & 2047u]);
            let ib = base + lid.y;
            a0 = a0 + vec4<f32>(wg_b[ib & 2047u]) * av;
            a1 = a1 + vec4<f32>(wg_b[(ib + 8u) & 2047u]) * av;
            a2 = a2 + vec4<f32>(wg_b[(ib + 16u) & 2047u]) * av;
            a3 = a3 + vec4<f32>(wg_b[(ib + 24u) & 2047u]) * av;
            a4 = a4 + vec4<f32>(wg_b[(ib + 32u) & 2047u]) * av;
            a5 = a5 + vec4<f32>(wg_b[(ib + 40u) & 2047u]) * av;
            a6 = a6 + vec4<f32>(wg_b[(ib + 48u) & 2047u]) * av;
            a7 = a7 + vec4<f32>(wg_b[(ib + 56u) & 2047u]) * av;
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 8>;
    pv[0u] = a0;
    pv[1u] = a1;
    pv[2u] = a2;
    pv[3u] = a3;
    pv[4u] = a4;
    pv[5u] = a5;
    pv[6u] = a6;
    pv[7u] = a7;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = rowbase + lid.y + 8u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + pv[i & 7u][j & 3u];
                }
            }
        }
    }
}
