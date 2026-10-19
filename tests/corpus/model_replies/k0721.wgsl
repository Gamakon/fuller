// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0: vec4<f32> = vec4<f32>(0.0); var e1: vec4<f32> = vec4<f32>(0.0); var e2: vec4<f32> = vec4<f32>(0.0); var e3: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0); var o1: vec4<f32> = vec4<f32>(0.0); var o2: vec4<f32> = vec4<f32>(0.0); var o3: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var k: u32 = 0u; var q: u32 = 0u; var j: u32 = 0u;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (q = 0u; q < 8u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e >> 6u;
            let cc = e & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + kk < cfg.m) {
                let base = (t + kk) * cfg.d;
                if (colbase + cc < cfg.d) {
                    va = phi[base + colbase + cc];
                }
                if (rowbase + cc < cfg.r_rows && cfg.r_off + rowbase + cc < cfg.d) {
                    vb = phi[base + cfg.r_off + rowbase + cc];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        var a0: vec4<f32> = vec4<f32>(0.0); var a1: vec4<f32> = vec4<f32>(0.0); var a2: vec4<f32> = vec4<f32>(0.0); var a3: vec4<f32> = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 1u) {
            let ba = k * 64u + lid.x;
            let bb = k * 64u + lid.y;
            let av = vec4<f32>(wg_a[ba], wg_a[ba + 16u], wg_a[ba + 32u], wg_a[ba + 48u]);
            a0 = a0 + vec4<f32>(wg_b[bb]) * av;
            a1 = a1 + vec4<f32>(wg_b[bb + 16u]) * av;
            a2 = a2 + vec4<f32>(wg_b[bb + 32u]) * av;
            a3 = a3 + vec4<f32>(wg_b[bb + 48u]) * av;
        }
        if (((t >> 5u) & 1u) == 0u) {
            e0 = e0 + a0; e1 = e1 + a1; e2 = e2 + a2; e3 = e3 + a3;
        } else {
            o0 = o0 + a0; o1 = o1 + a1; o2 = o2 + a2; o3 = o3 + a3;
        }
        workgroupBarrier();
    }
    pv[0u] = e0 + o0;
    pv[1u] = e1 + o1;
    pv[2u] = e2 + o2;
    pv[3u] = e3 + o3;
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = rowbase + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let c = colbase + lid.x + 16u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k][j];
                }
            }
        }
    }
}
