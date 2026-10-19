// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var p0: vec4<f32> = vec4<f32>(0.0);
    var p1: vec4<f32> = vec4<f32>(0.0);
    var p2: vec4<f32> = vec4<f32>(0.0);
    var p3: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let it = t + (idx >> 6u);
            let col = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (it < cfg.m) {
                if (((wid.x * 64u) + col) < cfg.d) {
                    f0 = phi[(it * cfg.d) + (wid.x * 64u) + col];
                }
                if ((((wid.y * 64u) + col) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + col) < cfg.d)) {
                    f1 = phi[(it * cfg.d) + cfg.r_off + (wid.y * 64u) + col];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        var c0: vec4<f32> = vec4<f32>(0.0);
        var c1: vec4<f32> = vec4<f32>(0.0);
        var c2: vec4<f32> = vec4<f32>(0.0);
        var c3: vec4<f32> = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let ba = (k * 64u) + (lid.x * 4u);
            let bb = (k * 64u) + (lid.y * 4u);
            let av = vec4<f32>(wg_a[ba], wg_a[ba + 1u], wg_a[ba + 2u], wg_a[ba + 3u]);
            let bv = vec4<f32>(wg_b[bb], wg_b[bb + 1u], wg_b[bb + 2u], wg_b[bb + 3u]);
            c0 = c0 + bv.x * av;
            c1 = c1 + bv.y * av;
            c2 = c2 + bv.z * av;
            c3 = c3 + bv.w * av;
        }
        p0 = p0 + c0;
        p1 = p1 + c1;
        p2 = p2 + c2;
        p3 = p3 + c3;
        workgroupBarrier();
    }
    pv[0] = p0;
    pv[1] = p1;
    pv[2] = p2;
    pv[3] = p3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        if ((((wid.y * 64u) + (lid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + (lid.y * 4u) + k) < cfg.d)) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((wid.x * 64u) + (lid.x * 4u) + k2) < cfg.d) {
                    let oi = (((wid.y * 64u) + (lid.y * 4u) + k) * cfg.d) + (wid.x * 64u) + (lid.x * 4u) + k2;
                    h[oi] = h[oi] + pv[k][k2];
                }
            }
        }
    }
}
