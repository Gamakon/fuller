// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 4u);
            let cv = (idx & 15u) * 4u;
            let col = wid.x * 64u + cv;
            let row = wid.y * 64u + cv;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                if (col < cfg.d) { v0.x = phi[base + col]; }
                if (col + 1u < cfg.d) { v0.y = phi[base + col + 1u]; }
                if (col + 2u < cfg.d) { v0.z = phi[base + col + 2u]; }
                if (col + 3u < cfg.d) { v0.w = phi[base + col + 3u]; }
                let g = cfg.r_off + row;
                if (row < cfg.r_rows && g < cfg.d) { v1.x = phi[base + g]; }
                if (row + 1u < cfg.r_rows && g + 1u < cfg.d) { v1.y = phi[base + g + 1u]; }
                if (row + 2u < cfg.r_rows && g + 2u < cfg.d) { v1.z = phi[base + g + 2u]; }
                if (row + 3u < cfg.r_rows && g + 3u < cfg.d) { v1.w = phi[base + g + 3u]; }
            }
            wg_va[idx & 511u] = v0;
            wg_vb[idx & 511u] = v1;
        }
        workgroupBarrier();
        var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b = wg_vb[(k * 16u + lid.y) & 511u];
            c0 = c0 + vec4<f32>(b.x) * a;
            c1 = c1 + vec4<f32>(b.y) * a;
            c2 = c2 + vec4<f32>(b.z) * a;
            c3 = c3 + vec4<f32>(b.w) * a;
        }
        if (((t >> 5u) & 1u) == 0u) {
            a0 = a0 + c0; a1 = a1 + c1; a2 = a2 + c2; a3 = a3 + c3;
        } else {
            b0 = b0 + c0; b1 = b1 + c1; b2 = b2 + c2; b3 = b3 + c3;
        }
        workgroupBarrier();
    }
    pv[0u] = a0 + b0;
    pv[1u] = a1 + b1;
    pv[2u] = a2 + b2;
    pv[3u] = a3 + b3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let r = gid.y * 4u + k;
        if (r < cfg.r_rows && cfg.r_off + r < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[r * cfg.d + c] = h[r * cfg.d + c] + pv[k][k2];
                }
            }
        }
    }
}
