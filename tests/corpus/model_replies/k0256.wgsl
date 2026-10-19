// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[(k & 63u)] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            f0 = 0.0;
            f1 = 0.0;
            let li = (lid.y * 16u + lid.x) + k * 256u;
            let tt = t + (li >> 6u);
            let cc = li & 63u;
            if (tt < cfg.m) {
                if (((wid.y * 64u + cc) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u + cc)) < cfg.d)) {
                    f0 = phi[tt * cfg.d + (cfg.r_off + (wid.y * 64u + cc))];
                }
                if ((wid.x * 64u + cc) < cfg.d) {
                    f1 = phi[tt * cfg.d + (wid.x * 64u + cc)];
                }
            }
            wg_a[li & 2047u] = f0;
            wg_b[li & 2047u] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_a[((k * 64u + lid.y)) & 2047u];
            let a1 = wg_a[((k * 64u + lid.y + 16u)) & 2047u];
            let a2 = wg_a[((k * 64u + lid.y + 32u)) & 2047u];
            let a3 = wg_a[((k * 64u + lid.y + 48u)) & 2047u];
            let b0 = wg_b[((k * 64u + lid.x)) & 2047u];
            let b1 = wg_b[((k * 64u + lid.x + 16u)) & 2047u];
            let b2 = wg_b[((k * 64u + lid.x + 32u)) & 2047u];
            let b3 = wg_b[((k * 64u + lid.x + 48u)) & 2047u];
            pa[0u] = pa[0u] + a0 * b0;
            pa[1u] = pa[1u] + a0 * b1;
            pa[2u] = pa[2u] + a0 * b2;
            pa[3u] = pa[3u] + a0 * b3;
            pa[4u] = pa[4u] + a1 * b0;
            pa[5u] = pa[5u] + a1 * b1;
            pa[6u] = pa[6u] + a1 * b2;
            pa[7u] = pa[7u] + a1 * b3;
            pa[8u] = pa[8u] + a2 * b0;
            pa[9u] = pa[9u] + a2 * b1;
            pa[10u] = pa[10u] + a2 * b2;
            pa[11u] = pa[11u] + a2 * b3;
            pa[12u] = pa[12u] + a3 * b0;
            pa[13u] = pa[13u] + a3 * b1;
            pa[14u] = pa[14u] + a3 * b2;
            pa[15u] = pa[15u] + a3 * b3;
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 64u + lid.y + k * 16u;
        if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x + k2 * 16u;
                if (col < cfg.d) {
                    h[row * cfg.d + col] = h[row * cfg.d + col] + pa[((k * 4u + k2) & 63u)];
                }
            }
        }
    }
}
