// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.y * 16u + lid.x;
    var e0 = vec4<f32>(0.0);
    var e1 = vec4<f32>(0.0);
    var e2 = vec4<f32>(0.0);
    var e3 = vec4<f32>(0.0);
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    let ax = lid.x * 4u;
    let by = lid.y * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = li + k * 256u;
            let item = t + (idx >> 6u);
            let col = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < cfg.m) {
                let gc = wid.x * 64u + col;
                if (gc < cfg.d) {
                    f0 = phi[item * cfg.d + gc];
                }
                let lr = wid.y * 64u + col;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    f1 = phi[item * cfg.d + cfg.r_off + lr];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
            let k0 = hh * 4u;
            let p0 = k0 * 64u;
            let p1 = p0 + 64u;
            let p2 = p0 + 128u;
            let p3 = p0 + 192u;
            let a0 = vec4<f32>(wg_a[p0 + ax], wg_a[p0 + ax + 1u], wg_a[p0 + ax + 2u], wg_a[p0 + ax + 3u]);
            let a1 = vec4<f32>(wg_a[p1 + ax], wg_a[p1 + ax + 1u], wg_a[p1 + ax + 2u], wg_a[p1 + ax + 3u]);
            let a2 = vec4<f32>(wg_a[p2 + ax], wg_a[p2 + ax + 1u], wg_a[p2 + ax + 2u], wg_a[p2 + ax + 3u]);
            let a3 = vec4<f32>(wg_a[p3 + ax], wg_a[p3 + ax + 1u], wg_a[p3 + ax + 2u], wg_a[p3 + ax + 3u]);
            let b0 = vec4<f32>(wg_b[p0 + by], wg_b[p0 + by + 1u], wg_b[p0 + by + 2u], wg_b[p0 + by + 3u]);
            let b1 = vec4<f32>(wg_b[p1 + by], wg_b[p1 + by + 1u], wg_b[p1 + by + 2u], wg_b[p1 + by + 3u]);
            let b2 = vec4<f32>(wg_b[p2 + by], wg_b[p2 + by + 1u], wg_b[p2 + by + 2u], wg_b[p2 + by + 3u]);
            let b3 = vec4<f32>(wg_b[p3 + by], wg_b[p3 + by + 1u], wg_b[p3 + by + 2u], wg_b[p3 + by + 3u]);
            let t0 = (a0 * b0.x + a1 * b1.x) + (a2 * b2.x + a3 * b3.x);
            let t1 = (a0 * b0.y + a1 * b1.y) + (a2 * b2.y + a3 * b3.y);
            let t2 = (a0 * b0.z + a1 * b1.z) + (a2 * b2.z + a3 * b3.z);
            let t3 = (a0 * b0.w + a1 * b1.w) + (a2 * b2.w + a3 * b3.w);
            if (hh == 0u) {
                e0 = e0 + t0;
                e1 = e1 + t1;
                e2 = e2 + t2;
                e3 = e3 + t3;
            } else {
                o0 = o0 + t0;
                o1 = o1 + t1;
                o2 = o2 + t2;
                o3 = o3 + t3;
            }
        }
        workgroupBarrier();
    }
    var s = array<vec4<f32>, 4>(e0 + o0, e1 + o1, e2 + o2, e3 + o3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = gid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let sv = s[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < cfg.d) {
                    let ix = lr * cfg.d + c;
                    h[ix] = h[ix] + sv[j];
                }
            }
        }
    }
}
