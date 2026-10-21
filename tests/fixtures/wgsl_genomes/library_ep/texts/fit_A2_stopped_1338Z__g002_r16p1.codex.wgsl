// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };

@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 4u;

    var q0: vec4<f32> = vec4<f32>(0.0);
    var q1: vec4<f32> = vec4<f32>(0.0);
    var q2: vec4<f32> = vec4<f32>(0.0);
    var q3: vec4<f32> = vec4<f32>(0.0);
    var q4: vec4<f32> = vec4<f32>(0.0);
    var q5: vec4<f32> = vec4<f32>(0.0);
    var q6: vec4<f32> = vec4<f32>(0.0);
    var q7: vec4<f32> = vec4<f32>(0.0);
    var a0: vec4<f32> = vec4<f32>(0.0);
    var a1: vec4<f32> = vec4<f32>(0.0);
    var a2: vec4<f32> = vec4<f32>(0.0);
    var a3: vec4<f32> = vec4<f32>(0.0);
    var b0: vec4<f32> = vec4<f32>(0.0);
    var b1: vec4<f32> = vec4<f32>(0.0);
    var b2: vec4<f32> = vec4<f32>(0.0);
    var b3: vec4<f32> = vec4<f32>(0.0);

    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let slot = lidx + k * 256u;
            let item = t + (slot >> 6u);
            let feature = slot & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;

            if (item < cfg.m) {
                if (wid.x * 64u + feature < cfg.d) {
                    va = phi[item * cfg.d + wid.x * 64u + feature];
                }
                if (wid.y * 64u + feature < cfg.r_rows) {
                    if (cfg.r_off + wid.y * 64u + feature < cfg.d) {
                        vb = phi[item * cfg.d + cfg.r_off + wid.y * 64u + feature];
                    }
                }
            }

            wg_a[slot] = va;
            wg_b[slot] = vb;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let base = k * 256u;
            let ax = base + lid.x * 4u;
            let by = base + lid.y * 4u;

            a0 = vec4<f32>(wg_a[ax], wg_a[ax + 1u], wg_a[ax + 2u], wg_a[ax + 3u]);
            a1 = vec4<f32>(wg_a[ax + 64u], wg_a[ax + 65u], wg_a[ax + 66u], wg_a[ax + 67u]);
            a2 = vec4<f32>(wg_a[ax + 128u], wg_a[ax + 129u], wg_a[ax + 130u], wg_a[ax + 131u]);
            a3 = vec4<f32>(wg_a[ax + 192u], wg_a[ax + 193u], wg_a[ax + 194u], wg_a[ax + 195u]);

            b0 = vec4<f32>(wg_b[by], wg_b[by + 1u], wg_b[by + 2u], wg_b[by + 3u]);
            b1 = vec4<f32>(wg_b[by + 64u], wg_b[by + 65u], wg_b[by + 66u], wg_b[by + 67u]);
            b2 = vec4<f32>(wg_b[by + 128u], wg_b[by + 129u], wg_b[by + 130u], wg_b[by + 131u]);
            b3 = vec4<f32>(wg_b[by + 192u], wg_b[by + 193u], wg_b[by + 194u], wg_b[by + 195u]);

            if (k == 0u) {
                q0 = q0 + a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x;
                q1 = q1 + a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y;
                q2 = q2 + a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z;
                q3 = q3 + a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w;
            } else {
                q4 = q4 + a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x;
                q5 = q5 + a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y;
                q6 = q6 + a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z;
                q7 = q7 + a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w;
            }
        }

        workgroupBarrier();
    }

    let s0 = q0 + q4;
    let s1 = q1 + q5;
    let s2 = q2 + q6;
    let s3 = q3 + q7;

    if (row0 < cfg.r_rows && cfg.r_off + row0 < cfg.d) {
        if (col0 < cfg.d) {
            h[row0 * cfg.d + col0] = h[row0 * cfg.d + col0] + s0.x;
        }
        if (col0 + 1u < cfg.d) {
            h[row0 * cfg.d + col0 + 1u] = h[row0 * cfg.d + col0 + 1u] + s0.y;
        }
        if (col0 + 2u < cfg.d) {
            h[row0 * cfg.d + col0 + 2u] = h[row0 * cfg.d + col0 + 2u] + s0.z;
        }
        if (col0 + 3u < cfg.d) {
            h[row0 * cfg.d + col0 + 3u] = h[row0 * cfg.d + col0 + 3u] + s0.w;
        }
    }
    if (row0 + 1u < cfg.r_rows && cfg.r_off + row0 + 1u < cfg.d) {
        if (col0 < cfg.d) {
            h[(row0 + 1u) * cfg.d + col0] = h[(row0 + 1u) * cfg.d + col0] + s1.x;
        }
        if (col0 + 1u < cfg.d) {
            h[(row0 + 1u) * cfg.d + col0 + 1u] = h[(row0 + 1u) * cfg.d + col0 + 1u] + s1.y;
        }
        if (col0 + 2u < cfg.d) {
            h[(row0 + 1u) * cfg.d + col0 + 2u] = h[(row0 + 1u) * cfg.d + col0 + 2u] + s1.z;
        }
        if (col0 + 3u < cfg.d) {
            h[(row0 + 1u) * cfg.d + col0 + 3u] = h[(row0 + 1u) * cfg.d + col0 + 3u] + s1.w;
        }
    }
    if (row0 + 2u < cfg.r_rows && cfg.r_off + row0 + 2u < cfg.d) {
        if (col0 < cfg.d) {
            h[(row0 + 2u) * cfg.d + col0] = h[(row0 + 2u) * cfg.d + col0] + s2.x;
        }
        if (col0 + 1u < cfg.d) {
            h[(row0 + 2u) * cfg.d + col0 + 1u] = h[(row0 + 2u) * cfg.d + col0 + 1u] + s2.y;
        }
        if (col0 + 2u < cfg.d) {
            h[(row0 + 2u) * cfg.d + col0 + 2u] = h[(row0 + 2u) * cfg.d + col0 + 2u] + s2.z;
        }
        if (col0 + 3u < cfg.d) {
            h[(row0 + 2u) * cfg.d + col0 + 3u] = h[(row0 + 2u) * cfg.d + col0 + 3u] + s2.w;
        }
    }
    if (row0 + 3u < cfg.r_rows && cfg.r_off + row0 + 3u < cfg.d) {
        if (col0 < cfg.d) {
            h[(row0 + 3u) * cfg.d + col0] = h[(row0 + 3u) * cfg.d + col0] + s3.x;
        }
        if (col0 + 1u < cfg.d) {
            h[(row0 + 3u) * cfg.d + col0 + 1u] = h[(row0 + 3u) * cfg.d + col0 + 1u] + s3.y;
        }
        if (col0 + 2u < cfg.d) {
            h[(row0 + 3u) * cfg.d + col0 + 2u] = h[(row0 + 3u) * cfg.d + col0 + 2u] + s3.z;
        }
        if (col0 + 3u < cfg.d) {
            h[(row0 + 3u) * cfg.d + col0 + 3u] = h[(row0 + 3u) * cfg.d + col0 + 3u] + s3.w;
        }
    }
}
