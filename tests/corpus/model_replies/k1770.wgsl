@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> sp: array<f32, 2112>;
var<workgroup> sc: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 64u;
    let local = base + lid;

    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var c0 = 0u; c0 < c; c0 += 16u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < dp; d0 += 32u) {
            for (var k = 0u; k < 32u; k += 1u) {
                let i = k * 64u + lid;
                let row = i / 32u;
                let col = i % 32u;
                var v = 0.0;
                if (base + row < n_band && d0 + col < dp) {
                    v = point_keys[(base + row) * dp + d0 + col];
                }
                sp[row * 33u + col] = v;
            }
            for (var k = 0u; k < 8u; k += 1u) {
                let i = k * 64u + lid;
                let row = i / 32u;
                let col = i % 32u;
                var v = 0.0;
                if (c0 + row < c && d0 + col < dp) {
                    v = centroid_keys[(c0 + row) * dp + d0 + col];
                }
                sc[i] = v;
            }
            workgroupBarrier();
            for (var d = 0u; d < 32u; d += 1u) {
                let p = sp[lid * 33u + d];
                a0 = a0 + p * vec4<f32>(sc[d], sc[32u + d], sc[64u + d], sc[96u + d]);
                a1 = a1 + p * vec4<f32>(sc[128u + d], sc[160u + d], sc[192u + d], sc[224u + d]);
                a2 = a2 + p * vec4<f32>(sc[256u + d], sc[288u + d], sc[320u + d], sc[352u + d]);
                a3 = a3 + p * vec4<f32>(sc[384u + d], sc[416u + d], sc[448u + d], sc[480u + d]);
            }
            workgroupBarrier();
        }
        for (var j = 0u; j < 16u; j += 1u) {
            var s: f32;
            let q = j / 4u;
            let r = j % 4u;
            if (q == 0u) { s = a0[r]; }
            else if (q == 1u) { s = a1[r]; }
            else if (q == 2u) { s = a2[r]; }
            else { s = a3[r]; }
            let ci = c0 + j;
            if (ci < c && s > best_sim) {
                best_sim = s;
                best = ci;
            }
        }
    }
    if (local < n_band) {
        assignments[row_off + local] = best;
    }
}
