// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s: array<vec4<f32>, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        s[q] = vec4<f32>(0.0);
    }
    let nch = (cfg.m + 31u) / 32u;
    for (var t0: u32 = 0u; t0 < nch; t0 = t0 + 1u) {
        let base = t0 * 32u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let it = e >> 4u;
            let qq = e & 15u;
            let item = base + it;
            let rowb = item * cfg.d;
            let gc = wid.x * 64u + qq * 4u;
            var va: vec4<f32> = vec4<f32>(0.0);
            if (item < cfg.m) {
                if (gc < cfg.d) { va.x = phi[rowb + gc]; }
                if (gc + 1u < cfg.d) { va.y = phi[rowb + gc + 1u]; }
                if (gc + 2u < cfg.d) { va.z = phi[rowb + gc + 2u]; }
                if (gc + 3u < cfg.d) { va.w = phi[rowb + gc + 3u]; }
            }
            wg_a[e] = va;
            let lrw = wid.y * 64u + qq * 4u;
            let gr = cfg.r_off + lrw;
            var vb: vec4<f32> = vec4<f32>(0.0);
            if (item < cfg.m) {
                if (lrw < cfg.r_rows && gr < cfg.d) { vb.x = phi[rowb + gr]; }
                if (lrw + 1u < cfg.r_rows && gr + 1u < cfg.d) { vb.y = phi[rowb + gr + 1u]; }
                if (lrw + 2u < cfg.r_rows && gr + 2u < cfg.d) { vb.z = phi[rowb + gr + 2u]; }
                if (lrw + 3u < cfg.r_rows && gr + 3u < cfg.d) { vb.w = phi[rowb + gr + 3u]; }
            }
            wg_b[e] = vb;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0);
        var p1: vec4<f32> = vec4<f32>(0.0);
        var p2: vec4<f32> = vec4<f32>(0.0);
        var p3: vec4<f32> = vec4<f32>(0.0);
        var p4: vec4<f32> = vec4<f32>(0.0);
        var p5: vec4<f32> = vec4<f32>(0.0);
        var p6: vec4<f32> = vec4<f32>(0.0);
        var p7: vec4<f32> = vec4<f32>(0.0);
        for (var it: u32 = 0u; it < 32u; it = it + 1u) {
            let a0 = wg_a[it * 16u + lid.x * 2u];
            let a1 = wg_a[it * 16u + lid.x * 2u + 1u];
            let b = wg_b[it * 16u + lid.y];
            p0 = p0 + b.x * a0;
            p1 = p1 + b.x * a1;
            p2 = p2 + b.y * a0;
            p3 = p3 + b.y * a1;
            p4 = p4 + b.z * a0;
            p5 = p5 + b.z * a1;
            p6 = p6 + b.w * a0;
            p7 = p7 + b.w * a1;
        }
        s[0] = s[0] + p0;
        s[1] = s[1] + p1;
        s[2] = s[2] + p2;
        s[3] = s[3] + p3;
        s[4] = s[4] + p4;
        s[5] = s[5] + p5;
        s[6] = s[6] + p6;
        s[7] = s[7] + p7;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let c = wid.x * 64u + lid.x * 8u + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                let v = s[i * 2u + (j >> 2u)];
                h[idx] = h[idx] + v[j & 3u];
            }
        }
    }
}
