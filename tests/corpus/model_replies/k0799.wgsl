// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0);
    }
    let lin = lid.y * 8u + lid.x;
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        let base = ((t >> 7u) & 1u) * 8u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i2 = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i4 = lin + 128u * k2;
                let item = i2 + (i4 >> 4u);
                let cv = (i4 & 15u) * 4u;
                let cb = wid.x * 64u + cv;
                let rb = wid.y * 64u + cv;
                let i7 = item * cfg.d + cb;
                let j7 = item * cfg.d + cfg.r_off + rb;
                var f0: f32 = 0.0;
                var f1: f32 = 0.0;
                var f2: f32 = 0.0;
                var f3: f32 = 0.0;
                var f4: f32 = 0.0;
                var f5: f32 = 0.0;
                var f6: f32 = 0.0;
                var f7: f32 = 0.0;
                if (item < cfg.m) {
                    if (cb < cfg.d) { f0 = phi[i7]; }
                    if (cb + 1u < cfg.d) { f1 = phi[i7 + 1u]; }
                    if (cb + 2u < cfg.d) { f2 = phi[i7 + 2u]; }
                    if (cb + 3u < cfg.d) { f3 = phi[i7 + 3u]; }
                    if (rb < cfg.r_rows) {
                        if (cfg.r_off + rb < cfg.d) { f4 = phi[j7]; }
                    }
                    if (rb + 1u < cfg.r_rows) {
                        if (cfg.r_off + rb + 1u < cfg.d) { f5 = phi[j7 + 1u]; }
                    }
                    if (rb + 2u < cfg.r_rows) {
                        if (cfg.r_off + rb + 2u < cfg.d) { f6 = phi[j7 + 2u]; }
                    }
                    if (rb + 3u < cfg.r_rows) {
                        if (cfg.r_off + rb + 3u < cfg.d) { f7 = phi[j7 + 3u]; }
                    }
                }
                wg_va[i4 & 511u] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[i4 & 511u] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            var a0: vec4<f32> = vec4<f32>(0.0);
            var a1: vec4<f32> = vec4<f32>(0.0);
            var a2: vec4<f32> = vec4<f32>(0.0);
            var a3: vec4<f32> = vec4<f32>(0.0);
            var a4: vec4<f32> = vec4<f32>(0.0);
            var a5: vec4<f32> = vec4<f32>(0.0);
            var a6: vec4<f32> = vec4<f32>(0.0);
            var a7: vec4<f32> = vec4<f32>(0.0);
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let i8 = q * 32u;
                let ca0 = wg_va[(i8 + 2u * lid.x) & 511u];
                let ca1 = wg_va[(i8 + 2u * lid.x + 1u) & 511u];
                let cb0 = wg_va[(i8 + 16u + 2u * lid.x) & 511u];
                let cb1 = wg_va[(i8 + 17u + 2u * lid.x) & 511u];
                let w0 = wg_vb[(i8 + lid.y) & 511u];
                let w1 = wg_vb[(i8 + 16u + lid.y) & 511u];
                a0 = a0 + (vec4<f32>(w0.x) * ca0 + vec4<f32>(w1.x) * cb0);
                a1 = a1 + (vec4<f32>(w0.x) * ca1 + vec4<f32>(w1.x) * cb1);
                a2 = a2 + (vec4<f32>(w0.y) * ca0 + vec4<f32>(w1.y) * cb0);
                a3 = a3 + (vec4<f32>(w0.y) * ca1 + vec4<f32>(w1.y) * cb1);
                a4 = a4 + (vec4<f32>(w0.z) * ca0 + vec4<f32>(w1.z) * cb0);
                a5 = a5 + (vec4<f32>(w0.z) * ca1 + vec4<f32>(w1.z) * cb1);
                a6 = a6 + (vec4<f32>(w0.w) * ca0 + vec4<f32>(w1.w) * cb0);
                a7 = a7 + (vec4<f32>(w0.w) * ca1 + vec4<f32>(w1.w) * cb1);
            }
            pv[base & 15u] = pv[base & 15u] + a0;
            pv[(base + 1u) & 15u] = pv[(base + 1u) & 15u] + a1;
            pv[(base + 2u) & 15u] = pv[(base + 2u) & 15u] + a2;
            pv[(base + 3u) & 15u] = pv[(base + 3u) & 15u] + a3;
            pv[(base + 4u) & 15u] = pv[(base + 4u) & 15u] + a4;
            pv[(base + 5u) & 15u] = pv[(base + 5u) & 15u] + a5;
            pv[(base + 6u) & 15u] = pv[(base + 6u) & 15u] + a6;
            pv[(base + 7u) & 15u] = pv[(base + 7u) & 15u] + a7;
            workgroupBarrier();
        }
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = gid.y * 4u + k;
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let sv = pv[(k * 2u + j) & 15u] + pv[(k * 2u + j + 8u) & 15u];
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                let col = gid.x * 8u + j * 4u + c;
                if (row < cfg.r_rows) {
                    if (cfg.r_off + row < cfg.d) {
                        if (col < cfg.d) {
                            h[row * cfg.d + col] = h[row * cfg.d + col] + sv[c & 3u];
                        }
                    }
                }
            }
        }
    }
}
